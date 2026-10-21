"""Causal semantic selection using class margins and time-aware visual evidence."""
from collections import deque

import torch
import torch.nn.functional as F

from semantic_cache import FixedSemanticCache


class RecentClassMemory:
    def __init__(self, classes, capacity=32, decay=.9, time_half_life=0.):
        if capacity < 1 or not 0 < decay <= 1 or time_half_life < 0:
            raise ValueError('Invalid memory capacity, decay or half life')
        self.entries = [deque(maxlen=capacity) for _ in range(classes)]
        self.decay = decay
        self.time_half_life = time_half_life

    def add(self, class_index, feature, reliability, step):
        if not 0 < reliability <= 1:
            raise ValueError('Reliability must be in (0, 1]')
        entries = self.entries[class_index]
        if entries and step <= entries[-1][2]:
            raise ValueError('Evidence must arrive in increasing stream order')
        entries.append((feature.detach().float().clone(), reliability, step))

    def evidence(self, class_index, step):
        entries = self.entries[class_index]
        if not entries:
            return None
        if step < entries[-1][2]:
            raise ValueError('Cannot read future evidence')
        features = torch.stack([item[0] for item in entries])
        if self.time_half_life:
            ages = features.new_tensor([step - item[2] for item in entries])
            decay = torch.exp2(-ages / self.time_half_life)
        else:
            ages = features.new_tensor(list(reversed(range(len(entries)))))
            decay = self.decay ** ages
        quality = features.new_tensor([item[1] for item in entries])
        return features, quality * decay

    def anchor(self, class_index, step):
        evidence = self.evidence(class_index, step)
        if evidence is None:
            return None
        features, weights = evidence
        mean = (features * weights[:, None]).sum(0) / weights.sum().clamp_min(1e-12)
        return F.normalize(mean, dim=0)


@torch.no_grad()
def select_descriptions(candidates, positive, positive_weights, competitors,
                        negatives, similarities, k, discrimination_weight=1.,
                        redundancy_weight=.1):
    """Evaluate the normalized aggregate after each greedy addition.

    Negative entries contain features, weights, own fixed prototype and freshness.
    Class-balanced logistic margin losses use CLIP's existing 100x cosine scale.
    """
    if not 1 <= k <= len(candidates) or not 0 <= discrimination_weight <= 1:
        raise ValueError('Invalid selection size or discrimination weight')
    candidates = candidates.float()
    pos_weights = positive_weights / positive_weights.sum().clamp_min(1e-12)
    positive = positive.float()
    reference = (positive @ competitors.float()).max(1).values
    selected, selected_scores = [], []
    total = torch.zeros_like(candidates[0])
    redundancy = candidates.new_zeros(len(candidates))
    negative_mass = sum(item[3] for item in negatives)
    for _ in range(k):
        proposed = F.normalize(total[None, :] + candidates, dim=1)
        pos_scores = positive @ proposed.T
        relevance = (pos_weights[:, None] * pos_scores).sum(0)
        positive_margin = pos_scores - reference[:, None]
        positive_loss = (pos_weights[:, None] * F.softplus(-100. * positive_margin) / 100.).sum(0)
        negative_loss = torch.zeros_like(positive_loss)
        for features, weights, own_prototype, freshness in negatives:
            own_scores = features @ own_prototype.float()
            margins = own_scores[:, None] - features @ proposed.T
            per_class = ((weights / weights.sum().clamp_min(1e-12))[:, None]
                         * F.softplus(-100. * margins) / 100.).sum(0)
            negative_loss += freshness * per_class
        # Do not normalize away stale classes' low absolute reliability.
        margin_score = -(positive_loss + negative_loss) / (1. + negative_mass)
        scores = ((1. - discrimination_weight) * relevance
                  + discrimination_weight * margin_score - redundancy_weight * redundancy)
        scores[selected] = -torch.inf
        index = int(scores.argmax())
        selected.append(index)
        selected_scores.append(float(scores[index]))
        total += candidates[index]
        redundancy = torch.maximum(redundancy, similarities[:, index].clamp_min(0))
    return selected, selected_scores


class RDSSCache:
    def __init__(self, embeddings, classnames, k=8, memory_capacity=32, decay=.9,
                 entropy_threshold=.2, discrimination_weight=1., redundancy_weight=.1,
                 time_half_life_per_class=4., prior_strength=4.):
        if not 0 <= entropy_threshold < 1 or prior_strength < 0 or len(classnames) < 2:
            raise ValueError('Invalid entropy threshold, prior strength or class count')
        if not 0 <= discrimination_weight <= 1 or redundancy_weight < 0:
            raise ValueError('Invalid selection weights')
        self.candidates = [embeddings[n] for n in classnames]
        if any(not 1 <= k <= len(c) for c in self.candidates):
            raise ValueError('Every class must contain at least k descriptions')
        self.fixed = FixedSemanticCache(embeddings, classnames).prototypes.clone()
        self.prototypes = self.fixed.clone()
        self.similarities = [c.float() @ c.float().T for c in self.candidates]
        self.memory = RecentClassMemory(len(classnames), memory_capacity, decay,
                                        time_half_life_per_class * len(classnames))
        self.k = k
        self.entropy_threshold = entropy_threshold
        self.discrimination_weight = discrimination_weight
        self.redundancy_weight = redundancy_weight
        self.prior_strength = prior_strength
        self.selected = [None for _ in classnames]
        self.accepted = [0 for _ in classnames]
        self.last_step = -1
        self.evidence_mass = self.fixed.new_zeros(len(classnames), dtype=torch.float32)
        self.evidence_concentration = torch.zeros_like(self.evidence_mass)
        self.updated_at = torch.zeros_like(self.evidence_mass)

    def adaptation_weights(self, step):
        mass = self.evidence_mass
        if self.memory.time_half_life:
            mass = mass * torch.exp2(-(step - self.updated_at) / self.memory.time_half_life)
        if self.prior_strength == 0:
            return (self.evidence_mass > 0).float()
        return self.evidence_concentration * mass / (mass + self.prior_strength)

    def adaptation_weight(self, class_index, step):
        return self.adaptation_weights(step)[class_index]

    @torch.no_grad()
    def read_prototypes(self, step):
        if step < self.last_step:
            raise ValueError('Cannot read a past state')
        alpha = self.adaptation_weights(step)[None, :]
        mean = (1. - alpha) * self.fixed.float() + alpha * self.prototypes.float()
        result = F.normalize(mean, dim=0).to(self.fixed.dtype)
        # Preserve exact full-pool cold-start values, including float16 rounding.
        return torch.where(alpha > 0, result, self.fixed)

    @torch.no_grad()
    def logits(self, image_features, step):
        """Read history only; the current query does not mutate the cache."""
        return 100. * image_features @ self.read_prototypes(step)

    @torch.no_grad()
    def observe(self, feature, class_index, entropy, consistent, hard_classes, step):
        """Called strictly after prediction. Labels and domain IDs are not inputs."""
        if step <= self.last_step:
            raise ValueError('Observations must follow stream order')
        self.last_step = step
        if not consistent or not 0 <= entropy <= self.entropy_threshold:
            return None
        self.memory.add(class_index, feature, 1. - entropy, step)
        self.accepted[class_index] += 1
        positive, weights = self.memory.evidence(class_index, step)
        mass = weights.sum()
        mean_evidence = (positive * weights[:, None]).sum(0) / mass.clamp_min(1e-12)
        self.evidence_mass[class_index] = mass
        self.evidence_concentration[class_index] = mean_evidence.norm().clamp(0, 1)
        self.updated_at[class_index] = step
        # Fixed full-pool competitors cannot co-adapt to the selected sentences.
        hard = list(dict.fromkeys(c for c in hard_classes if c != class_index))
        competitors = hard or [c for c in range(len(self.candidates)) if c != class_index]
        negatives, used_negatives = [], []
        for other in hard:
            evidence = self.memory.evidence(other, step)
            if evidence is None:
                continue
            vectors, other_weights = evidence
            last_seen = self.memory.entries[other][-1][2]
            freshness = (2. ** (-(step - last_seen) / self.memory.time_half_life)
                         if self.memory.time_half_life else 1.)
            negatives.append((vectors, other_weights, self.fixed[:, other], freshness))
            used_negatives.append(other)
        candidates = self.candidates[class_index]
        indices, scores = select_descriptions(
            candidates, positive, weights, self.fixed[:, competitors], negatives,
            self.similarities[class_index], self.k, self.discrimination_weight,
            self.redundancy_weight)
        mean = candidates[indices].float().mean(0)
        prototype = F.normalize(mean, dim=0).to(self.prototypes.dtype)
        changed = self.selected[class_index] != indices
        self.prototypes[:, class_index] = prototype
        self.selected[class_index] = indices
        return {'step': step, 'class_index': class_index, 'entropy': entropy,
                'accepted_count': self.accepted[class_index],
                'recent_count': len(self.memory.entries[class_index]),
                'hard_classes': hard, 'negative_classes_with_evidence': used_negatives,
                'negative_freshness': [item[3] for item in negatives],
                'effective_mass': float(weights.sum()),
                'adaptation_weight': float(self.adaptation_weight(class_index, step)),
                'selected_indices': indices, 'selection_changed': changed,
                'greedy_scores': scores}
