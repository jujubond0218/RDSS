"""Reliable, discriminative and diverse semantic selection from recent evidence."""
from collections import deque

import torch

from semantic_cache import FixedSemanticCache


@torch.no_grad()
def select_descriptions(relevance, discrimination, similarities, k,
                        discrimination_weight=1., redundancy_weight=.1):
    if not 1 <= k <= len(relevance):
        raise ValueError('k must fit the candidate pool')
    base = relevance + discrimination_weight * discrimination
    redundancy = torch.zeros_like(base)
    selected, selected_scores = [], []
    for _ in range(k):
        scores = base - redundancy_weight * redundancy
        if selected:
            scores[selected] = -torch.inf
        index = int(scores.argmax())
        selected.append(index)
        selected_scores.append(float(scores[index]))
        redundancy = torch.maximum(redundancy, similarities[:, index].clamp_min(0))
    return selected, selected_scores


class RecentClassMemory:
    def __init__(self, classes, capacity=32, decay=.9):
        if capacity < 1 or not 0 < decay <= 1:
            raise ValueError('Memory capacity must be positive and decay in (0, 1]')
        self.entries = [deque(maxlen=capacity) for _ in range(classes)]
        self.decay = decay

    def add(self, class_index, feature, reliability):
        self.entries[class_index].append((feature.detach().clone(), reliability))

    @torch.no_grad()
    def anchor(self, class_index):
        entries = self.entries[class_index]
        if not entries:
            return None
        features = torch.stack([item[0] for item in entries]).float()
        weights = features.new_tensor([q * self.decay ** (len(entries)-1-i)
                                      for i, (_, q) in enumerate(entries)])
        return (features * weights[:, None]).sum(0) / weights.sum()


class RDSSCache:
    def __init__(self, embeddings, classnames, k=8, memory_capacity=32, decay=.9,
                 entropy_threshold=.2, discrimination_weight=1., redundancy_weight=.1):
        if not 0 <= entropy_threshold < 1:
            raise ValueError('Normalized entropy threshold must be in [0, 1)')
        self.candidates = [embeddings[n] for n in classnames]
        if any(len(c) < k for c in self.candidates):
            raise ValueError('Every class must contain at least k descriptions')
        self.prototypes = FixedSemanticCache(embeddings, classnames).prototypes.clone()
        self.similarities = [c.float() @ c.float().T for c in self.candidates]
        self.memory = RecentClassMemory(len(classnames), memory_capacity, decay)
        self.k = k
        self.entropy_threshold = entropy_threshold
        self.discrimination_weight = discrimination_weight
        self.redundancy_weight = redundancy_weight
        self.selected = [None for _ in classnames]
        self.accepted = [0 for _ in classnames]

    @torch.no_grad()
    def logits(self, image_features):
        return 100. * image_features @ self.prototypes

    @torch.no_grad()
    def observe(self, feature, class_index, entropy, consistent, hard_classes, step):
        """Called after prediction; accepts no ground-truth label or future sample."""
        if not consistent or not 0 <= entropy <= self.entropy_threshold:
            return None
        self.memory.add(class_index, feature, 1.-entropy)
        self.accepted[class_index] += 1
        positive_anchor = self.memory.anchor(class_index)
        candidates = self.candidates[class_index]
        relevance = candidates.float() @ positive_anchor
        negative_anchors = []
        used_negatives = []
        for other in hard_classes:
            if other == class_index:
                continue
            anchor = self.memory.anchor(other)
            if anchor is not None:
                negative_anchors.append(anchor)
                used_negatives.append(other)
        if negative_anchors:
            negative_scores = candidates.float() @ torch.stack(negative_anchors, dim=1)
            discrimination = relevance - negative_scores.max(dim=1).values
        else:
            discrimination = torch.zeros_like(relevance)
        indices, scores = select_descriptions(relevance, discrimination,
                                              self.similarities[class_index], self.k,
                                              self.discrimination_weight, self.redundancy_weight)
        mean = candidates[indices].mean(0)
        prototype = mean / mean.norm()
        changed = self.selected[class_index] != indices
        self.prototypes[:, class_index] = prototype
        self.selected[class_index] = indices
        return {'step':step, 'class_index':class_index, 'entropy':entropy,
                'accepted_count':self.accepted[class_index],
                'recent_count':len(self.memory.entries[class_index]),
                'hard_classes':list(hard_classes), 'negative_classes_with_evidence':used_negatives,
                'selected_indices':indices, 'selection_changed':changed,
                'relevance':relevance[indices].cpu().tolist(),
                'discrimination':discrimination[indices].cpu().tolist(), 'greedy_scores':scores}
