"""Fixed class semantic prototypes and additive logit fusion."""
import torch


class FixedSemanticCache:
    """Equally aggregate every pre-normalized description embedding per class."""

    @torch.no_grad()
    def __init__(self, embeddings, classnames):
        vectors = []
        for name in classnames:
            entries = embeddings[name]
            if entries.ndim != 2 or len(entries) == 0:
                raise ValueError(f'Expected a nonempty embedding matrix for {name}')
            mean = entries.mean(dim=0)
            norm = mean.norm()
            if not torch.isfinite(norm) or norm <= 0:
                raise ValueError(f'Invalid semantic prototype for {name}')
            vectors.append(mean / norm)
        self.prototypes = torch.stack(vectors, dim=1)

    @torch.no_grad()
    def logits(self, image_features):
        return 100.0 * image_features @ self.prototypes


@torch.no_grad()
def fuse_semantic_logits(tda_logits, semantic_logits, weight):
    """TDA already includes template logits and positive/negative corrections."""
    if tda_logits.shape != semantic_logits.shape:
        raise ValueError('TDA and semantic logits must have identical shapes')
    return tda_logits + weight * semantic_logits
