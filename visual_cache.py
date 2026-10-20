"""Template-driven TDA visual state shared by semantic-cache experiments."""
from dataclasses import dataclass

import torch

from tda_runner import compute_cache_logits, update_cache
from utils import get_entropy, softmax_entropy


@dataclass
class VisualPrediction:
    template: torch.Tensor
    positive: torch.Tensor
    negative: torch.Tensor
    tda: torch.Tensor
    pseudo_label: int
    entropy: float


class TemplateVisualCache:
    def __init__(self, template_prototypes, config):
        self.template_prototypes = template_prototypes
        self.config = config
        self.positive = {}
        self.negative = {}

    @torch.no_grad()
    def predict(self, image):
        original = 100. * image @ self.template_prototypes
        loss = softmax_entropy(original)
        probabilities = original.softmax(1)
        pseudo = int(original.argmax(1))
        entropy = get_entropy(loss, self.template_prototypes)
        pos, neg = self.config['positive'], self.config['negative']
        if pos['enabled']:
            update_cache(self.positive, pseudo, [image, loss], pos['shot_capacity'])
        if neg['enabled'] and neg['entropy_threshold']['lower'] < entropy < neg['entropy_threshold']['upper']:
            update_cache(self.negative, pseudo, [image, loss, probabilities], neg['shot_capacity'], True)
        positive, negative = torch.zeros_like(original), torch.zeros_like(original)
        tda = original.clone()
        if pos['enabled'] and self.positive:
            positive = compute_cache_logits(image, self.positive, pos['alpha'], pos['beta'], self.template_prototypes)
            tda += positive
        if neg['enabled'] and self.negative:
            negative = compute_cache_logits(image, self.negative, neg['alpha'], neg['beta'], self.template_prototypes,
                                            (neg['mask_threshold']['lower'], neg['mask_threshold']['upper']))
            tda -= negative
        return VisualPrediction(original, positive, negative, tda, pseudo, float(entropy))
