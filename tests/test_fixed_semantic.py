import sys
from pathlib import Path
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from semantic_cache import FixedSemanticCache, fuse_semantic_logits


class FixedSemanticTests(unittest.TestCase):
    def test_all_100_descriptions_contribute_equally(self):
        entries = torch.cat([torch.tensor([[1., 0.]]).repeat(99, 1), torch.tensor([[0., 1.]])])
        cache = FixedSemanticCache({'class': entries}, ['class'])
        expected = torch.tensor([99., 1.]) / torch.tensor([99., 1.]).norm()
        torch.testing.assert_close(cache.prototypes[:, 0], expected)

    def test_class_order_controls_logit_columns(self):
        embeddings = {'a':torch.tensor([[1., 0.]]), 'b':torch.tensor([[0., 1.]])}
        cache = FixedSemanticCache(embeddings, ['b', 'a'])
        torch.testing.assert_close(cache.logits(torch.tensor([[1., 0.]])), torch.tensor([[0., 100.]]))

    def test_queries_do_not_update_cache_or_source(self):
        entries = torch.eye(2)
        original = entries.clone()
        cache = FixedSemanticCache({'class':entries}, ['class'])
        prototype = cache.prototypes.clone()
        cache.logits(torch.tensor([[1., 0.]]))
        cache.logits(torch.tensor([[0., 1.]]))
        torch.testing.assert_close(entries, original, rtol=0, atol=0)
        torch.testing.assert_close(cache.prototypes, prototype, rtol=0, atol=0)

    def test_fusion_preserves_zero_weight_and_adds_all_terms(self):
        template = torch.tensor([[3., 1.]])
        positive, negative = torch.tensor([[.5, .2]]), torch.tensor([[.1, .3]])
        semantic = torch.tensor([[-2., 4.]])
        tda = (template + positive) - negative
        torch.testing.assert_close(fuse_semantic_logits(tda, semantic, 0.), tda, rtol=0, atol=0)
        torch.testing.assert_close(fuse_semantic_logits(tda, semantic, 1.), template + positive - negative + semantic)
        self.assertNotEqual(int(tda.argmax()), int(fuse_semantic_logits(tda, semantic, 1.).argmax()))


if __name__ == '__main__':
    unittest.main()
