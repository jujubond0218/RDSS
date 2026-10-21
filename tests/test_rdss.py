import sys
from pathlib import Path
import unittest

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rdss import RDSSCache, RecentClassMemory, select_descriptions
from semantic_cache import FixedSemanticCache


class RDSSTests(unittest.TestCase):
    def cache(self, **kwargs):
        embeddings = {'dog': torch.eye(2), 'cat': torch.eye(2)}
        return RDSSCache(embeddings, ['dog', 'cat'], k=1, **kwargs)

    def test_cold_start_exactly_matches_full_pool(self):
        model = self.cache()
        fixed = FixedSemanticCache({'dog': torch.eye(2), 'cat': torch.eye(2)}, ['dog', 'cat'])
        torch.testing.assert_close(model.read_prototypes(0), fixed.prototypes, rtol=0, atol=0)

    def test_unreliable_and_inconsistent_samples_do_not_write(self):
        model = self.cache()
        initial = model.prototypes.clone()
        self.assertIsNone(model.observe(torch.tensor([1., 0.]), 0, .8, True, [1], 0))
        self.assertIsNone(model.observe(torch.tensor([1., 0.]), 0, .01, False, [1], 1))
        self.assertEqual(model.accepted, [0, 0])
        torch.testing.assert_close(model.prototypes, initial, rtol=0, atol=0)

    def test_prediction_reads_do_not_write_current_query(self):
        model = self.cache()
        image = torch.tensor([[1., 0.]])
        first = model.logits(image, 0)
        model.logits(torch.tensor([[0., 1.]]), 0)
        self.assertEqual(model.accepted, [0, 0])
        torch.testing.assert_close(model.logits(image, 0), first, rtol=0, atol=0)
        model.observe(image[0], 0, .01, True, [1], 0)
        self.assertGreater(float(model.logits(image, 1)[0, 0]), float(first[0, 0]))

    def test_global_time_reduces_absolute_evidence_mass(self):
        memory = RecentClassMemory(1, time_half_life=2)
        memory.add(0, torch.tensor([1., 0.]), 1., 0)
        self.assertAlmostEqual(float(memory.evidence(0, 4)[1].sum()), .25)

    def test_new_evidence_outweighs_old_evidence_after_gap(self):
        memory = RecentClassMemory(1, time_half_life=2)
        memory.add(0, torch.tensor([1., 0.]), 1., 0)
        memory.add(0, torch.tensor([0., 1.]), 1., 8)
        weights = memory.evidence(0, 8)[1]
        self.assertAlmostEqual(float(weights[1] / weights.sum()), 16 / 17, places=6)
        self.assertAlmostEqual(float(memory.anchor(0, 8).norm()), 1.)

    def test_future_evidence_and_repeated_observations_rejected(self):
        model = self.cache()
        model.observe(torch.tensor([1., 0.]), 0, .01, True, [], 3)
        with self.assertRaises(ValueError): model.logits(torch.tensor([[1., 0.]]), 2)
        with self.assertRaises(ValueError): model.observe(torch.tensor([1., 0.]), 0, .01, True, [], 3)

    def test_first_evidence_cannot_fully_replace_prior(self):
        model = self.cache(prior_strength=4.)
        event = model.observe(torch.tensor([1., 0.]), 0, 0., True, [], 0)
        self.assertAlmostEqual(event['adaptation_weight'], .2, places=6)
        self.assertLess(float(model.read_prototypes(0)[0, 0]), 1.)

    def test_stale_evidence_returns_smoothly_toward_prior(self):
        model = self.cache(time_half_life_per_class=1., prior_strength=4.)
        model.observe(torch.tensor([1., 0.]), 0, 0., True, [], 0)
        initial_distance = (model.read_prototypes(0) - model.fixed).norm()
        later_distance = (model.read_prototypes(20) - model.fixed).norm()
        self.assertLess(float(later_distance), float(initial_distance) / 100)

    def test_cached_statistics_match_explicit_time_weighted_memory(self):
        model = self.cache(time_half_life_per_class=2., prior_strength=4.)
        model.observe(torch.tensor([1., 0.]), 0, .01, True, [], 0)
        model.observe(torch.tensor([0., 1.]), 0, .1, True, [], 5)
        vectors, weights = model.memory.evidence(0, 19)
        mass = weights.sum()
        concentration = ((vectors * weights[:, None]).sum(0) / mass).norm()
        torch.testing.assert_close(model.adaptation_weight(0, 19), concentration * mass / (mass + 4.))

    def test_shift_and_return_preserve_candidates_and_memory_bound(self):
        model = self.cache(memory_capacity=3, discrimination_weight=0.,
                           time_half_life_per_class=.5, prior_strength=0.)
        for step in range(3): model.observe(torch.tensor([1., 0.]), 0, .01, True, [], step)
        self.assertEqual(model.selected[0], [0])
        for step in range(3, 6): model.observe(torch.tensor([0., 1.]), 0, .01, True, [], step)
        self.assertEqual(model.selected[0], [1])
        for step in range(6, 9): model.observe(torch.tensor([1., 0.]), 0, .01, True, [], step)
        self.assertEqual(model.selected[0], [0])
        self.assertEqual(len(model.memory.entries[0]), 3)
        self.assertEqual(len(model.candidates[0]), 2)

    def select(self, negatives, k=1, discrimination_weight=1.):
        candidates = F.normalize(torch.tensor([[.8, .6], [.7, -.714]]), dim=1)
        return select_descriptions(candidates, torch.tensor([[1., 0.]]), torch.ones(1),
                                   torch.tensor([[1.], [0.]]), negatives,
                                   candidates @ candidates.T, k, discrimination_weight, 0.)[0]

    def test_negative_margin_rejects_high_relevance_but_confusing_sentence(self):
        negative = (torch.tensor([[0., 1.]]), torch.ones(1), torch.tensor([1., 0.]), 1.)
        self.assertEqual(self.select([negative], discrimination_weight=0.), [0])
        self.assertEqual(self.select([negative]), [1])

    def test_repeating_negative_samples_does_not_change_class_weight(self):
        one = (torch.tensor([[0., 1.]]), torch.ones(1), torch.tensor([1., 0.]), 1.)
        repeated = (one[0].repeat(20, 1), torch.ones(20), one[2], 1.)
        self.assertEqual(self.select([one]), self.select([repeated]))

    def test_stale_negative_cannot_dominate_selection(self):
        stale = (torch.tensor([[0., 1.]]), torch.ones(1), torch.tensor([1., 0.]), 1e-8)
        self.assertEqual(self.select([stale]), [0])

    def test_no_duplicate_sentences_in_aggregate(self):
        self.assertEqual(set(self.select([], k=2)), {0, 1})

    def test_fp16_outputs_are_finite_and_normalized(self):
        pool = {'dog': torch.eye(2).half(), 'cat': torch.eye(2).half()}
        model = RDSSCache(pool, ['dog', 'cat'], k=1)
        model.observe(torch.tensor([1., 0.]).half(), 0, .01, True, [1], 0)
        prototypes = model.read_prototypes(1)
        self.assertEqual(prototypes.dtype, torch.float16)
        self.assertTrue(bool(torch.isfinite(prototypes).all()))
        torch.testing.assert_close(prototypes.float().norm(dim=0), torch.ones(2), atol=.001, rtol=0)


if __name__ == '__main__': unittest.main()
