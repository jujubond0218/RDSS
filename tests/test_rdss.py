import sys
from pathlib import Path
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rdss import RDSSCache, RecentClassMemory, select_descriptions
from semantic_cache import FixedSemanticCache


class RDSSTests(unittest.TestCase):
    def cache(self, **kwargs):
        candidates = {'dog':torch.eye(2), 'cat':torch.eye(2)}
        return RDSSCache(candidates, ['dog','cat'], k=1, memory_capacity=3, decay=.5, **kwargs)

    def test_cold_start_is_exactly_fixed_all_description_cache(self):
        embeddings = {'dog':torch.eye(2), 'cat':torch.eye(2)}
        current = self.cache()
        fixed = FixedSemanticCache(embeddings,['dog','cat'])
        torch.testing.assert_close(current.prototypes,fixed.prototypes,rtol=0,atol=0)

    def test_unreliable_or_inconsistent_predictions_do_not_update(self):
        cache = self.cache()
        initial = cache.prototypes.clone()
        self.assertIsNone(cache.observe(torch.tensor([1.,0.]),0,.8,True,[1],0))
        self.assertIsNone(cache.observe(torch.tensor([1.,0.]),0,.01,False,[1],1))
        self.assertEqual(cache.accepted,[0,0])
        torch.testing.assert_close(cache.prototypes,initial,rtol=0,atol=0)

    def test_attribute_shift_and_return_with_bounded_recent_memory(self):
        cache = self.cache()
        for step in range(3): cache.observe(torch.tensor([1.,0.]),0,.01,True,[],step)
        self.assertEqual(cache.selected[0],[0])
        for step in range(3,6): cache.observe(torch.tensor([0.,1.]),0,.01,True,[],step)
        self.assertEqual(cache.selected[0],[1])
        for step in range(6,9): cache.observe(torch.tensor([1.,0.]),0,.01,True,[],step)
        self.assertEqual(cache.selected[0],[0])
        self.assertEqual(len(cache.memory.entries[0]),3)
        self.assertEqual(len(cache.candidates[0]),2)

    def test_recency_weights_new_evidence_more(self):
        memory = RecentClassMemory(1,capacity=4,decay=.5)
        memory.add(0,torch.tensor([1.,0.]),1.)
        memory.add(0,torch.tensor([0.,1.]),1.)
        torch.testing.assert_close(memory.anchor(0),torch.tensor([1/3,2/3]))

    def test_discrimination_can_outweigh_generic_relevance(self):
        relevance, disc = torch.tensor([.9,.8]), torch.tensor([0.,.3])
        similarities = torch.eye(2)
        self.assertEqual(select_descriptions(relevance,disc,similarities,1,0.,0.)[0],[0])
        self.assertEqual(select_descriptions(relevance,disc,similarities,1,1.,0.)[0],[1])

    def test_redundancy_penalty_selects_complementary_evidence(self):
        rel = torch.tensor([.9,.89,.8]); disc = torch.zeros(3)
        sim = torch.tensor([[1.,.99,.1],[.99,1.,.1],[.1,.1,1.]])
        self.assertEqual(select_descriptions(rel,disc,sim,2,0.,0.)[0],[0,1])
        self.assertEqual(select_descriptions(rel,disc,sim,2,0.,.5)[0],[0,2])

    def test_only_observed_confusers_provide_negative_evidence(self):
        cache = self.cache()
        first = cache.observe(torch.tensor([1.,0.]),0,.01,True,[1],0)
        self.assertEqual(first['negative_classes_with_evidence'],[])
        cache.observe(torch.tensor([0.,1.]),1,.01,True,[0],1)
        event = cache.observe(torch.tensor([1.,0.]),0,.01,True,[1],2)
        self.assertEqual(event['negative_classes_with_evidence'],[1])
        self.assertGreater(event['discrimination'][0],0)


if __name__=='__main__':
    unittest.main()
