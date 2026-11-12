"""Fused-route cache policy mocks; no numerical or performance claims."""
import json
import unittest
from unittest.mock import Mock
from tools.canonical_prefix import CanonicalPrefix, FMA_POLICY, POLICY
from tests.test_canonical_prefix import target
from tests.test_fma_prefill_profile import CAPABILITY
from tests.test_suffix_chat import run


def fused(cached=True):
    e=target(cached)
    e.precision_profile='fp32_b1_fp32_fma_prefill'
    e.sim.block_size=4
    original=e.sim.command
    capability=dict(CAPABILITY,canonical_prefix_policy=FMA_POLICY if cached else 'none')
    e.sim.command=lambda text,cancel=None: json.dumps(capability) if text=='precision' else original(text,cancel)
    return e


class FusedCanonicalPrefixTests(unittest.TestCase):
    def test_all_partial_tails_and_matching_decode_excluded(self):
        for tail in range(8):
            with self.subTest(tail=tail):
                cached,fresh=fused(),fused(False)
                run(cached,ids=[1,2,3]*8+[1]*tail+[2],max_new=4)
                self.assertEqual(cached.canonical_prefix.end,24)
                ids=list(cached.cache_ids)+[1,2,3,1,2,3,1,2,3]
                a=run(cached,ids=ids,max_new=4,draft=3)
                b=run(fresh,ids=ids,max_new=4,draft=3)
                self.assertEqual(a['metrics']['reused_prefix_tokens'],24)
                self.assertEqual(a['token_ids'],b['token_ids'])
                self.assertEqual(cached.sim.routes,fresh.sim.routes)
                self.assertEqual(cached.canonical_prefix.end,(len(ids)-1)//8*8)
                self.assertEqual(cached.status()['canonical_prefix_policy'],FMA_POLICY)

    def test_mismatch_fails_closed_before_reuse_and_retry(self):
        for changes in ({'canonical_prefix_policy':'none'},{'canonical_prefix_policy':POLICY},
                        {'precision_profile':'fp32_b1_fp64_prefill'},
                        {'prefill_accumulation':'separate'}):
            with self.subTest(changes=changes):
                e=fused();run(e,ids=[1]*25,max_new=2)
                good=e.sim.command
                e.sim.command=Mock(return_value=json.dumps(dict(CAPABILITY,canonical_prefix_policy=FMA_POLICY) | changes))
                with self.assertRaises(RuntimeError):run(e,ids=[1]*25,max_new=2)
                self.assertIsNone(e.sim.process);self.assertEqual(e.canonical_prefix.end,0)
                self.assertEqual(e.cache_ids,[])
                e.sim.command=good
                self.assertEqual(run(e,ids=[1]*25,max_new=2)['metrics']['reused_prefix_tokens'],0)

    def test_epoch_and_route_signature_invalidate(self):
        prefix=CanonicalPrefix();epoch=object()
        prefix.bind(epoch,('profile',FMA_POLICY));prefix.acknowledge(0,8)
        prefix.bind(epoch,('profile',FMA_POLICY));self.assertEqual(prefix.end,8)
        prefix.bind(epoch,('other',POLICY));self.assertEqual(prefix.end,0)
        prefix.acknowledge(0,8);prefix.bind(object(),('other',POLICY))
        self.assertEqual(prefix.end,0)
        e=fused();run(e,ids=[1]*25,max_new=2)
        e.sim.close()
        self.assertEqual(run(e,ids=[1]*25,max_new=2)['metrics']['reused_prefix_tokens'],0)

    def test_capable_binary_can_disable_cache(self):
        e=fused(False);original=e.sim.command
        e.sim.command=lambda text,cancel=None: json.dumps(dict(CAPABILITY,canonical_prefix_policy=FMA_POLICY)) if text=='precision' else original(text,cancel)
        for _ in range(2):
            self.assertEqual(run(e,ids=[1]*25,max_new=2)['metrics']['reused_prefix_tokens'],0)
        self.assertEqual(e.sim.resets,2)

    def test_archived_none_policy_still_resets(self):
        e=fused(False)
        for _ in range(2):
            self.assertEqual(run(e,ids=[1]*25,max_new=2)['metrics']['reused_prefix_tokens'],0)
        self.assertEqual(e.sim.resets,2)

if __name__=='__main__':unittest.main()
