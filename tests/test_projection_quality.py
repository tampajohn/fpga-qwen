"""Tiny scoring/transport fixtures only: no checkpoint, simulator or neural math."""
import math
from pathlib import Path
import tempfile
import json
import unittest

import numpy as np

from tools.evaluate_projection_quality import (
    DEFAULT_CORPUS,read_dump_logits,aggregate,load_corpus,model_provenance,score_logits,teacher_force,validate_logits,
)


class StubTransport:
    def __init__(self,logits):
        self.logits=np.asarray(logits,dtype=np.float64);self.reset()
    def reset(self):self.position=0;self.fed=[];self.blocks=[]
    def token(self,token,logits=True):
        self.fed.append(token);self.position+=1
        return int(np.argmax(self.logits)),self.position,float(np.max(self.logits))
    def block(self,tokens,logits=False):
        self.blocks.append(list(tokens));return [self.token(t,logits)for t in tokens]


class ProjectionQualityTests(unittest.TestCase):
    def test_dump_barrier_flushes_before_exact_byte_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'logits.bin'
            class DelayedDump:
                lanes=8
                def __init__(self,complete):self.calls=[];self.complete=complete
                def command(self,command):
                    self.calls.append(command)
                    if command.startswith('dump '):
                        path.write_bytes(b'partial');return 'dumped'
                    if command=='lanes 8':
                        if self.complete:np.array([1.,2.,3.]).tofile(path)
                        return 'configured'
                    raise AssertionError(command)
            model=DelayedDump(True)
            np.testing.assert_array_equal(read_dump_logits(model,path,3),[1,2,3])
            self.assertEqual(model.calls,[f'dump {path}','lanes 8'])
            with self.assertRaisesRegex(ValueError,'expected 24, got 7'):
                read_dump_logits(DelayedDump(False),path,3)

    def test_raw_tokenizers_encoding_exposes_sliceable_ids(self):
        from tokenizers import Tokenizer,models,pre_tokenizers
        tokenizer=Tokenizer(models.WordLevel({'[UNK]':0,'amber':1,'72':2},unk_token='[UNK]'))
        tokenizer.pre_tokenizer=pre_tokenizers.Whitespace()
        ids=tokenizer.encode('amber 72',add_special_tokens=False).ids
        self.assertEqual(ids[:1],[1])
        self.assertEqual(ids,[1,2])

    def test_checkpoint_and_template_hash_provenance(self):
        import hashlib
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            (root/'model.safetensors').write_bytes(b'tiny synthetic checkpoint fixture')
            (root/'config.json').write_text('{}')
            (root/'tokenizer.json').write_text('{}')
            (root/'chat_template.jinja').write_text('{{ message }}')
            report=model_provenance(root)
            self.assertEqual(report['original_checkpoint']['sha256'],hashlib.sha256((root/'model.safetensors').read_bytes()).hexdigest())
            self.assertEqual(set(report['config_tokenizer_template_files']),{'config.json','tokenizer.json','chat_template.jinja'})
            self.assertTrue(report['same_original_checkpoint_for_both_engines'])

    def test_uniform_distribution_and_identity(self):
        row=score_logits([0,0,0],[0,0,0],2,3)
        self.assertAlmostEqual(row['reference_nll_nats'],math.log(3),places=14)
        self.assertEqual(row['delta_nll_nats'],0)
        self.assertEqual(row['kl_reference_to_candidate_nats'],0)
        self.assertTrue(row['bit_identical_logits'])
        self.assertEqual(row['reference_top1'],0)  # Deterministic first-index tie.

    def test_analytic_binary_distribution(self):
        row=score_logits(np.log([.75,.25]),np.log([.5,.5]),1,2)
        self.assertAlmostEqual(row['reference_nll_nats'],-math.log(.25),places=14)
        self.assertAlmostEqual(row['candidate_nll_nats'],-math.log(.5),places=14)
        self.assertAlmostEqual(row['delta_nll_nats'],math.log(.5),places=14)
        expected=.75*math.log(1.5)+.25*math.log(.5)
        self.assertAlmostEqual(row['kl_reference_to_candidate_nats'],expected,places=14)

    def test_large_logits_are_stable_and_shift_invariant(self):
        row=score_logits([1000,999,-1000],[2000,1999,0],2,3)
        self.assertTrue(math.isfinite(row['reference_nll_nats']))
        self.assertAlmostEqual(row['reference_nll_nats'],2000+math.log1p(math.exp(-1)),places=12)
        self.assertAlmostEqual(row['delta_nll_nats'],0,places=12)
        self.assertAlmostEqual(row['kl_reference_to_candidate_nats'],0,places=12)
        self.assertEqual(row['max_absolute_logit_error'],1000)

    def test_shape_finite_and_token_validation(self):
        for values in ([0,1],[[0,1,2]],[0,np.nan,1],[0,np.inf,1]):
            with self.subTest(values=values),self.assertRaises(ValueError):validate_logits(values,3)
        for target in (-1,3,True,1.5):
            with self.subTest(target=target),self.assertRaises(ValueError):score_logits([0]*3,[0]*3,target,3)

    def test_aggregate_is_token_weighted_not_case_weighted(self):
        one=score_logits([0,0],[0,1],0,2)
        three=score_logits([0,0],[0,2],0,2)
        result=aggregate([one,three,three,three])
        expected=(one['delta_nll_nats']+3*three['delta_nll_nats'])/4
        self.assertAlmostEqual(result['delta_mean_nll_nats'],expected,places=14)
        self.assertNotAlmostEqual(result['delta_mean_nll_nats'],(one['delta_nll_nats']+three['delta_nll_nats'])/2,places=8)
        self.assertEqual(result['scored_tokens'],4)
        with self.assertRaises(ValueError):aggregate([])

    def test_teacher_forces_authored_tokens_and_correct_alignment(self):
        reference,candidate=StubTransport([3,2,1]),StubTransport([1,3,2])
        rows=teacher_force(reference,candidate,[2,0],[1,2,1],lambda m,slot:m.logits,vocab_size=3)
        self.assertEqual(reference.fed,[2,0,1,2])
        self.assertEqual(candidate.fed,reference.fed)
        self.assertEqual([r['context_position']for r in rows],[2,3,4])
        self.assertEqual([r['target_id']for r in rows],[1,2,1])
        self.assertFalse(any(r['top1_agreement']for r in rows))

    def test_block_prefix_tails_and_reference_prefix_are_identical(self):
        a,b=StubTransport([1,2,3]),StubTransport([1,2,3])
        prompt=[0,1,2,0,1,2,0,1,2,0]
        teacher_force(a,b,prompt,[1,2],lambda m,slot:m.logits,1,4,3)
        self.assertEqual(b.blocks,[[0,1,2,0],[1,2,0,1],[2]])
        self.assertEqual(a.fed,b.fed)
        self.assertEqual(a.fed,prompt+[1])

    def test_position_and_dump_id_mismatches_fail(self):
        a,b=StubTransport([1,2,3]),StubTransport([1,2,3])
        with self.assertRaisesRegex(RuntimeError,'greedy IDs'):
            teacher_force(a,b,[0],[1],lambda m,slot:np.array([3,2,1]),vocab_size=3)
        class BadPosition(StubTransport):
            def token(self,*args,**kwargs):
                token,pos,logit=super().token(*args,**kwargs);return token,pos+1,logit
        with self.assertRaisesRegex(RuntimeError,'position'):
            teacher_force(a,BadPosition([1,2,3]),[0],[1],lambda m,slot:m.logits,vocab_size=3)

    def test_context_and_prompt_validation(self):
        a,b=StubTransport([1,2,3]),StubTransport([1,2,3])
        for prompt,target in [([], [1]),([0],[]),([3],[1]),([0]*4096,[1])]:
            with self.subTest(prompt_length=len(prompt)),self.assertRaises(ValueError):
                teacher_force(a,b,prompt,target,lambda m,slot:m.logits,vocab_size=3)

    def test_corpus_provenance_categories_and_known_training_overlap(self):
        corpus,_=load_corpus(DEFAULT_CORPUS,Path(__file__).resolve().parents[1]/'reports/arm-pgo-training.json')
        self.assertEqual(len(corpus['examples']),16)
        self.assertEqual({e['category']for e in corpus['examples']},{'code','prose','arithmetic','dialogue'})
        with tempfile.TemporaryDirectory() as directory:
            report=Path(directory)/'training.json'
            report.write_text(json.dumps(dict(training_cases=[dict(prompt=corpus['examples'][0]['messages'][0]['content'])])))
            with self.assertRaisesRegex(ValueError,'duplicates recorded local training'):
                load_corpus(DEFAULT_CORPUS,report)


if __name__=='__main__':unittest.main()
