"""Actual4096 boundary gate with phase-routing telemetry; no timing claim."""
import json
import os
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tests.test_prefill_model import PrefillModelTests

class PhaseCapacityTests(PrefillModelTests):
    observations=[]
    def model(self,reference=False,workers=8):
        model=super().model(reference,workers)
        if not reference:
            initial=json.loads(model.command('phase_pools'))
            self.assertEqual(initial['selector'],'PREFILL_12')
            for method in ('prefill','token','block'):
                original=getattr(model,method)
                def wrapped(*args,_method=method,_original=original,**kwargs):
                    result=_original(*args,**kwargs)
                    position=result[1]if _method=='token' else result[-1][1]
                    if position>=4092:
                        data=json.loads(model.command('phase_pools'))
                        self.assertEqual(data['pool_constructions'],2)
                        self.assertEqual(data['dispatch_counts'][0][1],0)
                        self.assertEqual(data['dispatch_counts'][1][0],0)
                        self.assertEqual(data['dispatch_counts'][1][2],0)
                        self.observations.append(dict(method=_method,position=position,routing=data))
                    return result
                setattr(model,method,wrapped)
        return model
    def test_phase_capacity(self):
        super().test_actual_capacity_boundary()
        for method in ('prefill','token','block'):
            self.assertTrue(any(r['method']==method and r['position']==4096 for r in self.observations))
        last=self.observations[-1]['routing']['dispatch_counts']
        self.assertGreater(last[0][0],0);self.assertGreater(last[0][2],0);self.assertGreater(last[1][1],0)
        path=Path(os.environ['PREFILL_CAPACITY_REPORT'])
        report=json.loads(path.read_text());report['phase_routing_at_boundary']=self.observations
        path.write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.TestSuite([PhaseCapacityTests('test_phase_capacity')]))
    raise SystemExit(not result.wasSuccessful())
