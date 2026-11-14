#!/usr/bin/env python3
"""Post-run provenance binding only; never compiles or executes the stress test."""
from pathlib import Path
import hashlib,json
B=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
report=B/'decode-attention-active8-v1-pool-stress-report.json';j=json.loads(report.read_text())
assert j['completed']and j['passed']and j['partition_jobs']==3000 and j['workers']==[1,6,8,12,64]
stress=B/'decode-attention-active8-v1-source/pool_stress.cpp';binary=B/'decode-attention-active8-v1-pool-stress'
metadata=json.loads((stress.parent/'experiment.json').read_text());assert sha(stress)==metadata['pool_stress']['source_sha256']
source=B/'decode-attention-active8-v2-source/core.cpp';text=source.read_text();pool=text[text.index('class Pool {'):text.index('struct Mapping {')]
assert pool in stress.read_text() and hashlib.sha256(pool.encode()).hexdigest()==metadata['pool_stress']['pool_class_sha256']
old=B/'decode-attention-active8-v1-source/core.cpp';assert sha(source)==sha(old)
out=B/'decode-attention-active8-v2-pool-stress-binding.json';assert not out.exists()
record=dict(completed=True,passed=True,run_executed_by_binder=False,report=str(report),report_sha256=sha(report),stress_source=str(stress),stress_source_sha256=sha(stress),stress_binary=str(binary),stress_binary_sha256=sha(binary),candidate_core=str(source),candidate_core_sha256=sha(source),pool_class_sha256=hashlib.sha256(pool.encode()).hexdigest(),workers=j['workers'],partition_jobs=j['partition_jobs'],binder_sha256=sha(__file__),limitations=['Post-run binding of existing root-run result and unchanged exact extracted Pool source; not a compiler provenance attestation captured at compile time.','v1 engine link failed because standalone stress main was colocated; v2 preserves runtime bytes and moves no math.','No model or worker-count speed claims.'])
with out.open('x')as f:json.dump(record,f,indent=2);f.write('\n')
print(out)
