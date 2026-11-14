#!/usr/bin/env python3
"""Root-run remaining qualification only; never starts timing or repeats API36."""
from pathlib import Path
import json,subprocess,sys
from check_active8_qualification_v1 import B,TAG,sha,validate
ROOT=B.parents[1];PY=sys.executable
candidate=B/f'{TAG}-build/qwen_cpu';weights=B/'weights.bin';out=B/f'{TAG}-qualification.json';assert not out.exists();validate(False)
commands=[('prefix',[ROOT/'tools/check_native_cpu_prefix.py','--binary',candidate,'--weights',weights,'--directory',B/f'{TAG}-prefix-check']),('suffix',[ROOT/'tools/check_native_suffix.py','--binary',candidate,'--reference-binary',B/'build-20/qwen_cpu','--weights',weights,'--output',B/f'{TAG}-suffix-trajectories.json']),('cli',[ROOT/'tools/check_native_suffix_cli.py','--binary',candidate,'--weights',weights,'--prompt',B/'suffix-eos-hi.json','--directory',B/f'{TAG}-suffix-cli']),('quality',[ROOT/'tools/benchmark_native_cpu.py','--binary',candidate,'--reference-binary',B/'build-20/qwen_cpu','--weights',weights,'--mode','quality','--threads',12,'--prefill-threads',6,'--output',B/f'{TAG}-quality.json']),('extended',[B/'check_active8_extended_quality_v1.py','--binary',candidate,'--weights',weights,'--output',B/f'{TAG}-extended-quality.json']),('seeded',[B/'check-seeded-stream-regression-v1.py','--candidate',candidate,'--baseline',B/'build-20/qwen_cpu','--directory',B/f'{TAG}-seeded-stream'])]
for suffix in ('prefix-check','suffix-trajectories.json','suffix-cli','quality.json','quality-logs','extended-quality.json','extended-quality-logs','seeded-stream'):assert not(B/f'{TAG}-{suffix}').exists()
paths=[Path(__file__).resolve(),B/'check_active8_qualification_v1.py']+[Path(args[0])for _,args in commands]+[ROOT/'tools'/n for n in ('native_cpu_chat.py','chat_protocol.py','suffix.py','benchmark_guard.py')]
frozen={str(p):sha(p)for p in paths};record={'completed':False,'passed':False,'source_sha256':frozen,'executed':[]}
def save():out.write_text(json.dumps(record,indent=2)+'\n')
save()
try:
 for name,args in commands:
  assert all(sha(p)==digest for p,digest in frozen.items())
  subprocess.run([PY,str(ROOT/'tools/benchmark_guard.py'),'--name','Active8 qualification '+name,'--',PY,*map(str,args)],cwd=ROOT,check=True);record['executed'].append(name);save()
 record.update(validate());assert all(sha(p)==digest for p,digest in frozen.items())
except BaseException as e:record['error']={'type':type(e).__name__,'message':str(e)};raise
finally:save()
print(out)
