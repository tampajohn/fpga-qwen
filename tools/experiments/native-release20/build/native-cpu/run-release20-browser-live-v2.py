#!/usr/bin/env python3
"""Root-run live correctness check only; requires verified successful promotion."""
import subprocess,sys,urllib.request
from release20_common import *
sys.path.insert(0,str(ROOT/'tools'))
from benchmark_guard import BenchmarkGate
assert not BenchmarkGate(ROOT/'build/benchmark-status').status()['running'], 'Live browser test runs outside outer guard; each chat and visibility check acquires it'
p=BASE/'build-20-promotion.json';j=read(p);assert j['completed'] and not j.get('error')
unchanged(j['helper_sha256'])
assert os.readlink(BASE/'qwen_cpu')=='build-20/qwen_cpu' and sha(BASE/'build-20/qwen_cpu')==j['binary_sha256']
with urllib.request.urlopen('http://127.0.0.1:8765/api/status',timeout=2) as response:s=json.load(response)
assert s['build_id']=='build-20' and s['default_backend']=='native-cpu-f16' and not s['gpu_inference'] and not s['pytorch_loaded']
out=BASE/'build-20-browser-live';assert not out.exists()
subprocess.run([sys.executable,str(ROOT/'tests/browser_native_cpu.py'),'--url','http://127.0.0.1:8765','--live','--expect-prefix-reuse','--expect-greedy-suffix','--expect-norm-accumulator-f64','--directory',str(out)],cwd=ROOT,check=True)
r=read(out/'report.json');assert r['completed'] and not r.get('error') and not r['console_errors']
with urllib.request.urlopen('http://127.0.0.1:8765/api/status',timeout=2) as response:s=json.load(response)
assert s['build_id']=='build-20' and os.readlink(BASE/'qwen_cpu')=='build-20/qwen_cpu'
unchanged(j['helper_sha256'])
write(BASE/'build-20-live-browser-binding.json',dict(completed=True,promotion_sha256=sha(p),browser_sha256=sha(out/'report.json'),binary_sha256=sha(BASE/'build-20/qwen_cpu'),status=s))

write(BASE/'build-20-live-controller-correction.json',dict(completed=True,reason='Original controller omitted required --live flag; argument parser rejected before browser execution. Original frozen helper remains unchanged.',original_helper_sha256=sha(BASE/'run-release20-browser-live.py'),executed_helper_sha256=sha(Path(__file__)),executed_helper=str(Path(__file__).resolve()),binding_sha256=sha(BASE/'build-20-live-browser-binding.json')))
