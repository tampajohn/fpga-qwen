"""Wrap paired SV comparison with startup log capture and measured RSS snapshots.

Accepts all compare_variants.py arguments; writes an adjacent .memory.json file.
RSS snapshots are per-process, include shared pages, and are not peak/private RSS.
"""
from pathlib import Path
import json
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tools'))
import compare_variants

original_start=compare_variants.SVEngine.start
original_generate=compare_variants.SVEngine.generate
records=[]

def snapshot(engine,phase,extra=None):
    pid=engine.model.process.pid
    record=dict(timestamp=time.time(),binary=str(engine.model.binary),pid=pid,phase=phase,
                rss_kib=int(subprocess.check_output(['ps','-o','rss=','-p',str(pid)],text=True).strip()))
    if extra:record.update(extra)
    records.append(record)

def start(engine):
    elapsed=original_start(engine)
    startup_log=Path(engine.model.log.name).read_text()
    snapshot(engine,'ready',dict(startup_seconds=elapsed,startup_stderr=startup_log))
    return elapsed

def generate(engine,prompt,count,fixed_count):
    result=original_generate(engine,prompt,count,fixed_count)
    snapshot(engine,'after_request',dict(prompt_tokens=len(prompt),output_tokens=result['output_token_count']))
    return result

compare_variants.SVEngine.start=start
compare_variants.SVEngine.generate=generate
output=Path(sys.argv[sys.argv.index('--output')+1]) if '--output' in sys.argv else ROOT/'build/compare-variants.json'
try:
    compare_variants.main()
finally:
    memory=dict(limitation='RSS snapshots include shared pages; not peak RSS, physical/private memory, or allocation size.',snapshots=records)
    output.with_suffix('.memory.json').write_text(json.dumps(memory,indent=2)+'\n')
    if output.exists():
        report=json.loads(output.read_text())
        for name,settings in report.get('settings',{}).items():
            if 'weight-cache-f32' in settings['binary']:
                settings['weight_storage']='Read-only FP32 cache expanded exactly from BF16 checkpoint; original file mapping retained'
        report['memory_report']=str(output.with_suffix('.memory.json'))
        output.write_text(json.dumps(report,indent=2)+'\n')
