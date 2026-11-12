"""Exercise real native process reset, sampling, cancellation and capacity bounds."""
import argparse
import json
from pathlib import Path
import subprocess
import threading

from benchmark_native_phase0 import guard, sha
from experiment import ROOT
from native_cpu_chat import NativeCPUChat, NativeCancelled


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--weights', type=Path, default=ROOT / 'build/native-cpu/weights.bin')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    guard(ROOT / 'build/benchmark-status')
    if args.output.exists():
        parser.error('Choose a new output path')
    frozen = {str(p.resolve()): sha(p) for p in (args.binary,args.weights,Path(str(args.weights)+'.index'))}
    report = dict(completed=False, passed=False, gpu=False, pytorch=False, inputs=frozen, checks={})
    engine = NativeCPUChat(None,binary=args.binary.resolve(),weights=args.weights.resolve())
    cancel = threading.Event()
    try:
        engine.start(cancel)
        samples=[]
        for _ in range(2):
            engine.command('reset',cancel)
            engine.command('mode chat 42',cancel)
            samples.append(engine.command('prefill 9707 11 1917',cancel)['token'])
        assert samples[0]==samples[1], 'Seeded chat/reset changed output'
        report['checks']['seeded_chat_repeat']=samples
        engine.command('reset',cancel)
        prompt=json.loads((ROOT/'build/native-phase0/prompt-long.json').read_text())
        timer=threading.Timer(.02,cancel.set)
        old_process=engine.process
        timer.start()
        try:
            engine.command('prefill '+' '.join(map(str,prompt)),cancel)
            raise AssertionError('Expected in-flight cancellation')
        except NativeCancelled:
            assert old_process.poll() is not None and engine.process is None
            report['checks']['inflight_cancellation_reaped_process']=True
        finally:
            timer.cancel();timer.join()
        cancel.clear();engine.start(cancel)
        engine.command('mode greedy',cancel)
        restored=engine.command('prefill 9707',cancel)
        assert restored['token']==21806 and restored['position']==1
        report['checks']['restart_after_cancellation']=True
        engine.close()
        commands='prefill 9707 11 1917\nstep 0\nstats\nprefill 151936\nstats\nprefill 9707\nquit\n'
        result=subprocess.run([str(args.binary.resolve()),'--weights',str(args.weights.resolve()),
            '--threads','2','--prefill-threads','2','--context','3'],input=commands,text=True,
            capture_output=True,timeout=60,check=True)
        rows=[json.loads(line) for line in result.stdout.splitlines()]
        assert len(rows)==8 and rows[1]['position']==3
        assert 'error' in rows[2] and rows[3]['position']==0
        assert 'error' in rows[4] and rows[5]['position']==0
        assert rows[6]['token']==21806 and rows[6]['position']==1
        report['checks']['capacity_and_invalid_id_reset']=True
        assert all(sha(Path(path))==digest for path,digest in frozen.items())
        report.update(completed=True,passed=True)
    except BaseException as error:
        report['error']=dict(type=type(error).__name__,message=str(error))
        raise
    finally:
        engine.close()
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report['checks']))


if __name__=='__main__':
    main()
