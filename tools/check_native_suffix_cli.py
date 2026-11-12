#!/usr/bin/env python3
"""Guarded real-model greedy-B2 protocol tests; no performance claims.

Supply a short prompt whose natural greedy trajectory reaches EOS within64 steps.
The reference is the same binary's ordinary prefill/step path, never a future-token
source for a production drafter. Explicit proposals here are correctness fixtures.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import select
import subprocess
import time

from benchmark_guard import BenchmarkGate

ROOT = Path(__file__).resolve().parents[1]
EOS = {151643, 151645}


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


class Client:
    def __init__(self, binary, weights, directory):
        self.process = None; self.buffer = b''
        self.log = (directory/'stderr.log').open('xb')
        env = {k:v for k,v in os.environ.items() if not k.startswith(('DYLD_', 'GGML_KLEIDIAI_', 'LLAMA_ARG_'))
               and k not in ('GGML_BACKEND_PATH', 'GGML_TOTAL_THREADS', 'VECLIB_MAXIMUM_THREADS')}
        env['VECLIB_MAXIMUM_THREADS'] = '6'
        try:
            self.process = subprocess.Popen([str(binary), '--weights', str(weights), '--threads', '12',
                '--prefill-threads', '6', '--context', '4096'], stdin=subprocess.PIPE,
                stdout=subprocess.PIPE, stderr=self.log, bufsize=0, env=env)
            self.ready = self.read()
            if self.ready.get('ready') is not True or self.ready.get('verify_greedy_b2') is not True or self.ready.get('gpu') is not False:
                raise RuntimeError('Greedy B2 CPU capability missing')
        except BaseException:
            self.close(); raise

    def read(self):
        deadline = time.monotonic()+180
        while b'\n' not in self.buffer:
            if time.monotonic()>deadline: raise TimeoutError('CLI response timeout')
            if select.select([self.process.stdout], [], [], .1)[0]:
                block=os.read(self.process.stdout.fileno(),65536)
                if not block: raise RuntimeError('CLI stdout closed')
                self.buffer+=block
                if len(self.buffer)>1024*1024: raise RuntimeError('Oversized CLI response')
        line,self.buffer=self.buffer.split(b'\n',1)
        return json.loads(line)

    def command(self, text, error=False):
        if '\n' in text or '\r' in text: raise ValueError('Single-line command required')
        data=memoryview((text+'\n').encode())
        while data:
            count=self.process.stdin.write(data)
            if not count: raise RuntimeError('CLI stdin closed')
            data=data[count:]
        result=self.read()
        if not isinstance(result,dict) or ('error' in result)!=error:
            raise RuntimeError('Unexpected protocol response')
        return result

    def close(self):
        if self.process:
            if self.process.poll() is None:
                self.process.terminate()
                try:self.process.wait(timeout=3)
                except subprocess.TimeoutExpired:self.process.kill();self.process.wait(timeout=5)
            for pipe in (self.process.stdin,self.process.stdout):
                if pipe:pipe.close()
        self.log.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('binary','weights','prompt','directory'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--guard-state-dir',type=Path,default=ROOT/'build/benchmark-status')
    args=parser.parse_args();state=args.guard_state_dir.resolve()
    if not BenchmarkGate(state).status()['running'] or json.loads((state/'run.json').read_text()).get('pid')!=os.getppid():
        raise RuntimeError('Run directly under benchmark_guard.py')
    binary,weights,directory=args.binary.resolve(),args.weights.resolve(),args.directory.resolve()
    prompt=json.loads(args.prompt.read_text())
    if not isinstance(prompt,list) or not prompt or len(prompt)>4000 or any(type(x)is not int or not 0<=x<151936 for x in prompt):
        raise ValueError('Invalid prompt ID array')
    directory.mkdir(parents=True,exist_ok=False)
    report=dict(completed=False,passed=False,diagnostic_only=True,performance_claim_eligible=False,cases=[])
    client=None
    def save(): (directory/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    try:
        paths=[binary,weights,Path(str(weights)+'.index'),Path(str(weights)+'.manifest.json'),
               binary.parent/'manifest.json',args.prompt.resolve(),Path(__file__).resolve()]
        report['inputs']={str(p):sha(p) for p in paths}
        client=Client(binary,weights,directory);report['ready']=client.ready
        def begin(index, predictions):
            if client.command('reset')!={'ok':True,'position':0}:raise RuntimeError('Bad reset')
            client.command('mode greedy')
            result=client.command('prefill '+' '.join(map(str,prompt)))
            for i in range(index):
                if result['token']!=predictions[i]:raise RuntimeError('Replay prediction changed')
                result=client.command(f'step {predictions[i]}')
            return result
        def snapshot(label,position):
            blobs=[];hashes=[]
            for command,count in (('dump',151936),('dump-hidden',1024)):
                path=directory/f'{label}-{command}.f32'
                ack=client.command(command+' '+str(path))
                if ack.get('ok')is not True or ack.get('count')!=count or ack.get('position')!=position:
                    raise RuntimeError('Bad diagnostic acknowledgement')
                raw=path.read_bytes()
                if len(raw)!=count*4:raise RuntimeError('Bad diagnostic size')
                blobs.append(raw);hashes.append(sha(path))
            return blobs,hashes
        predictions=[];reference=[]
        response=begin(0,[])
        for index in range(65):
            if response.get('position')!=len(prompt)+index:raise RuntimeError('Bad baseline position')
            predictions.append(response['token'])
            reference.append(snapshot(f'baseline-{index}',len(prompt)+index)[0])
            if response['token'] in EOS:break
            if index==64:raise RuntimeError('Prompt did not reach EOS within64 steps; provide a short EOS fixture')
            response=client.command(f"step {response['token']}")
        eos_index=len(predictions)-1
        if eos_index<2:raise RuntimeError('EOS fixture needs at least two preceding greedy tokens')
        report['reference_ids']=predictions
        wrong=lambda expected: 0 if expected!=0 else 1
        cases=[('reject_first',0,wrong(predictions[1])),('accept_bonus',0,predictions[1]),
               ('eos_first_match',eos_index-1,predictions[-1]),
               ('eos_first_reject',eos_index-1,wrong(predictions[-1])),
               ('eos_bonus',eos_index-2,predictions[-2])]
        for name,index,draft in cases:
            begin(index,predictions)
            first=predictions[index+1];accepted=int(first==draft)
            committed=2 if accepted and first not in EOS else 1
            expected=predictions[index+1:index+1+committed]
            result=client.command(f'verify-greedy {predictions[index]} {draft}')
            required=dict(tokens=expected,accepted=accepted,committed_inputs=committed,
                          position=len(prompt)+index+committed,verified_positions=2,
                          finish_reason='eos' if any(t in EOS for t in expected) else None)
            if any(result.get(k)!=v for k,v in required.items()):raise RuntimeError(f'Wrong B2 response: {name}')
            blobs,hashes=snapshot(name,required['position'])
            if blobs!=reference[index+committed]:raise RuntimeError(f'Committed state differs: {name}')
            report['cases'].append(dict(name=name,expected=required,full_state_sha256=hashes));save()
        for command in ('verify-greedy','verify-greedy -1 0','verify-greedy 0 151936','verify-greedy 0 0 extra'):
            begin(0,predictions)
            if client.command(command,error=True).get('position')!=0 or client.command('stats').get('position')!=0:
                raise RuntimeError('Malformed command did not reset state')
        begin(0,predictions);client.command('mode chat 42')
        if client.command('verify-greedy 0 0',error=True).get('position')!=0:raise RuntimeError('Chat verification was not rejected')
        report['invalid_and_chat_commands_reset']=True
        if any(sha(Path(p))!=h for p,h in report['inputs'].items()):raise RuntimeError('Immutable input changed')
        report.update(completed=True,passed=True)
    except BaseException as error:
        report['error']=dict(type=type(error).__name__,message=str(error));raise
    finally:
        try:
            if client:client.close()
        except BaseException as error:
            report.update(completed=False,passed=False,cleanup_error=str(error))
        save()
    if not report['passed']:raise RuntimeError('CLI test cleanup failed')
    print(directory/'report.json')


if __name__=='__main__':main()
