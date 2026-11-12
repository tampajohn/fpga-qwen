#!/usr/bin/env python3
"""Exercise tensor-wide lossless-BF16 fallback with one intentionally changed FP16 weight."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import select
import shutil
import struct
import subprocess
import time

from benchmark_guard import BenchmarkGate

ROOT = Path(__file__).resolve().parents[1]
CACHE_KEY = 'lossless_bf16_decode_cache_bytes'
VOCABULARY = 151936


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def guard(directory):
    status = BenchmarkGate(directory).status()
    metadata = json.loads((directory/'run.json').read_text())
    if not status['running'] or metadata.get('pid') != os.getppid():
        raise RuntimeError('Run directly under serving-checkout benchmark_guard.py')
    return dict(directory=str(directory),metadata=metadata)


def unchanged_except_word(reference, fixture, offset):
    """Stream-compare both full files, excluding precisely the mutated two bytes."""
    if reference.stat().st_size != fixture.stat().st_size:
        raise RuntimeError('Fixture length changed')
    position = 0
    with reference.open('rb') as left,fixture.open('rb') as right:
        while True:
            a,b=left.read(1024*1024),right.read(1024*1024)
            if not a:
                break
            begin,end=max(offset-position,0),min(offset+2-position,len(a))
            if begin<end:
                a=a[:begin]+a[end:];b=b[:begin]+b[end:]
            if a!=b:
                raise RuntimeError('Fixture changed bytes beyond the declared FP16 word')
            position+=min(1024*1024,reference.stat().st_size-position)


class RawEngine:
    def __init__(self,binary,weights,directory):
        self.command=[str(binary),'--weights',str(weights),'--threads','12','--prefill-threads','6','--context','16']
        self.buffer=b''
        self.log_path=directory/'stderr.log'
        self.log=self.log_path.open('xb')
        self.process=None
        try:
            self.process=subprocess.Popen(self.command,stdin=subprocess.PIPE,stdout=subprocess.PIPE,
                stderr=self.log,bufsize=0,env=dict(os.environ,VECLIB_MAXIMUM_THREADS='6'))
            self.ready=self.read()
            if (self.ready.get('ready') is not True or self.ready.get('protocol')!=1
                    or self.ready.get('gpu') is not False or self.ready.get('pytorch') is not False
                    or self.ready.get('weight_precision')!='f16' or self.ready.get('kv_precision')!='f16'
                    or self.ready.get('context')!=16):
                raise RuntimeError('Unexpected raw CPU handshake')
        except BaseException:
            self.close();raise

    def read(self):
        deadline=time.monotonic()+180
        while b'\n' not in self.buffer:
            if time.monotonic()>deadline:raise TimeoutError('Native fallback test response timeout')
            if select.select([self.process.stdout],[],[],.1)[0]:
                chunk=os.read(self.process.stdout.fileno(),65536)
                if not chunk:raise RuntimeError('Native process closed stdout')
                self.buffer+=chunk
                if len(self.buffer)>1024*1024:raise RuntimeError('Oversized protocol response')
        line,self.buffer=self.buffer.split(b'\n',1)
        value=json.loads(line)
        if not isinstance(value,dict) or 'error' in value:raise RuntimeError(f'Native error: {value}')
        return value

    def request(self,command):
        if '\n' in command or '\r' in command:raise ValueError('Command contains newline')
        data=memoryview((command+'\n').encode())
        while data:
            count=self.process.stdin.write(data)
            if not count:raise RuntimeError('Native stdin closed')
            data=data[count:]
        return self.read()

    def close(self):
        if self.process is not None:
            if self.process.poll() is None:
                self.process.terminate()
                try:self.process.wait(timeout=3)
                except subprocess.TimeoutExpired:self.process.kill();self.process.wait(timeout=5)
            for pipe in (self.process.stdin,self.process.stdout):
                if pipe:pipe.close()
        self.log.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('reference','candidate','weights','normal-capture','directory'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--guard-state-dir',type=Path,default=ROOT/'build/benchmark-status')
    args=parser.parse_args()
    before_guard=guard(args.guard_state_dir.resolve())
    directory=args.directory.resolve();directory.mkdir(parents=True,exist_ok=False)
    reference,candidate,canonical,normal_path=(getattr(args,n).resolve() for n in ('reference','candidate','weights','normal_capture'))
    report=dict(completed=False,passed=False,diagnostic_only=True,timing_claim=False,
        performance_claim_eligible=False,gpu=False,pytorch=False,guard_before=before_guard,
        purpose='Intentional changed-weight fixture exercises whole-embedding BF16 fallback; not canonical model quality',runs=[])
    def save():
        temporary=directory/'report.pending.json';temporary.write_text(json.dumps(report,indent=2)+'\n');temporary.replace(directory/'report.json')
    canonical_hash=None
    try:
        save()
        index=Path(str(canonical)+'.index')
        inputs=[reference,candidate,canonical,index,normal_path,Path(__file__).resolve()]
        report['inputs']={str(p):sha(p) for p in inputs};canonical_hash=report['inputs'][str(canonical)]
        normal=json.loads(normal_path.read_text())
        if normal.get('completed') is not True or normal.get('prompt_ids')!=[9707]:
            raise RuntimeError('Require completed canonical tiny capture beginning with9707')
        if (normal['inputs'].get(str(candidate))!=report['inputs'][str(candidate)]
                or normal['inputs'].get(str(canonical))!=canonical_hash):
            raise RuntimeError('Normal capture candidate binary/weights hashes differ')
        normal_cache=normal['startup'].get(CACHE_KEY)
        if type(normal_cache) is not int or normal_cache<=0:raise RuntimeError('Normal capture lacks actual auxiliary-cache byte count')
        normal_row=next(row for row in normal['runs'] if row['warmup'] is False)
        if not normal_row.get('own_repeat_ids_equal') or not normal_row.get('own_repeat_logits_bit_identical'):
            raise RuntimeError('Normal capture did not pass exact repeat checks')
        normal_dump=normal_row['logit_dumps'][0];normal_file=Path(normal_dump['path'])
        if sha(normal_file)!=normal_dump['sha256']:raise RuntimeError('Canonical logit dump changed')
        rows=index.read_text().splitlines()
        if rows[0]!='QWCPU01 310':raise RuntimeError('Unexpected index header')
        embedding=[row.split() for row in rows[1:] if row.split()[0]=='model.embed_tokens.weight']
        if len(embedding)!=1:raise RuntimeError('Missing/duplicate embedding')
        name,dtype,nrows,ncols,offset,length=embedding[0]
        offset,length=int(offset),int(length)
        if (dtype,nrows,ncols)!=('F16','151936','1024') or length!=151936*1024*2 or not 64<=offset<=canonical.stat().st_size-length:
            raise RuntimeError('Unexpected embedding format/bounds')
        fixture=directory/'weights.bin';fixture_index=Path(str(fixture)+'.index')
        cloned=subprocess.run(['cp','-c',str(canonical),str(fixture)],capture_output=True,text=True)
        report['copy_method']='APFS clone' if cloned.returncode==0 else 'copy fallback'
        if cloned.returncode:
            fixture.unlink(missing_ok=True);shutil.copyfile(canonical,fixture)
        fixture.chmod(0o600);shutil.copyfile(index,fixture_index)
        with fixture.open('r+b') as stream:
            stream.seek(offset);original=stream.read(2)
            if original==b'\x01\x3c':raise RuntimeError('Embedding firstword already equals fixture mutation')
            stream.seek(offset);stream.write(b'\x01\x3c');stream.flush();os.fsync(stream.fileno())
        unchanged_except_word(canonical,fixture,offset)
        fixture.chmod(0o444);fixture_index.chmod(0o444)
        report['mutation']=dict(tensor=name,offset_bytes=offset,original_fp16_hex=original.hex(),
            replacement_fp16_bits='0x3c01',replacement_value=1.0009765625,
            explanation='FP16 widened FP32 bits0x3f802000 have nonzero low16bits, forbidding lossless BF16 representation',
            all_other_bytes_equal=True,fixture_has_manifest=False)
        report['fixture']=dict(path=str(fixture),sha256=sha(fixture),index_sha256=sha(fixture_index))
        for label,binary in [('reference',reference),('candidate',candidate)]:
            run_dir=directory/label;run_dir.mkdir()
            record=dict(label=label,binary=str(binary),predictions=[],dumps=[]);report['runs'].append(record)
            engine=None
            try:
                engine=RawEngine(binary,fixture,run_dir);record.update(startup=engine.ready,command=engine.command)
                if engine.request('mode greedy').get('ok') is not True:raise RuntimeError('Greedy mode failed')
                for step,command in enumerate(('prefill 9707','step 11')):
                    response=engine.request(command)
                    if type(response.get('token')) is not int or not 0<=response['token']<VOCABULARY or response.get('position')!=step+1:
                        raise RuntimeError('Unexpected inference response')
                    record['predictions'].append(response['token'])
                    dump=run_dir/f'step-{step}.f32';ack=engine.request('dump '+str(dump))
                    if ack.get('ok') is not True or ack.get('count')!=VOCABULARY or ack.get('position')!=step+1 or dump.stat().st_size!=VOCABULARY*4:
                        raise RuntimeError('Invalid full-logit dump')
                    if not all(math.isfinite(v[0]) for v in struct.iter_unpack('<f',dump.read_bytes())):
                        raise RuntimeError('Nonfinite fixture logits')
                    record['dumps'].append(dict(path=str(dump),sha256=sha(dump)))
                    save()
            finally:
                if engine:engine.close()
                if (run_dir/'stderr.log').exists():record['stderr_sha256']=sha(run_dir/'stderr.log')
                save()
        left,right=report['runs']
        if left['predictions']!=right['predictions'] or [d['sha256'] for d in left['dumps']]!=[d['sha256'] for d in right['dumps']]:
            raise RuntimeError('Fallback differs from original FP16 path in full logits or greedyIDs')
        if normal_cache-length!=right['startup'].get(CACHE_KEY):
            raise RuntimeError('Candidate did not drop exactly the whole embedding auxiliary cache')
        with normal_file.open('rb') as stream:normal_zero=stream.read(4)
        with Path(left['dumps'][0]['path']).open('rb') as stream:fixture_zero=stream.read(4)
        if normal_zero==fixture_zero:raise RuntimeError('Changed embedding/output-head row0 was not exercised')
        report['checks']=dict(full_logits_bit_identical=True,greedy_ids_equal=True,
            normal_cache_bytes=normal_cache,embedding_bytes=length,fixture_cache_bytes=right['startup'][CACHE_KEY],
            whole_tensor_fallback=True,normal_row0_logit=struct.unpack('<f',normal_zero)[0],
            fixture_row0_logit=struct.unpack('<f',fixture_zero)[0],changed_weight_exercised=True)
        if report['inputs']!={str(p):sha(p) for p in inputs}:raise RuntimeError('Inputs changed during fallback check')
        if sha(fixture)!=report['fixture']['sha256'] or sha(fixture_index)!=report['fixture']['index_sha256']:
            raise RuntimeError('Fixture changed during runtime check')
        report.update(guard_after=guard(args.guard_state_dir.resolve()),completed=True,passed=True)
    except BaseException as error:
        report['error']=dict(type=type(error).__name__,message=str(error));raise
    finally:
        if canonical_hash is not None:
            report['canonical_unchanged']=sha(canonical)==canonical_hash
            if not report['canonical_unchanged']:report.update(completed=False,passed=False)
        save()
    if not report['passed']:raise RuntimeError('Fallback gate failed')
    print(directory/'report.json')


if __name__=='__main__':main()
