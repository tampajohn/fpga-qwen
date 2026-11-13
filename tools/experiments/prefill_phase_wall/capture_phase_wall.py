#!/usr/bin/env python3
"""Guarded diagnostic wall attribution, no benchmark or CPU-fraction claim."""
import argparse,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'tools'))
from benchmark_native_phase0 import guard,sha
from check_native_suffix_cli import Client

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--binary',type=Path,required=True)
    p.add_argument('--reference-capture',type=Path,required=True)
    p.add_argument('--directory',type=Path,required=True)
    p.add_argument('--weights',type=Path,default=ROOT/'build/native-cpu/weights.bin')
    p.add_argument('--guard-state-dir',type=Path,default=ROOT/'build/benchmark-status')
    a=p.parse_args();guard(a.guard_state_dir)
    out=a.directory.resolve();out.mkdir(exist_ok=False)
    r=dict(completed=False,passed=False,performance_claim_eligible=False,gpu=False,runs=[],scope='Instrumented coordinator wall phases; includes synchronization/contention, not CPU utilization or uninstrumented benchmark.',limits=['Timestamp/storage overhead is charged to the next phase.','Formatting and stderr I/O occur after append_ns.','One resident process, two fresh logical resets; first/repeated calls may differ; no statistical speed claim.'])
    client=None
    def save():
        (out/'report.json').write_text(json.dumps(r,indent=2)+'\n')
    try:
        refpath=a.reference_capture.resolve();ref=json.loads(refpath.read_text())
        assert ref['completed'] is True and len(ref['runs'])==2
        assert not ref.get('error') and not ref.get('cleanup_error')
        prompt=ref['prompt_ids'];teacher=ref['runs'][0]['consumed_continuation_ids']
        assert len(prompt) in (1024,3968) and ref['requested_steps']==2 and len(teacher)==1
        assert all(row['consumed_continuation_ids']==teacher for row in ref['runs'])
        binary=a.binary.resolve();weights=a.weights.resolve()
        mpath=binary.parent/'manifest.json';manifest=json.loads(mpath.read_text())
        assert manifest['completed'] is True and sha(binary)==manifest['binary_sha256']
        for path,digest in manifest['source_sha256'].items():assert sha(Path(path))==digest
        meta=Path(manifest['source_directory'])/'instrumentation.json'
        metadata=json.loads(meta.read_text());r['instrumentation']=metadata
        labels=metadata['phases']
        assert len(labels)==11 and len(set(labels))==11 and all(isinstance(v,str) and v for v in labels)
        assert metadata['diagnostic_only'] is True and metadata['base_build']=='build-18'
        base_manifest_path=ROOT/'build/native-cpu/build-18/manifest.json'
        base_manifest=json.loads(base_manifest_path.read_text())
        assert base_manifest['completed'] is True
        assert metadata['base_source_sha256']==base_manifest['source_sha256']
        for name in ('core.cpp','core.hpp','main.cpp'):
            source=(meta.parent/name).resolve()
            assert metadata['source_sha256'][name]==sha(source)==manifest['source_sha256'][str(source)]
        assert set(metadata['source_sha256'])=={'core.cpp','core.hpp','main.cpp'}
        paths=[refpath,binary,mpath,meta,weights,Path(str(weights)+'.index'),Path(str(weights)+'.manifest.json'),Path(__file__).resolve(),ROOT/'tools/check_native_suffix_cli.py',ROOT/'tools/benchmark_native_phase0.py']
        paths += [base_manifest_path]+[Path(path) for path in manifest['source_sha256']]
        for path,digest in ref['inputs'].items():
            assert sha(Path(path))==digest
            paths.append(Path(path))
        assert ref['inputs'][str(weights)]==sha(weights)
        pack=json.loads(Path(str(weights)+'.manifest.json').read_text())
        assert pack['completed'] is True and pack['binary']['sha256']==sha(weights)
        assert pack['index']['sha256']==sha(Path(str(weights)+'.index'))
        for row in ref['runs']:
            for dump in row['logit_dumps']:
                path=Path(dump['path']);assert sha(path)==dump['sha256'];paths.append(path)
        r['inputs']={str(path):sha(path) for path in paths}
        r.update(prompt_ids=prompt,consumed_continuation_ids=teacher,reference_capture=str(refpath),binary=str(binary))
        save();client=Client(binary,weights,out);r['ready']=client.ready
        for repeat in range(2):
            guard(a.guard_state_dir)
            assert client.command('reset')=={'ok':True,'position':0}
            assert client.command('mode greedy')['ok'] is True
            response=client.command('prefill '+' '.join(map(str,prompt)))
            row=dict(repeat=repeat,predicted_ids=[],dumps=[]);r['runs'].append(row)
            for step in range(2):
                assert response['position']==len(prompt)+step
                row['predicted_ids'].append(response['token'])
                dump=out/f'run{repeat}-step{step}.f32';ack=client.command('dump '+str(dump))
                assert ack.get('ok') is True and ack.get('count')==151936 and ack.get('position')==len(prompt)+step
                assert dump.stat().st_size==151936*4
                assert np.isfinite(np.fromfile(dump,dtype='<f4')).all()
                digest=sha(dump);expected=ref['runs'][repeat]['logit_dumps'][step]
                assert expected['context_tokens']==len(prompt)+step
                assert digest==expected['sha256'],'Instrumented logits differ from immutable18'
                assert response['token']==ref['runs'][repeat]['predicted_token_ids'][step]
                row['dumps'].append(dict(path=str(dump),sha256=digest,position=len(prompt)+step,exact_reference_match=True))
                if step==0:response=client.command('step '+str(teacher[0]))
            save()
        client.close();client=None
        records=[]
        for line in (out/'stderr.log').read_text().splitlines():
            try:item=json.loads(line)
            except json.JSONDecodeError:continue
            if item.get('diagnostic')=='prefill_phase_wall_v1':records.append(item)
        assert len(records)==4
        for i,item in enumerate(records):
            prefill=i%2==0
            assert item['prefill'] is prefill and item['base']==(0 if prefill else len(prompt))
            assert item['tokens']==(len(prompt) if prefill else 1)
            assert isinstance(item['head_ns'],int) and item['head_ns']>=0
            assert isinstance(item['append_ns'],int) and item['append_ns']>=0
            assert len(item['chunks'])==(item['tokens']+511)//512
            offset=0
            total=np.zeros(11,dtype=np.int64)
            for chunk in item['chunks']:
                assert chunk['base']==item['base']+offset
                assert chunk['batch']==min(512,item['tokens']-offset)
                offset+=chunk['batch']
                ns=np.asarray(chunk['layer_phase_ns'],dtype=np.int64)
                assert ns.shape==(29,11) and (ns>=0).all()
                assert (ns[:28,0]==0).all() and (ns[:28,10]==0).all()
                assert (ns[28,1:10]==0).all()
                total+=ns.sum(axis=0)
            assert offset==item['tokens']
            accounted=int(total.sum())+item['head_ns']
            assert 0<=accounted<=item['append_ns']
            item['phase_totals_ns']=dict(zip(metadata['phases'],map(int,total)))
            item['unattributed_ns']=item['append_ns']-accounted
        r.update(phase_records=records,stderr_sha256=sha(out/'stderr.log'))
        assert all(sha(Path(path))==digest for path,digest in r['inputs'].items())
        assert all(Path(d['path']).stat().st_size==151936*4 and sha(Path(d['path']))==d['sha256'] for row in r['runs'] for d in row['dumps'])
        r.update(completed=True,passed=True)
    except BaseException as e:
        r['error']=dict(type=type(e).__name__,message=str(e));raise
    finally:
        if client:client.close()
        save()
    print(out/'report.json')
if __name__=='__main__':main()
