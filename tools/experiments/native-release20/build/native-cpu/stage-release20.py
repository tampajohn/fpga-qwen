#!/usr/bin/env python3
"""Root-selected guarded staging. Never changes live symlink/server."""
import argparse,os,shutil,sys
from release20_common import *
sys.path.insert(0,str(ROOT/'tools'))
from benchmark_native_phase0 import guard
p=argparse.ArgumentParser();p.add_argument('--candidate-tag',required=True);p.add_argument('--prerequisites',required=True);args=p.parse_args()
os.chdir(ROOT);guard(ROOT/'build/benchmark-status')
prereq,j,candidate,source,m=selected(args.candidate_tag,args.prerequisites)
old=read(BASE/'build-19/staging.json');assert old['completed'] and not old.get('error')
assert sha(BASE/'build-19/qwen_cpu')==old['binary_sha256']
before={str(ROOT/'engine'/n):sha(ROOT/'engine'/n) for n in ('core.cpp','core.hpp','main.cpp')}
assert before==old['shared_source_after'],'Use build19 staging shared-source hashes, not copied isolated manifest paths'
dest=BASE/'build-20';assert not dest.exists();dest.mkdir()
record=dict(completed=False,promoted=False,candidate_tag=args.candidate_tag,prerequisites_path=str(prereq),prerequisites_sha256=sha(prereq),shared_source_before=before,live_symlink_untouched=True,server_untouched=True,policy=j['policy'],known_absolute_failures=j['known_absolute_failures'])
record['helper_sha256']=helper_hashes();write(dest/'staging.json',record);changed=[]
try:
 shutil.copy2(candidate/'qwen_cpu',dest/'qwen_cpu');assert sha(dest/'qwen_cpu')==m['binary_sha256']
 shutil.copyfile(candidate/'manifest.json',dest/'original-manifest.json')
 if (candidate/'build.log').exists():shutil.copyfile(candidate/'build.log',dest/'build.log')
 shutil.copyfile(prereq,dest/'prerequisites.json')
 release=dict(m);release['binary']=str(dest/'qwen_cpu');release['copy_provenance']=dict(operation='Byte-identical copy, NOT a recompilation',original_binary=str(candidate/'qwen_cpu'),original_manifest=str(candidate/'manifest.json'),original_manifest_sha256=sha(candidate/'manifest.json'),original_binary_sha256=m['binary_sha256'],original_source_and_command_fields_preserved=True)
 write(dest/'manifest.json',release)
 (dest/'rollback-source').mkdir()
 for name in ('core.cpp','core.hpp','main.cpp'):
  shutil.copyfile(ROOT/'engine'/name,dest/'rollback-source'/name)
  temp=ROOT/'engine'/('.'+name+'.release20.pending');assert not temp.exists();shutil.copyfile(source/name,temp)
  assert sha(temp)==m['source_sha256'][str((source/name).resolve())]
 unchanged(record['helper_sha256']);selected(args.candidate_tag,prereq);assert all(sha(path)==h for path,h in before.items())
 for name in ('core.cpp','core.hpp','main.cpp'):
  os.replace(ROOT/'engine'/('.'+name+'.release20.pending'),ROOT/'engine'/name);changed.append(name)
 record.update(completed=True,binary_sha256=sha(dest/'qwen_cpu'),release_manifest_sha256=sha(dest/'manifest.json'),shared_source_after={str(ROOT/'engine'/n):sha(ROOT/'engine'/n) for n in ('core.cpp','core.hpp','main.cpp')})
except BaseException as error:
 for name in changed:shutil.copyfile(dest/'rollback-source'/name,ROOT/'engine'/name)
 record['error']=dict(type=type(error).__name__,message=str(error));raise
finally:
 for name in ('core.cpp','core.hpp','main.cpp'):(ROOT/'engine'/('.'+name+'.release20.pending')).unlink(missing_ok=True)
 write(dest/'staging.json',record)
print(dest/'staging.json')
