#!/usr/bin/env python3
"""Archive a completed base12/6 release20. No deployment, inference, git or selection."""
import difflib,hashlib,importlib.util,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];B=ROOT/'build/native-cpu';R=B/'build-20'
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(p):return json.loads(Path(p).read_text())
def require(ok,message):
 if not ok:raise RuntimeError(message)
def success(j):require(j.get('completed')is True and not j.get('error')and not j.get('cleanup_error'),'Incomplete/error artifact')
def main():
 sanitizer=B/'archive-final-native-comparisons-v1.py'
 spec=importlib.util.spec_from_file_location('archive_sanitizer',sanitizer);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
 sources={
 'staging':R/'staging.json','manifest':R/'manifest.json','original-manifest':R/'original-manifest.json',
 'prerequisites':R/'prerequisites.json','promotion':B/'build-20-promotion.json',
 'browser-isolated':B/'build-20-browser-isolated/report.json',
 'browser-isolated-launch':B/'build-20-browser-isolated-launch/report.json',
 'browser-live':B/'build-20-browser-live/report.json','live-browser-binding':B/'build-20-live-browser-binding.json',
 'live-controller-correction':B/'build-20-live-controller-correction.json'}
 for phase in ('vs19','vsllama'):sources['final-'+phase+'-plan']=B/f'prefill-query-row-pair-final-{phase}-plan-v1.json'
 data={k:read(p)for k,p in sources.items()}
 for k,j in data.items():
  if k!='prerequisites':success(j)
 st=data['staging'];m=data['manifest'];original=data['original-manifest'];p=data['prerequisites'];promotion=data['promotion'];launch=data['browser-isolated-launch'];binding=data['live-browser-binding']
 require(st['candidate_tag']==p['candidate_tag']=='prefill-query-row-pair','Wrong selected kernel')
 require(p['authorize_selection']is True and p.get('runtime_defaults')is None and m.get('runtime_defaults')is None,'Only unmodified12/6 serving release allowed')
 require(p['policy']=='exact_nonregression_with_inherited_baseline19_gap'and p['known_absolute_failures']==['boundary63','boundary64']and p['absolute_oracle_pass_claimed']is False,'Known failure policy missing')
 require(st['prerequisites_sha256']==sha(sources['prerequisites'])and st['release_manifest_sha256']==sha(sources['manifest']),'Staging archive binding mismatch')
 require(m['binary_sha256']==original['binary_sha256']==st['binary_sha256']==promotion['binary_sha256']==launch['binary_sha256']==binding['binary_sha256'],'Binary digest disagreement')
 require(m['copy_provenance']['original_manifest_sha256']==sha(sources['original-manifest'])and m['copy_provenance']['original_binary_sha256']==m['binary_sha256'],'Copy provenance mismatch')
 require(promotion['prerequisites_sha256']==sha(sources['prerequisites'])and promotion['isolated_browser_sha256']==sha(sources['browser-isolated']),'Promotion evidence mismatch')
 require(promotion['target']=='build-20/qwen_cpu'and not promotion.get('rollback')and not promotion.get('rollback_error'),'Promotion not successful')
 require(launch['passed']is True and not launch['cleanup'].get('error')and launch['cleanup']['owned_server_returncode']is not None,'Isolated cleanup incomplete')
 require(launch['staging_sha256']==sha(sources['staging'])and launch['browser_report_sha256']==sha(sources['browser-isolated']),'Isolated browser binding mismatch')
 require(binding['promotion_sha256']==sha(sources['promotion'])and binding['browser_sha256']==sha(sources['browser-live']),'Live browser binding mismatch')
 for key in ('browser-isolated','browser-live'):require(not data[key]['console_errors'],'Browser console errors')
 status=binding['status'];require(status['build_id']=='build-20'and status['default_backend']=='native-cpu-f16'and not status['gpu_inference']and not status['pytorch_loaded']and not status['benchmark']['running'],'Wrong final runtime status')
 require(status['decode_workers']==12 and status['prefill_workers']==6,'Serving worker count differs')
 correction=data['live-controller-correction'];executed_live=B/'run-release20-browser-live-v2.py';original_live=B/'run-release20-browser-live.py'
 require(correction['original_helper_sha256']==sha(original_live)and correction['executed_helper_sha256']==sha(executed_live)and Path(correction['executed_helper']).resolve()==executed_live.resolve(),'Live correction source mismatch')
 require(correction['binding_sha256']==sha(sources['live-browser-binding']),'Live correction evidence mismatch')
 original_text=original_live.read_text();actual_text=executed_live.read_text()
 corrected_prefix=original_text.replace("'http://127.0.0.1:8765','--expect-prefix-reuse'","'http://127.0.0.1:8765','--live','--expect-prefix-reuse'")
 require(corrected_prefix!=original_text and actual_text.startswith(corrected_prefix),'Unexpected corrected live controller change')
 require(actual_text[len(corrected_prefix):].strip().startswith("write(BASE/'build-20-live-controller-correction.json'"),'Missing correction record writer')
 # Small source provenance only: no executable/weights rehash and no network status query.
 helpers={Path(path):digest for path,digest in st['helper_sha256'].items()}
 for path,digest in helpers.items():require(sha(path)==digest,'Release helper changed: '+str(path))
 for name in ('core.cpp','core.hpp','main.cpp'):
  require(sha(ROOT/'engine'/name)==st['shared_source_after'][str(ROOT/'engine'/name)],'Shared source differs')
  require(sha(R/'rollback-source'/name)==st['shared_source_before'][str(ROOT/'engine'/name)],'Rollback source differs')
 source=Path(m['copy_provenance']['original_binary']).parent.parent/'prefill-query-row-pair-source'
 for name in ('core.cpp','core.hpp','main.cpp'):
  require(sha(source/name)==m['source_sha256'][str(source/name)]==sha(ROOT/'engine'/name),'Candidate source mismatch')
 # Distinguish bound executed sources from additional current-source snapshots.
 executed={str(path):digest for path,digest in helpers.items()}
 executed[str(executed_live)]=correction['executed_helper_sha256']
 builder=B/'prepare-release20-base12-prerequisites-v1.py'
 require(sha(builder)==p['builder_sha256'],'Prerequisites builder changed');executed[str(builder)]=p['builder_sha256']
 for phase in ('vs19','vsllama'):
  plan=data['final-'+phase+'-plan'];require(plan['candidate_tag']=='prefill-query-row-pair'and plan['phase']==phase,'Wrong final controller plan')
  for path,digest in plan['source_sha256'].items():
   require(sha(path)==digest,'Final comparison source changed');executed[path]=digest
 for path,digest in m['source_sha256'].items():
  if path in {str(source/n)for n in ('core.cpp','core.hpp','main.cpp')}:executed[path]=digest
 # Additional unbound helper snapshots are explicitly current source, not execution proof.
 snapshots={Path(path)for path in executed}|{Path(__file__).resolve(),sanitizer,B/'prepare-release20-base12-prerequisites-v1.py',B/'run-final-native-comparisons-v1.py'}
 snapshots|={ROOT/'tools'/n for n in ('benchmark_native_phase0.py','benchmark_native_cpu.py','analyze_native_phase0.py','native_cpu_chat.py','suffix.py','chat_backends.py')}
 snapshots|={source/n for n in ('core.cpp','core.hpp','main.cpp')}
 snapshots|={R/'rollback-source'/n for n in ('core.cpp','core.hpp','main.cpp')}
 inventory=ROOT/'reports/native-cpu-build-20-release-archive-v1.json';require(not inventory.exists(),'Inventory already exists')
 destination=ROOT/'tools/experiments/native-release20';require(not destination.exists(),'Source snapshot directory already exists')
 prepared=[]
 for name,path in sources.items():
  dest=ROOT/f'reports/native-cpu-build-20-{name}.json';require(not dest.exists(),'Archive already exists: '+str(dest))
  prepared.append((path,dest,module.clean(data[name]),sha(path)))
 source_bytes={path:path.read_bytes()for path in snapshots}
 source_hashes={str(path):hashlib.sha256(value).hexdigest()for path,value in source_bytes.items()}
 patch=''
 for name in ('core.cpp','core.hpp','main.cpp'):
  patch+=''.join(difflib.unified_diff((R/'rollback-source'/name).read_text().splitlines(True),(source/name).read_text().splitlines(True),fromfile='a/engine/'+name,tofile='b/engine/'+name))
 # All prerequisite checks precede publication. Incomplete archive writes are never marked complete.
 records=[];destination.mkdir(parents=True)
 for path,contents in source_bytes.items():
  relative=path.relative_to(ROOT);dest=destination/relative;dest.parent.mkdir(parents=True,exist_ok=True)
  with dest.open('xb')as stream:stream.write(contents)
  records.append(dict(source=str(relative),source_sha256=source_hashes[str(path)],snapshot=str(dest.relative_to(ROOT)),snapshot_sha256=sha(dest),provenance=('original frozen controller; parser rejected before browser execution'if path==original_live else 'corrected live controller executed; bound to correction record'if path==executed_live else 'source hash bound to release/plan/build evidence'if str(path)in executed else 'current source snapshot; not independently bound to execution')))
 patchpath=destination/'build19-to-build20.patch';patchpath.write_text(patch)
 for path,dest,cleaned,digest in prepared:
  require(sha(path)==digest,'Report changed during archive')
  cleaned['archive_provenance']=dict(source_report=str(path.relative_to(ROOT)),source_sha256=digest,sanitization='Repository/home paths and process identifiers/names sanitized using final comparison archive helper. Full evidence retained.')
  with dest.open('x')as stream:stream.write(json.dumps(cleaned,indent=2,allow_nan=False)+'\n')
  records.append(dict(source=str(path.relative_to(ROOT)),source_sha256=digest,archive=str(dest.relative_to(ROOT)),archive_sha256=sha(dest)))
 require(all(sha(path)==digest for path,digest in source_hashes.items()),'Source changed during archive')
 require(all(sha(path)==digest for path,_,_,digest in prepared),'Report changed during archive')
 result=dict(completed=True,candidate_tag='prefill-query-row-pair',release='build-20',decode_threads=12,prefill_threads=6,records=records,source_patch=dict(path=str(patchpath.relative_to(ROOT)),sha256=sha(patchpath)),archive_helper_sha256=sha(Path(__file__)),known_absolute_failures=['boundary63','boundary64'],absolute_oracle_pass_claimed=False,limitations=['Archive validates recorded binary digest agreement; it does not rehash executable or model payloads.','No new inference, timing, deployment or live status query. Browser status is bound historical evidence.','Source snapshots are exact bytes; report paths and process identifiers are sanitized.','Optional worker8 patches are excluded and must remain unapplied.','Original live controller omitted --live and failed argument parsing before browser execution; separately archived v2 supplied --live and produced the successful bound live evidence. Frozen original helper was not changed.'])
 with inventory.open('x')as stream:stream.write(json.dumps(result,indent=2,allow_nan=False)+'\n')
 print(inventory)
if __name__=='__main__':main()
