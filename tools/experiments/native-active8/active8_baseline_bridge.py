"""Read-only root-run binding from build20 release evidence to old exact controls."""
from pathlib import Path
import hashlib,json
B=Path(__file__).resolve().parent
def sha(p):
 with Path(p).open('rb')as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(p):return json.loads(Path(p).read_text())
def bridge():
 release=B/'build-20';m=read(release/'manifest.json');p=read(release/'prerequisites.json');st=read(release/'staging.json')
 assert m['completed']and st['completed']and m['binary_sha256']==sha(release/'qwen_cpu')==st['binary_sha256']
 assert sha(release/'prerequisites.json')==st['prerequisites_sha256'] and sha(release/'manifest.json')==st['release_manifest_sha256']
 original=Path(m['copy_provenance']['original_binary']);assert sha(original)==m['binary_sha256']
 frozen={str(x):sha(x)for x in (Path(__file__).resolve(),release/'manifest.json',release/'prerequisites.json',release/'staging.json',release/'qwen_cpu',original)};refs={}
 for role,count in [('relative_exact',56),('timeboundary_exact',12),('newboundary_relative_exact',8)]:
  item=p['evidence'][role];assert sha(item['path'])==item['sha256'];j=read(item['path'])
  assert j['completed']and j['passed']and not j.get('error')and j['compared_arrays']==count
  assert j['inputs'][str(original)]==m['binary_sha256'];frozen[item['path']]=item['sha256']
  refs[role]={}
  for row in j['arrays']:
   assert row['bit_identical']and row['reference_sha256']==row['candidate_sha256']
   path=row['reference'];assert sha(path)==row['reference_sha256'];frozen[path]=row['reference_sha256'];refs[role][path]=row['reference_sha256']
 return frozen,refs
