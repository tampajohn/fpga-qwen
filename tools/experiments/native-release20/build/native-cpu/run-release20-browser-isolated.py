#!/usr/bin/env python3
"""Owned isolated browser validation only; no live deployment mutation."""
import hashlib,json,os,socket,subprocess,sys,time,urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tools'))
from benchmark_native_phase0 import guard
from release20_common import helper_hashes,unchanged

def sha(path):
 with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
 os.chdir(ROOT);guard(ROOT/'build/benchmark-status')
 base=ROOT/'build/native-cpu';release=base/'build-20';binary=release/'qwen_cpu'
 staging_path=release/'staging.json';staging=json.loads(staging_path.read_text())
 unchanged(staging['helper_sha256'])
 assert staging['completed'] is True and not staging.get('error')
 assert sha(binary)==staging['binary_sha256']
 assert sha(release/'manifest.json')==staging['release_manifest_sha256']
 directory=base/'build-20-browser-isolated';logs=base/'build-20-browser-isolated-launch';isolated=base/'build-20-browser-guard'
 for path in (directory,logs,isolated):assert not path.exists(),f'Refusing existing destination: {path}'
 with socket.socket(socket.AF_INET,socket.SOCK_STREAM) as probe:
  probe.bind(('127.0.0.1',8766))
 logs.mkdir();report=dict(completed=False,passed=False,live_mutation=False,staging_sha256=sha(staging_path),binary_sha256=sha(binary),commands=[],cleanup={})
 report['helper_sha256']=helper_hashes()
 def save():(logs/'report.json').write_text(json.dumps(report,indent=2)+'\n')
 server=None;stdout=None;stderr=None
 try:
  stdout=(logs/'server.stdout.log').open('xb');stderr=(logs/'server.stderr.log').open('xb')
  command=[sys.executable,str(ROOT/'tests/native_cpu_browser_server.py'),'--binary',str(binary),'--guard-state-dir',str(isolated)]
  report['commands'].append(command);save()
  server=subprocess.Popen(command,stdout=stdout,stderr=stderr,cwd=ROOT)
  deadline=time.monotonic()+90;ready=None
  while time.monotonic()<deadline:
   if server.poll() is not None:raise RuntimeError(f'Owned isolated server exited {server.returncode}')
   try:
    with urllib.request.urlopen('http://127.0.0.1:8766/api/status',timeout=1) as response:ready=json.load(response)
    assert ready['default_backend']=='native-cpu-f16'
    assert ready['build_id']=='build-20' and ready['gpu_inference'] is False and ready['pytorch_loaded'] is False
    assert ready['benchmark']['running'] is False
    break
   except (OSError,TimeoutError):time.sleep(.1)
  else:raise TimeoutError('Isolated server readiness timeout')
  report['ready_status']=ready;guard(ROOT/'build/benchmark-status')
  command=[sys.executable,str(ROOT/'tests/browser_native_cpu.py'),'--url','http://127.0.0.1:8766',
           '--expect-prefix-reuse','--expect-greedy-suffix','--expect-norm-accumulator-f64',
           '--guard-state-dir',str(isolated),'--directory',str(directory)]
  report['commands'].append(command);save()
  with (logs/'browser.stdout.log').open('xb') as out,(logs/'browser.stderr.log').open('xb') as err:
   result=subprocess.run(command,stdout=out,stderr=err,cwd=ROOT,timeout=900)
  report['browser_returncode']=result.returncode
  assert result.returncode==0,'Browser suite failed; retain logs/evidence'
  browser=json.loads((directory/'report.json').read_text())
  assert browser['completed'] is True and not browser.get('error') and not browser['console_errors']
  assert sha(binary)==report['binary_sha256']
  unchanged(report['helper_sha256'])
  report.update(completed=True,passed=True,browser_report_sha256=sha(directory/'report.json'))
 except BaseException as error:
  report['error']=dict(type=type(error).__name__,message=str(error));raise
 finally:
  if server:
   try:
    if server.poll() is None:
     server.terminate()
     try:server.wait(timeout=15)
     except subprocess.TimeoutExpired:
      report['cleanup']['kill_required']=True;server.kill();server.wait(timeout=5)
    report['cleanup']['owned_server_returncode']=server.returncode
   except BaseException as error:
    report['cleanup']['error']=str(error);report['completed']=False;report['passed']=False
  for stream in (stdout,stderr):
   if stream:stream.close()
  save()
 if not report['passed']:raise RuntimeError('Isolated cleanup failed')
 print(directory/'report.json')
if __name__=='__main__':main()
