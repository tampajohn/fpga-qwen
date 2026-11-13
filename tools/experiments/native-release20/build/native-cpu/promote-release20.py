#!/usr/bin/env python3
"""Explicit guarded promotion with owned-server checks and build19 rollback."""
import argparse,signal,subprocess,sys,time,urllib.request,shlex,shutil
from release20_common import *
sys.path.insert(0,str(ROOT/'tools'))
from benchmark_native_phase0 import guard
p=argparse.ArgumentParser();p.add_argument('--candidate-tag',required=True);p.add_argument('--prerequisites',required=True);p.add_argument('--previous-promotion',required=True);args=p.parse_args()
os.chdir(ROOT);guard(ROOT/'build/benchmark-status')
prereq,j,_,_,_=selected(args.candidate_tag,args.prerequisites)
release=BASE/'build-20';staging=read(release/'staging.json');assert staging['completed'] and not staging.get('error')
unchanged(staging['helper_sha256'])
for name in ('core.cpp','core.hpp','main.cpp'):assert sha(release/'rollback-source'/name)==staging['shared_source_before'][str(ROOT/'engine'/name)]
assert staging['candidate_tag']==args.candidate_tag and staging['prerequisites_sha256']==sha(prereq)
assert sha(release/'qwen_cpu')==staging['binary_sha256'] and sha(release/'manifest.json')==staging['release_manifest_sha256']
assert all(sha(path)==digest for path,digest in staging['shared_source_after'].items())
browser_path=BASE/'build-20-browser-isolated/report.json';browser=read(browser_path)
launch=read(BASE/'build-20-browser-isolated-launch/report.json')
assert browser['completed'] and not browser.get('error') and not browser['console_errors']
assert launch['completed'] and launch['passed'] and not launch.get('error') and not launch['cleanup'].get('error')
assert launch['binary_sha256']==staging['binary_sha256'] and launch['browser_report_sha256']==sha(browser_path)
previous_path=Path(args.previous_promotion).resolve();previous=read(previous_path)
assert previous['completed'] and previous['target']=='build-19/qwen_cpu'
assert previous['binary_sha256']==sha(BASE/'build-19/qwen_cpu')
pid=int(previous['server_pid']);assert pid>1
command=shlex.split(subprocess.check_output(['ps','-p',str(pid),'-o','command='],text=True).strip())
assert len(command)==5 and Path(command[0]).resolve()==Path(sys.executable).resolve()
assert Path(command[1]).resolve()==ROOT/'tools/chat_server.py' and command[2:]==['--native-only','--port','8765']
listeners=set(subprocess.check_output(['lsof','-nP','-t','-iTCP:8765','-sTCP:LISTEN'],text=True).split());assert listeners=={str(pid)}
link=BASE/'qwen_cpu';assert link.is_symlink();old=os.readlink(link);assert old=='build-19/qwen_cpu'
record_path=BASE/'build-20-promotion.json';assert not record_path.exists()
log=BASE/'build-20-server.log';rollbacklog=BASE/'build-20-rollback19-server.log'
assert not log.exists() and not rollbacklog.exists()
def status(expected):
 with urllib.request.urlopen('http://127.0.0.1:8765/api/status',timeout=2) as response:s=json.load(response)
 assert s['build_id']==expected and s['default_backend']=='native-cpu-f16' and s['benchmark']['running']
 assert s['gpu_inference'] is False and s['pytorch_loaded'] is False
 return s
status('build-19')
record=dict(completed=False,target='build-20/qwen_cpu',old_target=old,previous_promotion=str(previous_path),previous_promotion_sha256=sha(previous_path),old_server_pid=pid,verified_old_command=command,binary_sha256=staging['binary_sha256'],prerequisites_sha256=sha(prereq),isolated_browser_sha256=sha(browser_path),known_absolute_failures=j['known_absolute_failures'],policy=j['policy'])
record['helper_sha256']=helper_hashes();write(record_path,record);server=None;stopped=False

def setlink(target):
 temp=BASE/'.qwen_cpu.release20.pending';assert not temp.exists();temp.symlink_to(target);os.replace(temp,link)
def start(path):
 with path.open('xb') as f:return subprocess.Popen([sys.executable,'tools/chat_server.py','--native-only','--port','8765'],cwd=ROOT,stdout=f,stderr=f,start_new_session=True)
def ready(proc,expected):
 for _ in range(150):
  if proc.poll() is not None:raise RuntimeError('Owned server exited')
  try:return status(expected)
  except (OSError,TimeoutError):time.sleep(.1)
 raise TimeoutError('Owned server readiness timeout')
try:
 unchanged(record['helper_sha256']);setlink('build-20/qwen_cpu');os.kill(pid,signal.SIGTERM)
 for _ in range(150):
  if subprocess.run(['ps','-p',str(pid)],stdout=subprocess.DEVNULL).returncode!=0:stopped=True;break
  time.sleep(.1)
 if not stopped:raise RuntimeError('Old server did not exit; no force-kill of unowned PID')
 server=start(log);record['server_pid']=server.pid;write(record_path,record)
 record['status']=ready(server,'build-20');record['completed']=True
except BaseException as error:
 record['error']=dict(type=type(error).__name__,message=str(error))
 try:
  if server and server.poll() is None:
   server.terminate()
   try:server.wait(timeout=15)
   except subprocess.TimeoutExpired:server.kill();server.wait(timeout=5)
  setlink(old)
  for name in ('core.cpp','core.hpp','main.cpp'):
   assert sha(release/'rollback-source'/name)==staging['shared_source_before'][str(ROOT/'engine'/name)]
   shutil.copyfile(release/'rollback-source'/name,ROOT/'engine'/name)
  rollback=dict(target=old,shared_source_restored=True)
  if stopped:
   fallback=start(rollbacklog);rollback['server_pid']=fallback.pid;rollback['status']=ready(fallback,'build-19')
  else:rollback['server_pid']=pid;rollback['status']=status('build-19')
  rollback['completed']=True;record['rollback']=rollback
 except BaseException as failure:record['rollback_error']=str(failure)
 raise
finally:write(record_path,record)
print(record_path)
