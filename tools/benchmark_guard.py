"""Run CPU benchmarks with a browser-visible, process-owned exclusion lock.

Use this wrapper from the serving checkout even when the command runs in a
different worktree: python tools/benchmark_guard.py --name 'Run name' -- COMMAND
The child inherits the lock: a wrapper crash cannot unlock a still-running job.
The OS releases it when its last owner exits; stale metadata never blocks chat.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


class BenchmarkGate:
    def __init__(self, directory=None):
        self.directory = Path(directory) if directory is not None else ROOT / 'build/benchmark-status'

    def acquire(self, exclusive=False):
        self.directory.mkdir(parents=True, exist_ok=True)
        handle = (self.directory / 'activity.lock').open('a+')
        try:
            fcntl.flock(handle, (fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH) | fcntl.LOCK_NB)
        except BlockingIOError:
            handle.close()
            return None
        return handle

    def status(self):
        handle = self.acquire()
        try:
            metadata = json.loads((self.directory / 'run.json').read_text())
            if not isinstance(metadata, dict):
                metadata = {}
        except (OSError, ValueError):
            metadata = {}
        if handle is not None:
            handle.close()
            last_run = {key: metadata.get(key) for key in ('name', 'started_at', 'finished_at', 'exit_code')}
            return dict(running=False, last_run=last_run if metadata.get('started_at') else None)
        return dict(running=True, name=str(metadata.get('name', 'CPU benchmark')),
                    started_at=metadata.get('started_at'))

    def describe(self, name):
        temporary = self.directory / f'run.{os.getpid()}.tmp'
        temporary.write_text(json.dumps(dict(name=name, pid=os.getpid(),
                                             started_at=datetime.now(timezone.utc).isoformat())) + '\n')
        temporary.replace(self.directory / 'run.json')

    def finish(self, exit_code):
        metadata = json.loads((self.directory / 'run.json').read_text())
        metadata.update(finished_at=datetime.now(timezone.utc).isoformat(), exit_code=exit_code)
        temporary = self.directory / f'run.{os.getpid()}.tmp'
        temporary.write_text(json.dumps(metadata) + '\n')
        temporary.replace(self.directory / 'run.json')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name', required=True)
    parser.add_argument('--state-dir', type=Path)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command or not args.name.strip() or len(args.name) > 200:
        parser.error('Provide a short run name and a command after --')
    gate = BenchmarkGate(args.state_dir)
    handle = gate.acquire(exclusive=True)
    if handle is None:
        parser.exit(2, 'Chat or another benchmark is running; wait for it to finish.\n')
    process = None
    exit_code = 1
    described = False
    try:
        gate.describe(args.name)
        described = True
        process = subprocess.Popen(command, pass_fds=(handle.fileno(),), start_new_session=True)
        exit_code = process.wait()
        return exit_code
    except KeyboardInterrupt:
        exit_code = 130
        return 130
    finally:
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        try:
            if described:
                gate.finish(exit_code)
        finally:
            handle.close()


if __name__ == '__main__':
    sys.exit(main())
