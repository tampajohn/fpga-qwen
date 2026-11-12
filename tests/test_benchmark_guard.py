"""Cross-process benchmark exclusion and HTTP enforcement; no model inference."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request

from tools.benchmark_guard import BenchmarkGate
from tools.chat_server import make_server

ROOT = Path(__file__).resolve().parents[1]


def eventually(predicate):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(.02)
    raise AssertionError('Expected process state was not reached')


class FakeEngine:
    prepared = 0

    def status(self):
        return dict(device='cpu', gpu_inference=False)

    def prepare(self, payload):
        self.prepared += 1
        return payload

    def stream(self, request, cancel):
        yield dict(type='done', text='fixture')


class BenchmarkGuardTests(unittest.TestCase):
    def test_completed_run_remains_visible_without_blocking(self):
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run([sys.executable, str(ROOT / 'tools/benchmark_guard.py'),
                '--state-dir', directory, '--name', 'Finished fixture', '--',
                sys.executable, '-c', 'raise SystemExit(7)'], capture_output=True)
            self.assertEqual(result.returncode, 7)
            status = BenchmarkGate(directory).status()
            self.assertFalse(status['running'])
            self.assertEqual(status['last_run']['name'], 'Finished fixture')
            self.assertEqual(status['last_run']['exit_code'], 7)
            self.assertTrue(status['last_run']['finished_at'])

    def test_chat_blocks_benchmark_and_stale_metadata_is_idle(self):
        with tempfile.TemporaryDirectory() as directory:
            gate = BenchmarkGate(directory)
            chat = gate.acquire()
            try:
                gate.describe('stale metadata')
                self.assertFalse(gate.status()['running'])
                result = subprocess.run([sys.executable, str(ROOT / 'tools/benchmark_guard.py'),
                                         '--state-dir', directory, '--name', 'blocked', '--',
                                         sys.executable, '-c', 'raise AssertionError("must not run")'],
                                        capture_output=True, text=True)
                self.assertEqual(result.returncode, 2)
            finally:
                chat.close()
            self.assertFalse(gate.status()['running'])

    def test_status_and_http_enforce_lock_then_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            engine = FakeEngine()
            server = make_server(engine, 0, directory)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            url = f'http://127.0.0.1:{server.server_port}'
            gate = BenchmarkGate(directory)
            handle = gate.acquire(exclusive=True)
            try:
                gate.describe('Named benchmark')
                with urllib.request.urlopen(url + '/api/status') as response:
                    status = json.load(response)
                self.assertTrue(status['benchmark']['running'])
                self.assertEqual(status['benchmark']['name'], 'Named benchmark')
                request = urllib.request.Request(url + '/api/chat',
                    data=json.dumps(dict(request_id='fixture')).encode(),
                    headers={'Content-Type': 'application/json'})
                with self.assertRaises(urllib.error.HTTPError) as error:
                    urllib.request.urlopen(request)
                self.assertEqual(error.exception.code, 409)
                error.exception.close()
                self.assertEqual(engine.prepared, 0)
                handle.close()
                with urllib.request.urlopen(request) as response:
                    self.assertIn(b'fixture', response.read())
                self.assertEqual(engine.prepared, 1)
                self.assertFalse(gate.status()['running'])
            finally:
                handle.close()
                server.shutdown()
                server.server_close()
                thread.join(timeout=2)

    def test_wrapper_crash_keeps_live_child_locked_then_clears(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pidfile, release = root / 'child.pid', root / 'release'
            child = ('import os,time; from pathlib import Path; '
                     f'Path({str(pidfile)!r}).write_text(str(os.getpid())); '
                     f'exec("while not Path({str(release)!r}).exists(): time.sleep(.02)")')
            wrapper = subprocess.Popen([sys.executable, str(ROOT / 'tools/benchmark_guard.py'),
                '--state-dir', directory, '--name', 'Crash fixture', '--', sys.executable, '-c', child],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            gate = BenchmarkGate(directory)
            try:
                eventually(pidfile.exists)
                self.assertTrue(gate.status()['running'])
                wrapper.kill()
                wrapper.wait(timeout=5)
                self.assertTrue(gate.status()['running'])
                release.touch()
                eventually(lambda: not gate.status()['running'])
            finally:
                release.touch()
                if wrapper.poll() is None:
                    wrapper.kill()
                    wrapper.wait(timeout=5)


if __name__ == '__main__':
    unittest.main()
