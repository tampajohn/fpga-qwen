"""Pipe client for the Verilated KV metadata controller; no payload storage here."""
from dataclasses import dataclass
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class CursorState:
    error: int = 0
    active: int = 0
    committed: int = 0
    reserved: int = 0
    written: int = 0
    visible_end: int = 0


class RTLCursor:
    def __init__(self):
        self.process = subprocess.Popen([str(ROOT / 'build/kv/Vkv_cursor')],
                                        stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                        stderr=subprocess.PIPE, text=True, bufsize=1)
        self.state = CursorState()

    def command(self, op, count=0, *, allow_error=False):
        if op not in range(-1, 8) or not 0 <= count <= 65535:
            raise ValueError('Invalid command encoding')
        self.process.stdin.write(f'{op} {count}\n')
        self.process.stdin.flush()
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError('KV simulator stopped: ' + self.process.stderr.read())
        self.state = CursorState(*map(int, line.split()))
        if self.state.error and not allow_error:
            raise ValueError(f'KV command rejected: op={op}, count={count}, state={self.state}')
        return self.state

    def close(self):
        if self.process.stdin and not self.process.stdin.closed:
            self.process.stdin.close()
        try:
            result = self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait()
            raise RuntimeError('KV simulator did not terminate')
        error = self.process.stderr.read()
        self.process.stdout.close()
        self.process.stderr.close()
        if result:
            raise RuntimeError(error)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
