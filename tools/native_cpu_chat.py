"""Original native CPU process adapter; tokenization only, no Python neural math."""
import json
import os
from pathlib import Path
import secrets
import select
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
EOS = {151643, 151645}


class NativeCancelled(Exception):
    pass


class NativeCPUChat:
    def __init__(self, protocol, binary=None, weights=None, threads=12, prefill_threads=6):
        self.protocol = protocol
        self.binary = Path(binary or ROOT / 'build/native-cpu/qwen_cpu')
        self.weights = Path(weights or ROOT / 'build/native-cpu/weights.bin')
        self.threads, self.prefill_threads = threads, prefill_threads
        self.process = None
        self.log = None
        self.buffer = b''
        self.ready = None

    def available(self):
        return (self.binary.is_file() and os.access(self.binary, os.X_OK)
                and self.weights.is_file()
                and Path(str(self.weights) + '.index').is_file()
                and Path(str(self.weights) + '.manifest.json').is_file())

    def status(self):
        return dict(model='Qwen3-0.6B', device='cpu', precision='FP16 matrices / FP32 norms / FP16 KV',
            gpu_inference=False, pytorch_loaded=False, cpu_threads=self.threads,
            prefill_workers=self.prefill_threads, decode_workers=self.threads,
            context_limit=4096, rtl=None, inference='Original native CPU engine (experimental)',
            cached_tokens=0, prefix_reuse_enabled=False, suffix_supported=False,
            max_verified_draft=0, prefill_block_size=None, precision_profile=None,
            precision_verified=False, reduction_lanes=None, prefill_reduction_lanes=None)

    def close(self):
        process, self.process = self.process, None
        if process is not None:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    process.kill(); process.wait(timeout=5)
            for pipe in (process.stdin, process.stdout):
                if pipe:
                    pipe.close()
        if self.log:
            self.log.close(); self.log = None
        self.buffer = b''
        self.ready = None

    def _read(self, cancel, timeout=180):
        deadline = time.monotonic() + timeout
        while True:
            if cancel.is_set():
                self.close()
                raise NativeCancelled()
            if b'\n' in self.buffer:
                line, self.buffer = self.buffer.split(b'\n', 1)
                value = json.loads(line)
                if not isinstance(value, dict) or value.get('error'):
                    raise RuntimeError(f'Native CPU protocol error: {value}')
                return value
            if not self.process or self.process.poll() is not None:
                raise RuntimeError('Native CPU process exited; see build/native-cpu/chat.log')
            if time.monotonic() >= deadline:
                raise TimeoutError('Native CPU process response timed out')
            if select.select([self.process.stdout], [], [], .05)[0]:
                block = os.read(self.process.stdout.fileno(), 65536)
                if not block:
                    raise RuntimeError('Native CPU stdout closed')
                self.buffer += block
                if len(self.buffer) > 1024 * 1024:
                    raise RuntimeError('Oversized native CPU protocol response')

    def command(self, command, cancel):
        if '\n' in command or '\r' in command:
            raise ValueError('Native command must occupy one line')
        try:
            if cancel.is_set():
                raise NativeCancelled()
            payload = memoryview((command + '\n').encode())
            while payload:
                written = self.process.stdin.write(payload)
                if not written:
                    raise RuntimeError('Native CPU stdin closed')
                payload = payload[written:]
            self.process.stdin.flush()
            return self._read(cancel)
        except BaseException:
            self.close()
            raise

    def start(self, cancel):
        if self.process is not None and self.process.poll() is None:
            return
        self.close()
        if not self.available():
            raise RuntimeError('Build and export native CPU assets explicitly before selecting this backend')
        manifest = json.loads(Path(str(self.weights) + '.manifest.json').read_text())
        if manifest.get('format') != 'QWCPU01' or manifest.get('completed') is not True:
            raise RuntimeError('Native weight pack manifest is incomplete')
        self.log = (self.binary.parent / 'chat.log').open('ab')
        command = [str(self.binary), '--weights', str(self.weights), '--threads', str(self.threads),
                   '--prefill-threads', str(self.prefill_threads), '--context', '4096']
        try:
            self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                            stderr=self.log, bufsize=0)
            self.ready = self._read(cancel)
            if (self.ready.get('ready') is not True or self.ready.get('protocol') != 1
                    or self.ready.get('gpu') is not False or self.ready.get('pytorch') is not False
                    or self.ready.get('weight_precision') != 'f16' or self.ready.get('kv_precision') != 'f16'
                    or self.ready.get('norm_precision') != 'f32' or self.ready.get('activation_precision') != 'f32'
                    or self.ready.get('decode_threads') != self.threads
                    or self.ready.get('prefill_threads') != self.prefill_threads or self.ready.get('context') != 4096):
                raise RuntimeError('Unexpected native CPU startup handshake')
        except BaseException:
            self.close()
            raise

    def stream(self, request, cancel):
        started = time.perf_counter()
        generated, text, first = [], '', None
        metrics = dict(prompt_tokens=len(request.ids), reused_prefix_tokens=0, output_tokens=0,
            rounds=0, proposed_tokens=0, accepted_tokens=0, tokens_per_round=0,
            decode_tokens_per_second=0, first_token_ms=None, elapsed_seconds=0,
            kv_tokens=0, context_limit=4096)
        reason = 'length'
        try:
            yield dict(type='loading', message='Loading original native CPU engine…')
            self.start(cancel)
            reset = self.command('reset', cancel)
            if reset.get('ok') is not True or reset.get('position') != 0:
                raise RuntimeError('Native reset did not clear the context')
            mode = 'mode greedy' if request.decoding == 'greedy' else f'mode chat {request.seed if request.seed is not None else secrets.randbits(32)}'
            if self.command(mode, cancel).get('ok') is not True:
                raise RuntimeError('Native decoding mode was not acknowledged')
            yield dict(type='start', metrics=dict(metrics))
            response = self.command('prefill ' + ' '.join(map(str, request.ids)), cancel)
            metrics['elapsed_seconds'] = time.perf_counter()-started
            yield dict(type='prefill', processed_tokens=len(request.ids), metrics=dict(metrics))
            for index in range(request.max_new):
                token, position = response.get('token'), response.get('position')
                if type(token) is not int or not 0 <= token < 151936 or position != len(request.ids) + index:
                    raise RuntimeError('Invalid native token or context position')
                metrics['kv_tokens'] = position
                if token in EOS:
                    reason = 'eos'
                    break
                generated.append(token)
                now = time.perf_counter()
                if first is None:
                    first = now
                    metrics['first_token_ms'] = (now-started)*1000
                text = self.protocol.tokenizer.decode(generated, skip_special_tokens=True)
                metrics.update(output_tokens=len(generated), rounds=len(generated), tokens_per_round=1,
                    elapsed_seconds=now-started,
                    decode_tokens_per_second=(len(generated)-1)/(now-first) if now>first else 0)
                yield dict(type='update', text=text, metrics=dict(metrics))
                if index + 1 < request.max_new:
                    response = self.command(f'step {token}', cancel)
            if cancel.is_set():
                raise NativeCancelled()
        except NativeCancelled:
            self.close()
            reason = 'stopped'
        except BaseException:
            self.close()
            raise
        metrics['elapsed_seconds'] = time.perf_counter()-started
        yield dict(type='done', text=text, metrics=dict(metrics), finish_reason=reason)
