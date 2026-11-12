"""Selectable local CPU engines behind the browser's streaming protocol."""
import http.client
import json
from pathlib import Path
import socket
import subprocess
import time

try:
    from .native_cpu_chat import NativeCPUChat
    from .chat_protocol import ChatProtocol
except ImportError:
    from native_cpu_chat import NativeCPUChat
    from chat_protocol import ChatProtocol

ROOT = Path(__file__).resolve().parents[1]


class BackendRouter:
    def __init__(self, simulation=None, hybrid=None, phase=None, fused_phase=None, *, protocol=None):
        self.simulation = simulation
        self.protocol = protocol if protocol is not None else simulation
        if self.protocol is None:
            raise ValueError('A tokenizer ChatProtocol or existing simulation is required')
        self.native_cpu = NativeCPUChat(self.protocol)
        self.simulations = {"rtl": simulation} if simulation is not None else {}
        if hybrid is not None:
            self.simulations["rtl-fp32"] = hybrid
        if phase is not None:
            self.simulations['rtl-fp32-phase'] = phase
        if fused_phase is not None:
            self.simulations['rtl-fp32-fused-phase'] = fused_phase
        self.process = None
        self.loaded = None
        self.port = None
        self.log = None
        self.binary = ROOT / 'build/llama-cpu/bin/llama-server'
        self.models = {key: ROOT / f'build/bench/qwen3-0.6b-{kind}.gguf'
                       for key, kind in [('llama-q4', 'q4_k_m'), ('llama-f32', 'f32'), ('llama-f16', 'f16')]}
        self.active = self.default_backend()

    def default_backend(self):
        available = {entry['id'] for entry in self.catalog() if entry['available']}
        selected = next((key for key in ('native-cpu-f16', 'rtl-fp32-fused-phase',
                    'rtl-fp32-phase', 'rtl-fp32', 'rtl', 'llama-f16', 'llama-q4', 'llama-f32')
                    if key in available), None)
        if selected is None:
            raise RuntimeError('No available CPU backend. Build/export native assets explicitly, or install a comparison backend.')
        return selected

    def catalog(self):
        simulations = [dict(id=key,
            label=('SystemVerilog · Fused FP32 prefill · 12 prefill / 8 decode' if key == 'rtl-fp32-fused-phase' else
                   'SystemVerilog · FP32 hybrid · 12 prefill / 8 decode' if key == 'rtl-fp32-phase' else
                   'SystemVerilog · FP32 hybrid' if key == 'rtl-fp32' else
                   'SystemVerilog · fast' if engine.sim.lanes == 8 else 'SystemVerilog · reference'),
            available=True, cpu_threads=engine.threads, simulation=True,
            min_workers=1 if key.startswith('rtl-fp32') else 0,
            fixed_workers=getattr(engine,'fixed_workers',None),
            prefill_workers=engine.status().get('prefill_workers'),
            decode_workers=engine.status().get('decode_workers'),
            precision=engine.status()['precision'],prefix_reuse_supported=engine.status()["prefix_reuse_enabled"],
            canonical_prefix_policy=engine.status().get("canonical_prefix_policy"),
            suffix_supported=engine.sim.block_size == 4,max_draft=engine.sim.block_size-1,
            prefill_block_size=engine.sim.prefill_size) for key,engine in self.simulations.items()]
        if 'rtl' not in self.simulations:
            simulations.append(dict(id='rtl', label='SystemVerilog · fast', available=False,
                                    simulation=True, min_workers=0, suffix_supported=False))
        if 'rtl-fp32' not in self.simulations:
            simulations.append(dict(id='rtl-fp32',label='SystemVerilog · FP32 hybrid',
                                    available=False,simulation=True,min_workers=1,suffix_supported=False))
        if 'rtl-fp32-phase' not in self.simulations:
            simulations.append(dict(id='rtl-fp32-phase',label='SystemVerilog · FP32 hybrid · 12 prefill / 8 decode',
                                    available=False,simulation=True,min_workers=1,fixed_workers=8,suffix_supported=False))
        if 'rtl-fp32-fused-phase' not in self.simulations:
            simulations.append(dict(id='rtl-fp32-fused-phase',label='SystemVerilog · Fused FP32 prefill · 12 prefill / 8 decode',
                                    available=False,simulation=True,min_workers=1,fixed_workers=8,suffix_supported=False))
        native_build = self.native_cpu.binary.resolve().parent.name if self.native_cpu.binary.is_file() else 'not installed'
        return [dict(id='native-cpu-f16',
            label=f'Native CPU · FP16 · {native_build} · {self.native_cpu.threads}/{self.native_cpu.prefill_threads} threads',
            build_id=native_build, available=self.native_cpu.available(),
            cpu_threads=self.native_cpu.threads, prefill_workers=self.native_cpu.prefill_threads,
            simulation=False, prefix_reuse_supported=False, suffix_supported=False,
            max_draft=0, default_eligible=True)] + simulations + [
            dict(id=key, label=label, available=self.binary.is_file() and self.models[key].is_file(),
                 cpu_threads=6, suffix_supported=False)
            for key, label in [('llama-q4', 'llama.cpp · Q4_K_M'), ('llama-f16', 'llama.cpp · FP16'), ('llama-f32', 'llama.cpp · FP32')]]

    def status(self):
        if self.active in self.simulations:
            status = self.simulations[self.active].status()
        elif self.active == 'native-cpu-f16':
            status = self.native_cpu.status()
            status['build_id'] = self.native_cpu.binary.resolve().parent.name
        else:
            status = dict(model='Qwen3-0.6B', device='cpu', gpu_inference=False,
                          pytorch_loaded=False, context_limit=4096, precision={'llama-q4': 'Q4_K_M', 'llama-f16': 'FP16', 'llama-f32': 'FP32'}[self.active],
                          cpu_threads=6, rtl=None, inference='llama.cpp CPU', cached_tokens=None,
                          precision_profile=None,precision_verified=False,prefix_reuse_enabled=True,
                          suffix_supported=False, max_verified_draft=0, prefill_block_size=None,
                          reduction_lanes=None, prefill_reduction_lanes=None)
        return dict(status, backends=self.catalog(), active_backend=self.active, default_backend=self.default_backend())

    def prepare(self, payload):
        if not isinstance(payload, dict):
            raise ValueError('Expected a JSON object')
        backend = payload.get('backend', self.default_backend())
        entry = next((b for b in self.catalog() if b['id'] == backend), None)
        if not entry or not entry['available']:
            raise ValueError('Unknown or unavailable backend')
        simulation = self.simulations.get(backend)
        request = (simulation.prepare(dict(payload, max_draft=payload.get('max_draft', 0)))
                   if simulation is not None else ChatProtocol.prepare(self.protocol, payload))
        if request.max_draft and not entry['suffix_supported']:
            raise ValueError('Suffix drafting is unavailable for the selected backend')
        if backend in self.simulations:
            workers = payload.get('simulation_workers', simulation.sim.workers)
            if type(workers) is not int or workers not in (0, 1, 2, 4, 8):
                raise ValueError('Choose serial (0), 1, 2, 4, or 8 simulation workers')
            if workers < entry["min_workers"]:
                raise ValueError("Hybrid requires at least one simulation worker")
            if entry.get('fixed_workers') is not None and workers != entry['fixed_workers']:
                raise ValueError('This backend requires fixed 8 decode / 12 prefill workers')
            request.workers = workers
        return backend, request

    def close(self):
        for simulation in self.simulations.values():
            simulation.close()
        self.close_native()

    def close_native(self):
        self.native_cpu.close()
        if self.process:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        self.process = None
        self.loaded = None
        if self.log:
            self.log.close()
            self.log = None

    def start(self, backend, cancel):
        if self.loaded == backend and self.process and self.process.poll() is None:
            return
        self.close()
        # Fail closed if a locally rebuilt executable could contain GPU backends.
        cache = (ROOT / 'build/llama-cpu/CMakeCache.txt').read_text()
        for flag in ('METAL', 'CUDA', 'VULKAN', 'OPENCL'):
            if f'GGML_{flag}:BOOL=OFF' not in cache:
                raise RuntimeError(f'CPU-only build required: GGML_{flag} must be OFF')
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', 0))
            self.port = sock.getsockname()[1]
        self.log = (ROOT / 'build/llama-chat.log').open('w')
        self.process = subprocess.Popen([
            str(self.binary), '-m', str(self.models[backend]), '--host', '127.0.0.1',
            '--port', str(self.port), '-t', '6', '-tb', '6', '-c', '4096', '-np', '1',
            '-ngl', '0', '--no-kv-offload', '--no-op-offload', '--no-webui',
            '--no-warmup'], stdout=self.log, stderr=self.log)
        deadline = time.monotonic() + 90
        while time.monotonic() < deadline:
            if cancel.is_set():
                self.close()
                return
            if self.process.poll() is not None:
                raise RuntimeError('llama.cpp failed to start; see build/llama-chat.log')
            connection = http.client.HTTPConnection('127.0.0.1', self.port, timeout=1)
            try:
                connection.request('GET', '/health')
                if connection.getresponse().status == 200:
                    self.loaded = backend
                    return
            except OSError:
                pass
            finally:
                connection.close()
            cancel.wait(.1)
        self.close()
        raise RuntimeError('llama.cpp startup timed out')

    def stream(self, prepared, cancel):
        backend, request = prepared
        self.active = backend
        if backend == 'native-cpu-f16':
            if self.process is not None:
                self.close_native()
            for simulation in self.simulations.values():
                simulation.close()
            yield from self.native_cpu.stream(request, cancel)
            return
        self.native_cpu.close()
        if backend in self.simulations:
            self.close_native()
            for key,simulation in self.simulations.items():
                if key != backend:
                    simulation.close()
            yield from self.simulations[backend].stream(request, cancel)
            return
        started = time.perf_counter()
        text = ''
        metrics = dict(prompt_tokens=len(request.ids), reused_prefix_tokens=0,
                       output_tokens=0, rounds=0, proposed_tokens=0, accepted_tokens=0,
                       tokens_per_round=0, decode_tokens_per_second=0, first_token_ms=None,
                       elapsed_seconds=0, kv_tokens=len(request.ids), context_limit=4096)
        yield dict(type='loading', message='Loading local llama.cpp CPU backend…')
        self.start(backend, cancel)
        connection = None
        finished = False
        reason = 'stopped'
        try:
            if not cancel.is_set():
                yield dict(type='start', metrics=metrics)
                connection = http.client.HTTPConnection('127.0.0.1', self.port, timeout=90)
                connection.request('POST', '/completion', json.dumps(dict(
                    prompt=request.ids, n_predict=request.max_new,
                    temperature=.7 if request.decoding == 'chat' else 0,
                    top_k=20, top_p=.8, min_p=0, repeat_penalty=1,
                    presence_penalty=1.5 if request.decoding == 'chat' else 0,
                    repeat_last_n=256, seed=request.seed if request.seed is not None else -1,
                    stream=True, id_slot=0, cache_prompt=True, return_tokens=True,
                    timings_per_token=True, return_progress=True)), {'Content-Type': 'application/json'})
                response = connection.getresponse()
                if response.status != 200:
                    raise RuntimeError(f'llama.cpp request failed ({response.status})')
                while not cancel.is_set():
                    line = response.readline()
                    if not line:
                        break
                    if not line.startswith(b'data: '):
                        continue
                    event = json.loads(line[6:])
                    if 'error' in event:
                        raise RuntimeError(str(event['error']))
                    progress = event.get('prompt_progress')
                    if progress:
                        metrics['reused_prefix_tokens'] = progress.get('cache', 0)
                    text += event.get('content', '')
                    metrics['output_tokens'] += len(event.get('tokens', []))
                    timings = event.get('timings', {})
                    if timings:
                        metrics['output_tokens'] = timings.get('predicted_n', metrics['output_tokens'])
                        metrics['decode_tokens_per_second'] = timings.get('predicted_per_second', 0)
                        metrics['reused_prefix_tokens'] = timings.get('cache_n', metrics['reused_prefix_tokens'])
                    if metrics['output_tokens'] and metrics['first_token_ms'] is None:
                        metrics['first_token_ms'] = (time.perf_counter()-started)*1000
                    metrics['elapsed_seconds'] = time.perf_counter()-started
                    metrics['kv_tokens'] = len(request.ids) + metrics['output_tokens']
                    if event.get('stop'):
                        finished = True
                        reason = 'length' if event.get('stop_type') == 'limit' else 'eos'
                        break
                    yield dict(type='update', text=text, metrics=dict(metrics))
                if not finished and not cancel.is_set():
                    raise RuntimeError('llama.cpp stream ended before completion')
            metrics['elapsed_seconds'] = time.perf_counter()-started
            yield dict(type='done', text=text, metrics=metrics,
                       finish_reason='stopped' if cancel.is_set() else reason)
        finally:
            if connection:
                connection.close()
