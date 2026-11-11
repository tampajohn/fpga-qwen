"""Browser chat through the full behavioral SystemVerilog model."""
import time
import sys
import random
try:
    from .chat_protocol import ChatProtocol, common_prefix
    from .rtl_model import RTLModel, SimulationCancelled
except ImportError:
    from chat_protocol import ChatProtocol, common_prefix
    from rtl_model import RTLModel, SimulationCancelled


class RTLChatEngine(ChatProtocol):
    def __init__(self, model=None, workers=8, binary=None, lanes=1):
        self.sim = RTLModel(model, workers=workers, binary=binary, lanes=lanes)
        super().__init__(self.sim.model)
        self.cache_ids = []
        self.cache_session = None
        self.eos = {151643, 151645}

    @property
    def threads(self):
        return max(1, self.sim.workers)

    def status(self):
        return dict(model='Qwen3-0.6B', device='cpu', precision='BF16 weights / SV real arithmetic',
                    gpu_inference=False, cpu_threads=self.threads, simulation_workers=self.sim.workers, context_limit=4096,
                    rtl='Full behavioral SystemVerilog; not synthesizable or cycle-accurate',
                    inference='Verilator: all model math and KV payload in SystemVerilog',
                    reduction_lanes=self.sim.lanes if self.sim.workers else 1,
                    cached_tokens=len(self.cache_ids), pytorch_loaded=('torch' in sys.modules))

    def close(self):
        self.sim.close()
        self.cache_ids = []
        self.cache_session = None

    def stream(self, request, cancel):
        if request.max_draft:
            raise ValueError('Suffix drafting is not implemented in the full SystemVerilog backend')
        rng = random.Random(request.seed)
        started = time.perf_counter()
        decoded_at = None
        generated = []
        text = ''
        reason = 'length'
        m = dict(prompt_tokens=len(request.ids), reused_prefix_tokens=0, output_tokens=0,
                 rounds=0, proposed_tokens=0, accepted_tokens=0, tokens_per_round=0,
                 decode_tokens_per_second=0, first_token_ms=None, elapsed_seconds=0,
                 kv_tokens=0, context_limit=4096)
        def metrics():
            m['elapsed_seconds'] = time.perf_counter()-started
            m['kv_tokens'] = len(self.cache_ids)
            m['simulation_workers'] = self.sim.workers
            m['decoding'] = request.decoding
            m['reduction_lanes'] = self.sim.lanes if self.sim.workers else 1
            if decoded_at is not None:
                m['decode_tokens_per_second'] = len(generated)/max(time.perf_counter()-decoded_at, 1e-9)
            return dict(m)
        try:
            yield dict(type='loading', message='Starting full SystemVerilog simulation…')
            if not self.sim.process or self.sim.process.poll() is not None:
                self.cache_ids = []
                self.cache_session = None
            if request.workers is not None:
                self.sim.set_workers(request.workers, cancel)
            self.sim.start(cancel)
            if self.cache_session != request.session:
                self.sim.reset(cancel)
                self.cache_ids = []
                self.cache_session = request.session
            reused = common_prefix(self.cache_ids, request.ids[:-1])
            if len(self.cache_ids) != reused:
                if self.sim.command(f'truncate {reused}', cancel) != 'truncated':
                    raise RuntimeError('Invalid simulator truncation response')
                self.cache_ids = self.cache_ids[:reused]
            m['reused_prefix_tokens'] = reused
            yield dict(type='start', metrics=metrics())
            for token in request.ids[reused:-1]:
                _, position, _ = self.sim.token(token, logits=False, cancel=cancel)
                self.cache_ids.append(token)
                if position != len(self.cache_ids):
                    raise RuntimeError('SystemVerilog KV position mismatch')
                yield dict(type='prefill', processed_tokens=len(self.cache_ids), metrics=metrics())
            pending = request.ids[-1]
            decoded_at = time.perf_counter()
            for _ in range(request.max_new):
                token, position, _ = self.sim.token(pending, cancel=cancel,
                    uniform=rng.random() if request.decoding == 'chat' else None)
                self.cache_ids.append(pending)
                if position != len(self.cache_ids):
                    raise RuntimeError('SystemVerilog KV position mismatch')
                generated.append(token)
                m.update(output_tokens=len(generated), rounds=len(generated), tokens_per_round=1)
                if m['first_token_ms'] is None:
                    m['first_token_ms'] = (time.perf_counter()-started)*1000
                text = self.tokenizer.decode(generated, skip_special_tokens=True)
                yield dict(type='update', text=text, metrics=metrics())
                if token in self.eos:
                    reason = 'eos'
                    break
                pending = token
        except SimulationCancelled:
            reason = 'stopped'
            self.close()
        except Exception:
            self.close()
            raise
        finally:
            # A disconnected browser can close the generator between tokens.
            if cancel.is_set():
                self.close()
        yield dict(type='done', text=text, finish_reason=reason, metrics=metrics())
