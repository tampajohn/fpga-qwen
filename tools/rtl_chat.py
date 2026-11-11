"""Browser chat through the full behavioral SystemVerilog model."""
import time
import sys
import random
try:
    from .chat_protocol import ChatProtocol, common_prefix
    from .rtl_model import RTLModel, SimulationCancelled
    from .suffix import SuffixDrafter, greedy_accept
except ImportError:
    from chat_protocol import ChatProtocol, common_prefix
    from rtl_model import RTLModel, SimulationCancelled
    from suffix import SuffixDrafter, greedy_accept


class RTLChatEngine(ChatProtocol):
    def __init__(self, model=None, workers=8, binary=None, lanes=1, block_size=1):
        self.sim = RTLModel(model, workers=workers, binary=binary, lanes=lanes, block_size=block_size)
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
                    prefill_block_size=self.sim.block_size, suffix_supported=self.sim.block_size == 4,
                    max_verified_draft=self.sim.block_size-1,
                    cached_tokens=len(self.cache_ids), pytorch_loaded=('torch' in sys.modules))

    def close(self):
        self.sim.close()
        self.cache_ids = []
        self.cache_session = None

    def stream(self, request, cancel):
        if request.max_draft and self.sim.block_size != 4:
            raise ValueError('Suffix drafting requires the block-enabled SystemVerilog backend')
        if not request.ids or len(request.ids)+request.max_new > 4096:
            raise ValueError('Request exceeds the simulator context limit')
        drafter = SuffixDrafter()
        draft_limit = min(request.max_draft, self.sim.block_size-1)
        rng = random.Random(request.seed)
        started = time.perf_counter()
        decoded_at = None
        generated = []
        text = ''
        reason = 'length'
        m = dict(prompt_tokens=len(request.ids), reused_prefix_tokens=0, output_tokens=0,
                 rounds=0, proposed_tokens=0, accepted_tokens=0, verified_tokens=0, tokens_per_round=0,
                 decode_tokens_per_second=0, first_token_ms=None, elapsed_seconds=0,
                 kv_tokens=0, context_limit=4096, prefill_block_size=self.sim.block_size,
                 requested_max_draft=request.max_draft, effective_max_draft=draft_limit)
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
            prefill = request.ids[reused:-1]
            for offset in range(0, len(prefill), self.sim.block_size):
                chunk = prefill[offset:offset+self.sim.block_size]
                if self.sim.block_size == 4:
                    results = self.sim.block(chunk, logits=False, cancel=cancel)
                else:
                    results = [self.sim.token(chunk[0], logits=False, cancel=cancel)]
                for token, (_, position, _) in zip(chunk, results):
                    self.cache_ids.append(token)
                    if position != len(self.cache_ids):
                        raise RuntimeError('SystemVerilog KV position mismatch')
                yield dict(type='prefill', processed_tokens=len(self.cache_ids), metrics=metrics())
            pending = request.ids[-1]
            decoded_at = time.perf_counter()
            while len(generated) < request.max_new:
                if cancel.is_set():
                    raise SimulationCancelled()
                remaining = request.max_new-len(generated)
                capacity = 4096-len(self.cache_ids)
                count = min(draft_limit, remaining-1, capacity-1)
                draft = drafter.propose(self.cache_ids+[pending], count)[:count] if count > 0 else []
                inputs = [pending]+draft
                if draft:
                    speculative_rng = random.Random()
                    speculative_rng.setstate(rng.getstate())
                    uniforms = ([speculative_rng.random() for _ in inputs]
                                if request.decoding == 'chat' else None)
                    results = self.sim.block(inputs, logits=True, uniforms=uniforms, cancel=cancel)
                    emitted, accepted = greedy_accept(draft, [result[0] for result in results])
                else:
                    results = [self.sim.token(pending, cancel=cancel,
                        uniform=rng.random() if request.decoding == 'chat' else None)]
                    emitted, accepted = [results[0][0]], 0
                if cancel.is_set():
                    raise SimulationCancelled()
                old_length = len(self.cache_ids)
                if len(results) != len(inputs) or any(result[1] != old_length+i+1 for i,result in enumerate(results)):
                    raise RuntimeError('SystemVerilog KV position mismatch')
                for index, token in enumerate(emitted):
                    if token in self.eos:
                        emitted = emitted[:index+1]
                        reason = 'eos'
                        break
                if len(emitted) != len(inputs):
                    if self.sim.command(f'truncate {old_length+len(emitted)}', cancel) != 'truncated':
                        raise RuntimeError('Invalid simulator truncation response')
                self.cache_ids.extend(inputs[:len(emitted)])
                # Rejected speculative predictions consume no caller RNG draws.
                if draft and request.decoding == 'chat':
                    for _ in emitted:
                        rng.random()
                generated.extend(emitted)
                m['rounds'] += 1
                m['verified_tokens'] += len(results)
                m['proposed_tokens'] += len(draft)
                m['accepted_tokens'] += min(accepted, len(emitted))
                m.update(output_tokens=len(generated), tokens_per_round=len(generated)/m['rounds'])
                if m['first_token_ms'] is None:
                    m['first_token_ms'] = (time.perf_counter()-started)*1000
                text = self.tokenizer.decode(generated, skip_special_tokens=True)
                yield dict(type='update', text=text, metrics=metrics())
                if reason == 'eos':
                    break
                pending = emitted[-1]
        except GeneratorExit:
            self.close()
            raise
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
