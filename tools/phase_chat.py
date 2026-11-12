"""Trusted fixed-phase adapter; mathematical precision is the existing hybrid."""
import json
try:
    from .rtl_chat import RTLChatEngine
    from .rtl_model import PHASE_BINARY, FMA_PHASE_BINARY
except ImportError:
    from rtl_chat import RTLChatEngine
    from rtl_model import PHASE_BINARY, FMA_PHASE_BINARY


class PhasePoolChatEngine(RTLChatEngine):
    fixed_workers = 8
    numerical_profile = 'fp32_b1_fp64_prefill'

    def __init__(self, model=None, binary=PHASE_BINARY):
        super().__init__(model, workers=8, binary=binary, lanes=8, block_size=4,
                         prefill_size=8, reuse_prefix=False,
                         precision_profile=self.numerical_profile, canonical_prefix_reuse=True)
        self.sim.phase_pool_selector = 'PREFILL_12'
        self.phase_verified = False

    @property
    def threads(self):
        return 20  # Resident workers; only one pool receives work at a time.

    def status(self):
        return dict(super().status(), fixed_workers=8, prefill_workers=12, decode_workers=8,
                    resident_worker_threads=20, phase_pool_selector='PREFILL_12',
                    phase_verified=self.phase_verified)

    def close(self):
        self.phase_verified = False
        super().close()

    def verify_precision(self, cancel):
        self.phase_verified = False
        super().verify_precision(cancel)
        actual = json.loads(self.sim.command('phase_pools', cancel))
        expected = dict(selector='PREFILL_12', resident_decode_workers=8,
                        resident_prefill_workers=12, prefill_workers=12,
                        decode_workers=8, pool_constructions=2)
        if not isinstance(actual, dict) or any(type(actual.get(k)) is not type(v) or actual[k] != v
                                               for k, v in expected.items()):
            raise RuntimeError('Simulator does not match the fixed 12-prefill/8-decode policy')
        self.phase_verified = True

    def stream(self, request, cancel):
        if request.workers is not None and (type(request.workers) is not int or request.workers != 8):
            raise ValueError('This backend requires fixed 8 decode / 12 prefill workers')
        for event in super().stream(request, cancel):
            if 'metrics' in event:
                event['metrics'].update(prefill_workers=12, decode_workers=8,
                                        resident_worker_threads=20)
            yield event


class FusedPhasePoolChatEngine(PhasePoolChatEngine):
    """Explicit fused-prefill profile; separate decode/verification arithmetic."""
    numerical_profile = 'fp32_b1_fp32_fma_prefill'

    def __init__(self, model=None, binary=FMA_PHASE_BINARY):
        super().__init__(model=model, binary=binary)
