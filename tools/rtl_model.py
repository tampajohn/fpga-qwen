"""Transport for the full behavioral SystemVerilog model. No tensor framework."""
import json
import select
import subprocess
import time
import threading
from pathlib import Path

try:
    from .experiment import ROOT, locate_model, read_header
except ImportError:
    from experiment import ROOT, locate_model, read_header

FAST_BINARY = ROOT / 'build/variants/scalar-O3-lto-unchecked-masked-activation-unsigned-indices-fast-attention-parallel-attention-batched-prefill-block-unroll-compact-projection-group-projections-prefill-b8-scalar-group/full/Vqwen_behavioral'

HYBRID_BINARY = FAST_BINARY.parents[1].with_name(FAST_BINARY.parents[1].name + '-fp32-b1-verify-b4') / 'full/Vqwen_behavioral'


def prepare_layout(model):
    model = locate_model(model)
    config = json.loads((model / 'config.json').read_text())
    expected = dict(model_type='qwen3', hidden_size=1024, intermediate_size=3072,
                    num_hidden_layers=28, num_attention_heads=16, num_key_value_heads=8,
                    head_dim=128, vocab_size=151936, rms_norm_eps=1e-6, rope_theta=1000000,
                    attention_bias=False, hidden_act='silu', use_sliding_window=False)
    if any(config.get(k) != v for k, v in expected.items()) or config.get('rope_scaling') is not None:
        raise ValueError('The simulator supports the original Qwen3-0.6B configuration only')
    header, start, _ = read_header(model / 'model.safetensors')
    specs = [('model.embed_tokens.weight', [151936, 1024]), ('model.norm.weight', [1024]),
             ('lm_head.weight' if 'lm_head.weight' in header else 'model.embed_tokens.weight', [151936, 1024])]
    for layer in range(28):
        for suffix, shape in [('input_layernorm.weight', [1024]),
                              ('self_attn.q_proj.weight', [2048, 1024]),
                              ('self_attn.k_proj.weight', [1024, 1024]),
                              ('self_attn.v_proj.weight', [1024, 1024]),
                              ('self_attn.o_proj.weight', [1024, 2048]),
                              ('self_attn.q_norm.weight', [128]),
                              ('self_attn.k_norm.weight', [128]),
                              ('post_attention_layernorm.weight', [1024]),
                              ('mlp.gate_proj.weight', [3072, 1024]),
                              ('mlp.up_proj.weight', [3072, 1024]),
                              ('mlp.down_proj.weight', [1024, 3072])]:
            specs.append((f'model.layers.{layer}.{suffix}', shape))
    offsets = []
    for name, shape in specs:
        tensor = header[name]
        if tensor['dtype'] != 'BF16' or tensor['shape'] != shape:
            raise ValueError(f'Unexpected checkpoint tensor: {name}')
        offsets.append((start + tensor['data_offsets'][0]) // 2)
    directory = ROOT / 'build/full'
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / 'offsets.txt'
    path.write_text(''.join(f'{offset}\n' for offset in offsets))
    return model, path


class SimulationCancelled(Exception):
    pass


class RTLModel:
    def __init__(self, model=None, workers=0, binary=None, lanes=1, block_size=1, prefill_size=None):
        if type(workers) is not int or not 0 <= workers <= 32:
            raise ValueError("Workers must be an integer from 0 to 32")
        if type(lanes) is not int or lanes not in (1, 8):
            raise ValueError('Reduction lanes must be 1 or 8')
        if type(block_size) is not int or block_size not in (1, 4):
            raise ValueError("Block size must be 1 or 4")
        if prefill_size is None:
            prefill_size = block_size
        if type(prefill_size) is not int or prefill_size not in (1, 4, 8) or prefill_size < block_size:
            raise ValueError('Prefill size must be 1, 4, or 8 and at least block size')
        if prefill_size != block_size and (prefill_size, block_size) not in ((8, 1), (8, 4)):
            raise ValueError('Prefill must match block size, except B8 prefill with B1 or B4 decode')
        self.prefill_size = prefill_size
        self.block_size = block_size
        self.lanes = lanes
        self.workers = workers
        self.binary = Path(binary).expanduser().resolve() if binary is not None else ROOT / 'build/full/Vqwen_behavioral'
        self.model, self.layout = prepare_layout(model)
        self.process = None
        self.log = None
        self.lifecycle_lock = threading.Lock()

    def close(self):
        # Shutdown and cancellation may arrive together. Detach ownership once.
        with self.lifecycle_lock:
            process, log = self.process, self.log
            self.process = None
            self.log = None
        if process:
            process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            process.stdin.close()
            process.stdout.close()
        if log:
            log.close()

    def read(self, cancel=None):
        deadline = time.monotonic() + 300
        while time.monotonic() < deadline:
            if cancel is not None and cancel.is_set():
                self.close()
                raise SimulationCancelled()
            readable, _, _ = select.select([self.process.stdout], [], [], .05)
            if readable:
                line = self.process.stdout.readline()
                if not line:
                    raise RuntimeError('SystemVerilog simulation exited; see build/full/runtime.log')
                return line.strip()
        self.close()
        raise RuntimeError('SystemVerilog simulation timed out')

    def start(self, cancel=None):
        if self.process and self.process.poll() is None:
            return
        self.close()
        self.log = (ROOT / 'build/full/runtime.log').open('w')
        self.process = subprocess.Popen([str(self.binary),
            str(self.model / 'model.safetensors'), str(self.layout), str(self.workers)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log, text=True, bufsize=1)
        if self.read(cancel) != 'ready':
            raise RuntimeError('Invalid simulator handshake')
        if self.lanes != 1:
            if self.command(f'lanes {self.lanes}', cancel) != 'configured':
                raise RuntimeError('Invalid reduction configuration response')

    def command(self, command, cancel=None):
        self.process.stdin.write(command + '\n')
        self.process.stdin.flush()
        return self.read(cancel)

    def set_lanes(self, lanes, cancel=None):
        if type(lanes) is not int or lanes not in (1, 8):
            raise ValueError('Reduction lanes must be 1 or 8')
        if lanes == self.lanes:
            return
        if self.process and self.process.poll() is None:
            if self.command(f'lanes {lanes}', cancel) != 'configured':
                raise RuntimeError('Invalid reduction configuration response')
        self.lanes = lanes

    def set_workers(self, workers, cancel=None):
        if type(workers) is not int or not 0 <= workers <= 32:
            raise ValueError('Workers must be an integer from 0 to 32')
        if workers == self.workers:
            return
        if self.process and self.process.poll() is None:
            if self.command(f'workers {workers}', cancel) != 'configured':
                raise RuntimeError('Invalid worker configuration response')
        self.workers = workers

    def reset(self, cancel=None):
        if self.command('reset', cancel) != 'reset':
            raise RuntimeError('Invalid reset response')

    def token(self, token, logits=True, cancel=None, uniform=None, presence_penalty=1.5):
        if uniform is None:
            command = f'token {token} {int(logits)}'
        else:
            if not 0 <= uniform < 1 or not 0 <= presence_penalty <= 2:
                raise ValueError('Invalid sampling parameters')
            command = f'sample {token} {int(logits)} {uniform:.17g} {presence_penalty:.17g}'
        response = self.command(command, cancel).split()
        if len(response) != 3:
            raise RuntimeError('Invalid simulator token response')
        return int(response[0]), int(response[1]), float(response[2])

    def block(self, tokens, logits=False, uniforms=None, presence_penalty=1.5, cancel=None):
        """Transport an explicitly enabled, causal B<=4 SV block; no host math."""
        tokens = list(tokens)
        if self.block_size != 4:
            raise ValueError('This simulator is not configured for block execution')
        if not 1 <= len(tokens) <= 4 or any(type(token) is not int or not 0 <= token < 151936 for token in tokens):
            raise ValueError('A block requires 1–4 valid token IDs')
        sampled = uniforms is not None
        uniforms = list(uniforms) if sampled else [0.0]*len(tokens)
        if len(uniforms) != len(tokens) or any(not 0 <= value < 1 for value in uniforms):
            raise ValueError('Provide one sampling uniform in [0, 1) per block token')
        if not 0 <= presence_penalty <= 2:
            raise ValueError('Invalid presence penalty')
        command = ' '.join(['block', str(len(tokens)), str(int(logits)), str(int(sampled)),
                            format(presence_penalty, '.17g'), *map(str, tokens),
                            *(format(value, '.17g') for value in uniforms)])
        response = self.command(command, cancel).split()
        if len(response) != 2+3*len(tokens) or response[:2] != ['block', str(len(tokens))]:
            raise RuntimeError('Invalid simulator block response')
        return [(int(response[i]), int(response[i+1]), float(response[i+2]))
                for i in range(2, len(response), 3)]

    def prefill(self, tokens, cancel=None):
        """No-logits prefill; wider capacity is explicitly configured by caller."""
        tokens = list(tokens)
        if not 1 <= len(tokens) <= self.prefill_size or any(type(t) is not int or not 0 <= t < 151936 for t in tokens):
            raise ValueError('Invalid prefill tokens or capacity')
        if self.prefill_size != 8:
            if self.block_size == 4:
                return self.block(tokens, logits=False, cancel=cancel)
            return [self.token(tokens[0], logits=False, cancel=cancel)]
        response = self.command(' '.join(['prefill', str(len(tokens)), *map(str, tokens)]), cancel).split()
        if len(response) != 3 or response[:2] != ['prefill', str(len(tokens))]:
            raise RuntimeError('Invalid simulator prefill response')
        end = int(response[2])
        if not len(tokens) <= end <= 4096:
            raise RuntimeError('Invalid simulator prefill position')
        return [(0, end-len(tokens)+i+1, 0.0) for i in range(len(tokens))]
