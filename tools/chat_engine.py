"""CPU-only reference inference with prefix reuse, suffix drafting and RTL KV metadata."""
import os

# Set before importing torch. CUDA is hidden; MPS is excluded by explicit CPU
# placement of the model, every input tensor, and checks on all model buffers.
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

from dataclasses import dataclass
import threading
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from .experiment import locate_model
    from .kv import RTLCursor
    from .suffix import SuffixDrafter, greedy_accept
except ImportError:
    from experiment import locate_model
    from kv import RTLCursor
    from suffix import SuffixDrafter, greedy_accept


SYSTEM_PROMPT = 'You are a helpful assistant. Answer clearly and concisely.'
CONTEXT_LIMIT = 4096


@dataclass
class ChatRequest:
    session: str
    ids: list[int]
    max_new: int
    max_draft: int


def common_prefix(a, b):
    n = 0
    for left, right in zip(a, b):
        if left != right:
            break
        n += 1
    return n


class ChatEngine:
    """One model and one hot conversation cache; callers serialize generation."""
    def __init__(self, model=None, threads=4):
        if not 1 <= threads <= 64:
            raise ValueError('CPU threads must be between 1 and 64')
        torch.set_num_threads(threads)
        path = locate_model(model)
        self.tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            path, local_files_only=True, dtype=torch.float32,
            attn_implementation='eager').to(torch.device('cpu')).eval()
        self.assert_cpu()
        eos = self.model.generation_config.eos_token_id
        self.eos = set(eos if isinstance(eos, list) else [eos])
        self.threads = threads
        self.cache = None
        self.cache_ids = []
        self.cache_session = None

    def assert_cpu(self):
        if any(t.device.type != 'cpu' for t in
               list(self.model.parameters()) + list(self.model.buffers())):
            raise RuntimeError('CPU-only requirement violated: model has a non-CPU tensor')

    def status(self):
        self.assert_cpu()
        return dict(model='Qwen3-0.6B', device='cpu', precision='FP32',
                    gpu_inference=False, cpu_threads=self.threads,
                    context_limit=CONTEXT_LIMIT, rtl='KV metadata controller',
                    inference='PyTorch CPU reference; not full-model RTL',
                    cached_tokens=len(self.cache_ids))

    def prepare(self, payload):
        if not isinstance(payload, dict):
            raise ValueError('Expected a JSON object')
        session = payload.get('session')
        if not isinstance(session, str) or not 1 <= len(session) <= 100:
            raise ValueError('Missing or invalid session identifier')
        messages = payload.get('messages')
        if not isinstance(messages, list) or not 1 <= len(messages) <= 63 or len(messages) % 2 != 1:
            raise ValueError('Send a conversation ending with a user message (up to 32 turns)')
        total = 0
        for i, message in enumerate(messages):
            role = 'user' if i % 2 == 0 else 'assistant'
            if not isinstance(message, dict) or message.get('role') != role:
                raise ValueError('Messages must alternate user and assistant roles')
            text = message.get('content')
            if not isinstance(text, str) or not text.strip() or len(text) > 12000:
                raise ValueError('Each message must contain 1–12,000 characters')
            total += len(text)
        if total > 40000:
            raise ValueError('Conversation is too long. Start a new chat.')
        max_new = payload.get('max_new_tokens', 256)
        max_draft = payload.get('max_draft', 7)
        if type(max_new) is not int or not 1 <= max_new <= 512:
            raise ValueError('Reply limit must be between 1 and 512 tokens')
        if type(max_draft) is not int or not 0 <= max_draft <= 7:
            raise ValueError('Draft length must be between 0 and 7')
        text = self.tokenizer.apply_chat_template(
            [{'role': 'system', 'content': SYSTEM_PROMPT}] + messages,
            tokenize=False, add_generation_prompt=True, enable_thinking=False)
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        if len(ids) + max_new > CONTEXT_LIMIT:
            raise ValueError(f'Context is full: {len(ids)} prompt tokens plus {max_new} reserved '
                             f'for the reply exceeds {CONTEXT_LIMIT}. Start a new chat or shorten the reply.')
        return ChatRequest(session, ids, max_new, max_draft)

    def clear_cache(self):
        self.cache = None
        self.cache_ids = []
        self.cache_session = None

    def _forward(self, ids, *, prefill=False):
        inputs = torch.tensor([ids], dtype=torch.long, device='cpu')
        kwargs = {'logits_to_keep': 1} if prefill else {}
        result = self.model(input_ids=inputs, past_key_values=self.cache, use_cache=True, **kwargs)
        if result.logits.device.type != 'cpu':
            raise RuntimeError('CPU-only requirement violated: non-CPU model output')
        self.cache = result.past_key_values
        return result

    def stream(self, request, cancel=None):
        cancel = cancel or threading.Event()
        # inference_mode must cover generator execution, not just its creation.
        with torch.inference_mode():
            yield from self._stream(request, cancel)

    def _stream(self, request, cancel):
        self.assert_cpu()
        started = time.monotonic()
        ids = request.ids
        if self.cache_session != request.session:
            self.clear_cache()
        self.cache_session = request.session
        prefix = ids[:-1]
        reused = common_prefix(self.cache_ids, prefix)
        if self.cache is not None:
            self.cache.crop(reused)
        self.cache_ids = prefix[:reused]
        generated, rounds, proposed, accepted_total = [], 0, 0, 0
        first_token_ms = None
        decode_start = None
        reason = 'length'
        proposer = SuffixDrafter()

        def metrics():
            now = time.monotonic()
            decode_seconds = now - decode_start if decode_start is not None else 0
            return dict(prompt_tokens=len(ids), reused_prefix_tokens=reused,
                        output_tokens=len(generated), rounds=rounds, proposed_tokens=proposed,
                        accepted_tokens=accepted_total,
                        tokens_per_round=len(generated)/rounds if rounds else 0,
                        decode_tokens_per_second=len(generated)/decode_seconds if decode_seconds else 0,
                        first_token_ms=first_token_ms, elapsed_seconds=now-started,
                        kv_tokens=len(self.cache_ids), context_limit=CONTEXT_LIMIT)

        yield dict(type='start', metrics=metrics(), device='cpu', rtl='KV metadata controller')
        try:
            with RTLCursor() as cursor:
                # Reconstruct ownership of the hot prefix; no model re-prefill.
                for first in range(0, reused, 8):
                    if cancel.is_set():
                        break
                    n = min(8, reused-first)
                    cursor.command(0, n)
                    cursor.command(1, n)
                    cursor.command(2, n)
                # Bounded prefill chunks make cancellation responsive. Metadata
                # is committed only after the model has written every layer's KV.
                for first in range(reused, len(prefix), 128):
                    if cancel.is_set():
                        break
                    chunk = prefix[first:first+128]
                    self._forward(chunk, prefill=True)
                    self.cache_ids.extend(chunk)
                    for at in range(0, len(chunk), 8):
                        n = min(8, len(chunk)-at)
                        cursor.command(0, n)
                        cursor.command(1, n)
                        cursor.command(2, n)
                    yield dict(type='prefill', processed_tokens=len(self.cache_ids),
                               prompt_tokens=len(ids), metrics=metrics())
                decode_start = time.monotonic()
                pending = ids[-1]
                while len(generated) < request.max_new and not cancel.is_set():
                    draft = proposer.propose(ids+generated,
                                             min(request.max_draft, request.max_new-len(generated)-1))
                    base = self.cache.get_seq_length()
                    if cursor.state.committed != base or base != len(self.cache_ids):
                        raise AssertionError('CPU cache and RTL ownership diverged')
                    cursor.command(0, 1+len(draft))
                    result = self._forward([pending]+draft)
                    emitted, accepted = greedy_accept(draft, result.logits[0].argmax(-1).tolist())
                    ended = False
                    for i, token in enumerate(emitted):
                        if token in self.eos:
                            emitted = emitted[:i+1]
                            ended = True
                            break
                    cursor.command(1, 1+len(draft))
                    cursor.command(2, len(emitted))
                    self.cache.crop(base+len(emitted))
                    generated.extend(emitted)
                    self.cache_ids = (ids+generated)[:-1]
                    pending = emitted[-1]
                    if self.cache.get_seq_length() != len(self.cache_ids):
                        raise AssertionError('Incorrect cache length after verification')
                    rounds += 1
                    proposed += len(draft)
                    accepted_total += min(accepted, len(emitted))
                    if first_token_ms is None:
                        first_token_ms = (time.monotonic()-started)*1000
                    text = self.tokenizer.decode(generated, skip_special_tokens=True,
                                                 clean_up_tokenization_spaces=False)
                    yield dict(type='update', text=text.rstrip('\ufffd'), metrics=metrics())
                    if ended:
                        reason = 'eos'
                        break
                if cancel.is_set():
                    reason = 'stopped'
        except Exception:
            # A failed forward can partially mutate layer caches. Never reuse it.
            self.clear_cache()
            raise
        yield dict(type='done', text=self.tokenizer.decode(generated, skip_special_tokens=True,
                                                         clean_up_tokenization_spaces=False),
                   finish_reason=reason, metrics=metrics())
