"""Chat input validation and checkpoint tokenization, without a tensor runtime."""
from dataclasses import dataclass
import json
from jinja2.sandbox import ImmutableSandboxedEnvironment
from tokenizers import Tokenizer

SYSTEM_PROMPT = ("You are Qwen3-0.6B, a small AI assistant running locally. "
                 "Answer the user's latest question directly. Use earlier messages as context, "
                 "not as a script to repeat. You may quote earlier replies when asked. If you do not know something, say so.")
TURN_REMINDER = ("Answer the next user message directly. You are Qwen3-0.6B, "
                 "a local AI assistant. Avoid repeating earlier assistant replies unless asked to quote them.")
CONTEXT_LIMIT = 4096

@dataclass
class ChatRequest:
    session: str
    ids: list[int]
    max_new: int
    max_draft: int
    workers: int | None = None
    decoding: str = 'greedy'
    seed: int | None = None


def common_prefix(a, b):
    n = 0
    for left, right in zip(a, b):
        if left != right:
            break
        n += 1
    return n


class ChatProtocol:
    def __init__(self, model):
        self.tokenizer = Tokenizer.from_file(str(model / 'tokenizer.json'))
        config = json.loads((model / 'tokenizer_config.json').read_text())
        self.template = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True).from_string(config['chat_template'])

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
        max_draft = payload.get('max_draft', 0)
        if type(max_new) is not int or not 1 <= max_new <= 512:
            raise ValueError('Reply limit must be between 1 and 512 tokens')
        if type(max_draft) is not int or not 0 <= max_draft <= 7:
            raise ValueError('Draft length must be between 0 and 7')
        conversation = [{'role': 'system', 'content': SYSTEM_PROMPT}]
        for i, message in enumerate(messages):
            # Keep the small model oriented on each new question. Retain these
            # reminders on later turns so cached prefixes remain byte-identical.
            if i > 0 and message['role'] == 'user':
                conversation.append({'role': 'system', 'content': TURN_REMINDER})
            conversation.append(message)
        text = self.template.render(messages=conversation,
                                    add_generation_prompt=True, enable_thinking=False)
        ids = self.tokenizer.encode(text, add_special_tokens=False).ids
        if len(ids) + max_new > CONTEXT_LIMIT:
            raise ValueError(f'Context is full: {len(ids)} prompt tokens plus {max_new} reserved '
                             f'for the reply exceeds {CONTEXT_LIMIT}. Start a new chat or shorten the reply.')
        decoding = payload.get('decoding', 'chat')
        if decoding not in ('chat', 'greedy'):
            raise ValueError('Choose chat or greedy decoding')
        seed = payload.get('seed')
        if seed is not None and (type(seed) is not int or not 0 <= seed <= 4294967295):
            raise ValueError('Seed must be an integer from 0 to 4294967295')
        return ChatRequest(session, ids, max_new, max_draft, decoding=decoding, seed=seed)

