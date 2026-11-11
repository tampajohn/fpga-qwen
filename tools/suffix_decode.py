#!/usr/bin/env python3
"""Full-model CPU greedy verification with RTL KV metadata, not FPGA inference."""
import argparse
import json
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from experiment import ROOT, locate_model
from kv import RTLCursor
from suffix import SuffixDrafter, greedy_accept


def commit_prefill(cursor, length):
    for first in range(0, length, 8):
        n = min(8, length-first)
        cursor.command(0, n)
        cursor.command(1, n)
        cursor.command(2, n)


@torch.inference_mode()
def speculative(model, ids, max_new, max_draft, eos):
    if len(ids) < 2 or len(ids)+max_new > 4096:
        raise ValueError('Need at least 2 prompt tokens and at most 4096 total tokens')
    proposer = SuffixDrafter()
    generated, rounds = [], []
    # Keep one pending token outside the cache. A verification call processes
    # [pending, draft...] so all logits align without an extra correction pass.
    cache = model(input_ids=torch.tensor([ids[:-1]]), use_cache=True).past_key_values
    pending = ids[-1]
    start = time.monotonic()
    with RTLCursor() as cursor:
        commit_prefill(cursor, len(ids)-1)
        while len(generated) < max_new:
            budget = max_new-len(generated)
            draft = proposer.propose(ids+generated, min(max_draft, budget-1))
            base = cache.get_seq_length()
            if cursor.state.committed != base:
                raise AssertionError('CPU and RTL cache lengths diverged before verification')
            cursor.command(0, 1+len(draft))
            result = model(input_ids=torch.tensor([[pending]+draft]), past_key_values=cache, use_cache=True)
            cache = result.past_key_values
            predictions = result.logits[0].argmax(dim=-1).tolist()
            emitted, accepted = greedy_accept(draft, predictions)
            ended = False
            for i, token in enumerate(emitted):
                if token in eos:
                    emitted = emitted[:i+1]
                    ended = True
                    break
            # Leave the last emitted token pending, even when it is EOS. This
            # preserves the invariant cached length = context length - 1.
            keep = len(emitted)
            cursor.command(1, 1+len(draft))
            cursor.command(2, keep)
            cache.crop(base+keep)
            generated.extend(emitted)
            pending = emitted[-1]
            if cache.get_seq_length() != len(ids)+len(generated)-1:
                raise AssertionError('Incorrect speculative cache crop')
            if cursor.state.committed != cache.get_seq_length():
                raise AssertionError('RTL accepted-prefix commit mismatch')
            rounds.append(dict(drafted=len(draft), accepted=accepted,
                               emitted=len(emitted), verification_positions=1+len(draft),
                               cache_length=cache.get_seq_length()))
            if ended:
                break
    elapsed = time.monotonic()-start
    return generated, rounds, elapsed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model')
    parser.add_argument('--max-new-tokens', type=int, default=48)
    parser.add_argument('--max-draft', type=int, default=7)
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.max_draft <= 7 or not 1 <= args.max_new_tokens <= 256 or args.threads < 1:
        raise ValueError('Use draft 1..7, output length 1..256, and positive threads')
    torch.set_num_threads(args.threads)
    path = locate_model(args.model)
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    print('Loading cached full Qwen3-0.6B on CPU (FP32 reference)...', flush=True)
    model = AutoModelForCausalLM.from_pretrained(path, local_files_only=True, dtype=torch.float32,
                                               attn_implementation='eager').eval()
    eos = model.generation_config.eos_token_id
    eos = set(eos if isinstance(eos, list) else [eos])
    workloads = {
        'repeated_text': 'Output the phrase "red blue green" exactly 20 times, separated by spaces. '
                         'Use no introduction or explanation.',
        'code_copy': 'Repeat this code exactly three times without markdown or explanation:\n'
                     'def add(a, b):\n    return a + b\n',
        'open_prose': 'In three short sentences, explain why the sky appears blue. '
                      'Do not use a list or an introduction.',
    }
    summaries = []
    for name, prompt in workloads.items():
        text = tokenizer.apply_chat_template([{'role': 'user', 'content': prompt}],
                                             tokenize=False, add_generation_prompt=True,
                                             enable_thinking=False)
        ids = tokenizer.encode(text, add_special_tokens=False)
        print(f'{name}: checking greedy baseline against suffix verification...', flush=True)
        with torch.inference_mode():
            baseline = model.generate(input_ids=torch.tensor([ids]),
                                      attention_mask=torch.ones((1, len(ids)), dtype=torch.long),
                                      max_new_tokens=args.max_new_tokens, do_sample=False,
                                      temperature=1.0, top_p=1.0, top_k=50,
                                      pad_token_id=tokenizer.eos_token_id)[0, len(ids):].tolist()
        actual, rounds, elapsed = speculative(model, ids, args.max_new_tokens, args.max_draft, eos)
        if actual != baseline:
            raise AssertionError(f'{name}: speculative output differs from greedy baseline')
        drafted = sum(r['drafted'] for r in rounds)
        accepted = sum(r['accepted'] for r in rounds)
        summary = dict(workload=name, prompt_tokens=len(ids), output_tokens=len(actual),
                       rounds=len(rounds), proposed_tokens=drafted, accepted_tokens=accepted,
                       acceptance_fraction=accepted/drafted if drafted else None,
                       tokens_per_verification=len(actual)/len(rounds),
                       verification_positions=sum(r['verification_positions'] for r in rounds),
                       cpu_experiment_seconds=elapsed, exact_greedy_match=True,
                       text=tokenizer.decode(actual, skip_special_tokens=True), trace=rounds)
        summaries.append(summary)
        print(f"  PASS: {len(actual)} identical tokens; {len(rounds)} verification rounds; "
              f"{len(actual)/len(rounds):.2f} tokens/round; {accepted}/{drafted} proposals accepted", flush=True)
    report = dict(model='Qwen/Qwen3-0.6B', precision='FP32 CPU full model; not W4A8 RTL',
                  drafting='longest-match n-gram lookup in current prompt and generated text; '
                  'no neural drafter; no future target output in lookup',
                  rtl='KV metadata cursor only; model arithmetic uses PyTorch CPU',
                  max_draft=args.max_draft, max_new_tokens=args.max_new_tokens,
                  limitations=['Tokens/round is NOT wall-clock speedup or FPGA throughput',
                               'Acceptance on FP32 does not establish acceptance on quantized hardware',
                               'Three short prompts are smoke tests, not a representative benchmark',
                               'Only greedy decoding is implemented; no speculative sampling'],
                  workloads=summaries)
    destination = ROOT / 'build/suffix_report.json'
    destination.write_text(json.dumps(report, indent=2)+'\n')
    print(f'Wrote {destination}', flush=True)


if __name__ == '__main__':
    main()
