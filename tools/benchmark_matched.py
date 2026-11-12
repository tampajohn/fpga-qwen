#!/usr/bin/env python3
"""Matched local CPU inference timings; no Torch, GPU, or remote model service.

Performance mode requires a fixed number of greedy tokens before EOS, so both
engines execute the same number of decoding steps without masking stop tokens.
Early EOS rejects the comparison. --quality evaluates ordinary completed replies.
"""
import argparse
import hashlib
import http.client
import json
import platform
from pathlib import Path
import socket
import statistics
import subprocess
import tempfile
import time

try:
    from .chat_protocol import ChatProtocol
    from .experiment import ROOT, locate_model
    from .rtl_model import RTLModel
except ImportError:
    from chat_protocol import ChatProtocol
    from experiment import ROOT, locate_model
    from rtl_model import RTLModel

EOS = {151643, 151645}


def assert_cpu_build():
    cache = (ROOT/'build/llama-cpu/CMakeCache.txt').read_text()
    for name in ('METAL', 'CUDA', 'VULKAN', 'OPENCL'):
        if f'GGML_{name}:BOOL=OFF' not in cache:
            raise RuntimeError(f'CPU-only native build required: GGML_{name}=OFF')


def timing_result(start, arrivals, ids, text, stopped, backend_timings=None):
    elapsed = time.perf_counter()-start
    # The first output includes prompt work. Subsequent token intervals measure decode.
    steady = arrivals[-1]-arrivals[0] if len(arrivals)>1 else None
    return dict(output_token_ids=ids, output_token_count=len(ids), text=text,
                finish_reason=stopped, first_token_seconds=arrivals[0]-start if arrivals else None,
                request_seconds=elapsed,
                steady_decode_seconds=steady,
                steady_decode_tokens_per_second=(len(arrivals)-1)/steady if steady and steady>0 else None,
                backend_timings=backend_timings)


class SVEngine:
    def __init__(self, checkpoint, threads, tokenizer, binary=None, lanes=1, block_size=1, prefill_size=None, precision_profile='fp64', verification_capacity=1):
        if precision_profile not in ('fp64', 'fp32_b1_fp64_prefill', 'fp32_b1_fp32_prefill', 'fp32_b1_fp64_prefill_fp32_kv'):
            raise ValueError('Unknown simulator precision profile')
        if precision_profile != 'fp64' and (not threads or lanes != 8 or block_size != 1 or prefill_size != 8):
            raise ValueError('Hybrid timing requires workers, eight lanes, B8 prefill and B1 decode')
        if verification_capacity not in (1,4):raise ValueError("Invalid verification capacity")
        self.verification_capacity = verification_capacity
        self.precision_profile = precision_profile
        kwargs = {'binary': binary} if binary is not None else {}
        self.model = RTLModel(checkpoint, workers=threads, block_size=block_size, prefill_size=prefill_size, **kwargs)
        self.lanes = lanes
        self.tokenizer = tokenizer
        self.metadata = dict(weight_storage='BF16 checkpoint', arithmetic='SystemVerilog real / FP64',
                             kv='SystemVerilog real / FP64', parallel_workers=threads, reduction_lanes=lanes,
                             prefill_block_size=self.model.prefill_size, verification_block_size=block_size,
                             wide_prefill_reduction_lanes=4 if self.model.prefill_size==8 and threads and lanes==8 else None,
                             executable=str(binary or ROOT/'build/full/Vqwen_behavioral'),
                             cpu_thread_note='Persistent projection workers plus coordinator; no GPU')
        self.metadata['precision_profile'] = precision_profile
        if precision_profile == 'fp32_b1_fp64_prefill_fp32_kv':
            self.metadata['kv'] = 'SystemVerilog shortreal / FP32 storage, exact FP64 widening for attention'
        if precision_profile != 'fp64':
            prefill_precision = 'FP32' if precision_profile == 'fp32_b1_fp32_prefill' else 'FP64'
            self.metadata['arithmetic'] = f'SV FP32 scalar projections; {prefill_precision} prefill projections; FP64 attention and other model arithmetic'
            self.metadata['routing'] = f'Fresh {prefill_precision} B8 prompt[:-1] with B4 tails, then FP32 B1 final prompt token and decode; no prefix reuse or suffix verification'

    def start(self):
        start = time.perf_counter()
        self.model.start()
        if self.precision_profile != 'fp64':
            capabilities = json.loads(self.model.command('precision'))
            expected = dict(precision_profile=self.precision_profile, scalar_projection='fp32',
                            prefill_projection='fp32' if self.precision_profile == 'fp32_b1_fp32_prefill' else 'fp64', attention_kv='fp64',
                            max_verification_block=self.verification_capacity, prefix_reuse_supported=False)
            if self.precision_profile == 'fp32_b1_fp64_prefill_fp32_kv':
                expected.pop('attention_kv')
                expected.update(attention_arithmetic='fp64',kv_storage='fp32',canonical_prefix_policy='none')
            if self.verification_capacity == 4:expected['verification_projection']='fp32'
            if self.precision_profile == 'fp32_b1_fp32_prefill':expected['canonical_prefix_policy']='none'
            if not isinstance(capabilities, dict) or any(
                type(capabilities.get(key)) is not type(value) or capabilities[key] != value
                for key, value in expected.items()
            ):
                raise RuntimeError('Simulator precision capabilities do not match the requested hybrid profile')
            self.metadata['verified_runtime_capabilities'] = capabilities
        self.model.set_lanes(self.lanes)
        return time.perf_counter()-start

    def close(self):
        self.model.close()

    def prefill(self, tokens):
        for offset in range(0, len(tokens), self.model.prefill_size):
            chunk = tokens[offset:offset+self.model.prefill_size]
            if self.model.prefill_size == 8:
                self.model.prefill(chunk)
            elif self.model.block_size == 4:
                self.model.block(chunk, logits=False)
            else:
                self.model.token(chunk[0], logits=False)

    def generate(self, prompt, count, fixed_count):
        self.model.reset()
        start = time.perf_counter()
        self.prefill(prompt[:-1])
        pending, output, arrivals = prompt[-1], [], []
        reason = 'length'
        for _ in range(count):
            token, _, _ = self.model.token(pending)
            arrivals.append(time.perf_counter())
            output.append(token)
            if token in EOS:
                reason = 'eos'
                break
            pending = token
        if fixed_count and (reason == 'eos' or len(output) != count):
            raise RuntimeError('SV reached EOS before fixed workload completed; shorten --tokens or use a longer-answer prompt')
        result = timing_result(start, arrivals, output, self.tokenizer.decode(output, skip_special_tokens=True), reason)
        result['eos_observed'] = any(token in EOS for token in output)
        return result


class NativeEngine:
    def __init__(self, precision, threads, directory):
        assert_cpu_build()
        self.precision, self.threads, self.directory = precision, threads, directory
        self.model = ROOT/f'build/bench/qwen3-0.6b-{precision}.gguf'
        if not self.model.is_file():
            raise FileNotFoundError(self.model)
        self.process = None
        self.log = None
        self.metadata = dict(weight_storage=precision, arithmetic='llama.cpp native kernels; mixed precision',
                             kv='FP16', cpu_threads=threads, cpu_thread_note='Same numeric worker count as SV')

    def start(self):
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', 0))
            self.port = sock.getsockname()[1]
        self.log = (self.directory/f'llama-{self.precision}.log').open('w')
        self.command = [str(ROOT/'build/llama-cpu/bin/llama-server'), '-m', str(self.model),
                       '--host', '127.0.0.1', '--port', str(self.port), '-t', str(self.threads),
                       '-tb', str(self.threads), '-c', '4096', '-np', '1', '-ngl', '0',
                       '--no-kv-offload', '--no-op-offload', '--no-webui', '--no-warmup',
                       '-ctk', 'f16', '-ctv', 'f16']
        start = time.perf_counter()
        self.process = subprocess.Popen(self.command, stdout=self.log, stderr=self.log)
        while time.perf_counter()-start<90:
            if self.process.poll() is not None:
                raise RuntimeError(f'llama-server failed; see {self.log.name}')
            conn = http.client.HTTPConnection('127.0.0.1', self.port, timeout=1)
            try:
                conn.request('GET', '/health')
                if conn.getresponse().status == 200:
                    return time.perf_counter()-start
            except OSError:
                pass
            finally:
                conn.close()
            time.sleep(.05)
        raise TimeoutError('Native startup timed out')

    def close(self):
        if self.process:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill(); self.process.wait()
        self.process = None
        if self.log:
            self.log.close()
        self.log = None

    def generate(self, prompt, count, fixed_count):
        conn = http.client.HTTPConnection('127.0.0.1', self.port, timeout=180)
        start = time.perf_counter()
        output, arrivals, text, reason, timings = [], [], '', 'incomplete', None
        try:
            conn.request('POST', '/completion', json.dumps(dict(
                prompt=prompt, n_predict=count, temperature=0, seed=42,
                repeat_penalty=1, presence_penalty=0, frequency_penalty=0,
                ignore_eos=False, cache_prompt=False, id_slot=0,
                stream=True, return_tokens=True, timings_per_token=True)),
                {'Content-Type':'application/json'})
            response = conn.getresponse()
            if response.status != 200:
                raise RuntimeError(f'Native HTTP {response.status}: {response.read().decode()}')
            for line in response:
                if not line.startswith(b'data: '):
                    continue
                event = json.loads(line[6:])
                if 'error' in event:
                    raise RuntimeError(str(event['error']))
                tokens = event.get('tokens', [])
                now = time.perf_counter()
                output.extend(tokens)
                arrivals.extend([now]*len(tokens))
                text += event.get('content', '')
                timings = event.get('timings', timings)
                if event.get('stop'):
                    reason = 'length' if event.get('stop_type') == 'limit' else 'eos'
                    break
            if reason == 'incomplete':
                raise RuntimeError('Incomplete native stream')
            result = timing_result(start, arrivals, output, text, reason, timings)
            result['eos_observed'] = reason == 'eos' or any(token in EOS for token in output)
            if fixed_count and (result['eos_observed'] or len(output) != count):
                raise RuntimeError(f'Native reached EOS or returned {len(output)} tokens for fixed {count} steps; shorten --tokens or use a longer-answer prompt')
            return result
        finally:
            conn.close()


def make_prompt(protocol, profile):
    filler = ('Memory bandwidth measures data transfer rate between a processor and memory. '
              'Caches retain recently accessed information and can reduce transfers. ')
    repetitions = {'short':0, 'medium':4, 'long':16}[profile]
    content = filler*repetitions + 'Explain memory bandwidth and caching in four concise sentences.'
    request = protocol.prepare(dict(session='benchmark', messages=[dict(role='user', content=content)],
                                    max_new_tokens=128, decoding='greedy'))
    return request.ids


def compare_sv_logits(checkpoint, protocol, reference_binary, candidate_binary, workers, lanes, count, atol, profile='short', reference_lanes=1, require_bit_identical=False, candidate_block_size=1, candidate_prefill_size=None):
    """Teacher-force both engines from the same prompt; test every output logit.

    Reassociation may change low bits. Exact sampled text is not required, but
    greedy IDs must agree and max absolute logit error must meet the declared
    tolerance. The candidate is fed the reference token so context cannot drift.
    """
    import numpy as np
    reference = SVEngine(checkpoint,workers,protocol.tokenizer,reference_binary,reference_lanes)
    candidate = SVEngine(checkpoint,workers,protocol.tokenizer,candidate_binary,lanes,candidate_block_size,candidate_prefill_size)
    prompt = make_prompt(protocol,profile)
    errors, choices, finite, bit_identical = [], [], True, True
    try:
        reference.start(); candidate.start()
        reference.prefill(prompt[:-1])
        candidate.prefill(prompt[:-1])
        pending=prompt[-1]
        with tempfile.TemporaryDirectory() as directory:
            ref_path, cand_path = Path(directory)/'reference.bin', Path(directory)/'candidate.bin'
            for _ in range(count):
                expected, _, _ = reference.model.token(pending)
                actual, _, _ = candidate.model.token(pending)
                reference.model.command(f'dump {ref_path}')
                candidate.model.command(f'dump {cand_path}')
                a=np.fromfile(ref_path,dtype=np.float64)
                b=np.fromfile(cand_path,dtype=np.float64)
                if len(a)!=151936 or len(b)!=151936:
                    raise RuntimeError('Unexpected full-vocabulary logit shape')
                bit_identical = bit_identical and bool(np.array_equal(a.view(np.uint64), b.view(np.uint64)))
                current_finite=bool(np.isfinite(a).all() and np.isfinite(b).all())
                finite=finite and current_finite
                errors.append(float(np.max(np.abs(a-b))) if current_finite else None)
                choices.append(dict(reference=expected,candidate=actual))
                pending=expected
        maximum=max(errors) if finite else None
        matched=all(choice['reference']==choice['candidate'] for choice in choices)
        result=dict(reference_binary=str(reference_binary),candidate_binary=str(candidate_binary),
            reference_sha256=hashlib.sha256(reference_binary.read_bytes()).hexdigest(),
            candidate_sha256=hashlib.sha256(candidate_binary.read_bytes()).hexdigest(),
            prompt_tokens=len(prompt),teacher_forced_generated_tokens=count,
            candidate_reduction_lanes=lanes,reference_reduction_lanes=reference_lanes,
            candidate_prefill_block_size=candidate.model.prefill_size,reference_prefill_block_size=1,
            bit_identical_logits=bit_identical,require_bit_identical=require_bit_identical,absolute_error_tolerance=atol,
            per_position_max_absolute_logit_errors=errors,max_absolute_logit_error=maximum,
            greedy_choices=choices,unchanged_greedy_sequence=matched,all_logits_finite=finite,
            passed=finite and matched and maximum<=atol and (bit_identical or not require_bit_identical))
        print(json.dumps(dict(numerical_gate_passed=result['passed'],max_absolute_logit_error=maximum,
                              unchanged_greedy_sequence=matched)),flush=True)
        return result
    finally:
        reference.close();candidate.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--engines', default='rtl,llama-f32,llama-f16,llama-q4_k_m')
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--tokens', type=int, default=64)
    parser.add_argument('--repetitions', type=int, default=3)
    parser.add_argument('--profiles', default='short,medium,long')
    parser.add_argument('--model')
    parser.add_argument('--sv-binary', type=Path, help='Trusted local simulator executable; never accepted through HTTP')
    parser.add_argument('--sv-precision-profile', choices=['fp64', 'fp32_b1_fp64_prefill', 'fp32_b1_fp32_prefill'], default='fp64',
                        help='Hybrid profile requires matching runtime precision capabilities')
    parser.add_argument('--sv-verification-capacity', type=int, choices=[1,4], default=1,
                        help='Expected simulator verification capability; timed decoding remains sequential')
    parser.add_argument('--sv-lanes', type=int, choices=[1,8], default=1)
    parser.add_argument('--sv-block-size', type=int, choices=[1,4], default=1,
                        help='Explicitly enable B4 prefill for a block-capable SV executable')
    parser.add_argument('--sv-prefill-size', type=int, choices=[1,4,8], default=None,
                        help='Prompt capacity; 8 requires a B8 executable and --sv-block-size 4')
    parser.add_argument('--reference-sv-binary', type=Path, help='Optional baseline for a separate numerical gate before timing')
    parser.add_argument('--reference-sv-lanes', type=int, choices=[1,8], default=1)
    parser.add_argument('--require-bit-identical', action='store_true', help='Require exact logit bit patterns as well as the numerical/greedy gate')
    parser.add_argument('--gate-tokens', type=int, default=16)
    parser.add_argument('--gate-profile', choices=['short','medium','long'], default='short')
    parser.add_argument('--logit-atol', type=float, default=1e-6)
    parser.add_argument('--quality', action='store_true')
    parser.add_argument('--quality-only', action='store_true')
    parser.add_argument('--quality-cases', default='', help='Comma-separated objective case names; empty means all')
    parser.add_argument('--output', type=Path, default=ROOT/'build/matched-benchmark.json')
    args = parser.parse_args()
    if args.require_bit_identical and not args.reference_sv_binary:
        parser.error('--require-bit-identical requires --reference-sv-binary')
    if not 1<=args.gate_tokens<=128 or not 0<args.logit_atol<1:
        parser.error('Invalid numerical gate limits')
    for name in ('sv_binary', 'reference_sv_binary'):
        value = getattr(args, name)
        if value is not None:
            value = value.expanduser().resolve()
            if not value.is_file():
                parser.error(f'Missing executable: {value}')
            setattr(args, name, value)
    if not 1<=args.threads<=32 or not 2<=args.tokens<=512 or not 1<=args.repetitions<=20:
        parser.error('Invalid threads, tokens or repetitions')
    profiles = [] if args.quality_only else args.profiles.split(',')
    if any(p not in ('short','medium','long') for p in profiles):
        parser.error('Unknown context profile')
    checkpoint = locate_model(args.model)
    protocol = ChatProtocol(checkpoint)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = dict(cpu=(subprocess.check_output(['sysctl','-n','machdep.cpu.brand_string'],text=True).strip()
                  if platform.system()=='Darwin' else platform.processor() or platform.machine()), threads_or_workers=args.threads,
        checkpoint=str(checkpoint), timestamp_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        decoding='greedy, no penalties, identical EOS-respecting selection; performance requires requested count before EOS',
        gpu=False, pytorch=False, runs=[], quality=[], sv_lanes=args.sv_lanes, sv_block_size=args.sv_block_size, sv_prefill_size=args.sv_prefill_size or args.sv_block_size,
        limitations=['Same prompt token IDs, context, generated count and numeric worker count; NOT identical arithmetic or KV precision.',
                     'SV: BF16 stored weights / declared per-engine arithmetic profile / FP64 KV. Native: named stored weights / mixed arithmetic / FP16 KV.',
                     'Startup is process/model startup, not OS page-cache cold; no page-cache eviction is attempted.',
                     'Every timed request has empty logical KV. Warm repetitions retain the process and resident weight pages.',
                     'Both measurements include their existing transport (SV pipes, llama HTTP); steady decode excludes first-token prompt processing.',
                     'Fixed-count performance rejects any early EOS rather than masking it. Quality mode permits EOS and records objective checks separately.'])
    sv_binary = args.sv_binary or ROOT/'build/full/Vqwen_behavioral'
    generated_dirs = [sv_binary.parent, sv_binary.parent.parent/'projection']
    generated = sorted(path for directory in generated_dirs for suffix in ('*.cpp','*.h') for path in directory.glob(suffix))
    report['generated_files'] = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in generated}
    report['generated_source_sha256'] = hashlib.sha256(json.dumps(report['generated_files'],sort_keys=True).encode()).hexdigest()
    for key, binary in [('sv_binary_sha256', sv_binary), ('native_binary_sha256', ROOT/'build/llama-cpu/bin/llama-server')]:
        report[key] = hashlib.sha256(binary.read_bytes()).hexdigest() if binary.is_file() else None
    if args.reference_sv_binary:
        report['numerical_gate'] = compare_sv_logits(checkpoint, protocol, args.reference_sv_binary,
            sv_binary, args.threads, args.sv_lanes, args.gate_tokens, args.logit_atol, args.gate_profile, args.reference_sv_lanes, args.require_bit_identical, args.sv_block_size, args.sv_prefill_size)
        args.output.write_text(json.dumps(report,indent=2)+'\n')
        if not report['numerical_gate']['passed']:
            raise RuntimeError('Numerical gate failed; candidate timing skipped')
    try:
        report['llama_commit'] = (subprocess.check_output(['git','-C',str(ROOT/'build/llama.cpp'),'rev-parse','HEAD'],text=True).strip()
                                  if (ROOT/'build/llama.cpp/.git').exists() else None)
        for name in args.engines.split(','):
            engine = SVEngine(checkpoint,args.threads,protocol.tokenizer,args.sv_binary,args.sv_lanes,args.sv_block_size,args.sv_prefill_size,args.sv_precision_profile,args.sv_verification_capacity) if name=='rtl' else NativeEngine(name.removeprefix('llama-'),args.threads,args.output.parent)
            try:
                startup = engine.start()
                for profile in profiles:
                    ids = make_prompt(protocol,profile)
                    first = engine.generate(ids,args.tokens,True)
                    warm = [engine.generate(ids,args.tokens,True) for _ in range(args.repetitions)]
                    record = dict(engine=name, metadata=engine.metadata, startup_seconds=startup,
                        profile=profile,prompt_tokens=len(ids),prompt_ids_sha256=hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
                        first_request=first,warm_requests=warm,
                        warm_median_ttft_seconds=statistics.median(x['first_token_seconds'] for x in warm),
                        warm_median_decode_tokens_per_second=statistics.median(x['steady_decode_tokens_per_second'] for x in warm))
                    requests = [first]+warm
                    record['greedy_sequence_repeatable'] = all(x['output_token_ids']==first['output_token_ids'] for x in requests)
                    record['warm_decode_rate_min'] = min(x['steady_decode_tokens_per_second'] for x in warm)
                    record['warm_decode_rate_max'] = max(x['steady_decode_tokens_per_second'] for x in warm)
                    report['runs'].append(record)
                    print(json.dumps({k:record[k] for k in ('engine','profile','prompt_tokens','warm_median_ttft_seconds','warm_median_decode_tokens_per_second')}),flush=True)
                if args.quality or args.quality_only:
                    try:
                        from .conversation_eval import CASES, evaluate_response, suite_fingerprint
                    except ImportError:
                        from conversation_eval import CASES, evaluate_response, suite_fingerprint
                    unknown = set(args.quality_cases.split(','))-{case.name for case in CASES} if args.quality_cases else set()
                    if unknown:
                        raise ValueError(f'Unknown quality cases: {sorted(unknown)}')
                    selected = [case for case in CASES if not args.quality_cases or case.name in args.quality_cases.split(',')]
                    if not selected:
                        raise ValueError('No quality cases selected')
                    report['quality_suite_sha256'] = suite_fingerprint(selected)
                    for case in selected:
                        request=protocol.prepare(dict(session=case.name,messages=list(case.messages),max_new_tokens=case.max_new_tokens,decoding='greedy'))
                        result=engine.generate(request.ids,case.max_new_tokens,False)
                        grade = evaluate_response(case,result['text'])
                        report['quality'].append(dict(engine=name,case=case.name,result=result,grade=grade))
                        print(json.dumps(dict(engine=name,case=case.name,passed=grade['passed'],text=result['text'])),flush=True)
                args.output.write_text(json.dumps(report,indent=2)+'\n')
            finally:
                engine.close()
    finally:
        args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(f'Saved {args.output}',flush=True)


if __name__=='__main__':
    main()
