"""CPU-only Phase0 discovery/confirmation using the existing native protocol.

Run directly under this serving checkout's tools/benchmark_guard.py. Discovery
is explicitly exploratory (1..3 rounds). Confirmation requires an explicit
shortlist and at least10 alternating/rotated paired rounds. One native server
is live at a time; process startup and workload warmup are outside scored time.

--config-file accepts a JSON list of objects with name, build_dir,
decode_threads and prefill_threads, optionally priority(0/1, default0) and
poll(0..100, default50), both applied to decode and prefill. Without a file, --build-dir plus
--decode-threads/--prefill-threads forms a discovery cross product; --configs
'DECODE:PREFILL,...' supplies an explicit shortlist within one build.
"""
import argparse
import hashlib
import http.client
import itertools
import json
import math
import os
from pathlib import Path
import platform
import re
import socket
import statistics
import subprocess
import time

from benchmark_guard import BenchmarkGate
from benchmark_matched import NativeEngine, make_prompt
from chat_protocol import ChatProtocol
from experiment import ROOT, locate_model

GPU_BACKENDS = ('METAL', 'CUDA', 'VULKAN', 'OPENCL', 'HIP', 'MUSA', 'SYCL')
OTHER_ACCELERATORS = ('WEBGPU', 'VIRTGPU', 'HEXAGON', 'CANN', 'OPENVINO', 'RPC')
METRICS = ('first_token_seconds', 'steady_decode_tokens_per_second', 'request_seconds')
EXPECTED_PROMPT_LENGTHS = {'short': 85, 'long': 452}


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def guard(directory):
    directory = Path(directory).resolve()
    status = BenchmarkGate(directory).status()
    metadata = json.loads((directory / 'run.json').read_text())
    if not status['running'] or metadata.get('pid') != os.getppid():
        raise RuntimeError('Run as a direct child of the serving checkout benchmark_guard.py')
    return dict(directory=str(directory), metadata=metadata)


def parse_cache(path):
    result = {}
    for line in path.read_text().splitlines():
        if not line or line.startswith(('#', '//')) or '=' not in line or ':' not in line.split('=', 1)[0]:
            continue
        name_type, value = line.split('=', 1)
        name, kind = name_type.split(':', 1)
        result[name] = dict(type=kind, value=value)
    return result


def freeze_build(directory):
    directory = Path(directory).resolve()
    cache_path = directory / 'CMakeCache.txt'
    server = directory / 'bin/llama-server'
    cache = parse_cache(cache_path)
    for name in GPU_BACKENDS:
        if cache.get('GGML_' + name) != dict(type='BOOL', value='OFF'):
            raise RuntimeError(f'CPU-only build required: GGML_{name}=OFF in {directory}')
    for name in OTHER_ACCELERATORS:
        if 'GGML_' + name in cache and cache['GGML_' + name] != dict(type='BOOL', value='OFF'):
            raise RuntimeError(f'Additional accelerator/remote backend enabled: {name}')
    if cache.get('GGML_CPU') != dict(type='BOOL', value='ON'):
        raise RuntimeError('Native CPU backend must be enabled')
    if cache.get('GGML_BACKEND_DL') != dict(type='BOOL', value='OFF'):
        raise RuntimeError('Dynamic backend discovery must be disabled for this CPU-only baseline')
    artifacts = {str(path.relative_to(directory)): sha(path) for path in (directory / 'bin').iterdir()
                 if path.is_file() and (path.name in ('llama-server', 'llama-bench') or '.dylib' in path.name)}
    if 'bin/llama-server' not in artifacts:
        raise RuntimeError(f'No native server in {directory}')
    manifest_path = directory / 'manifest.json'
    manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else None
    if manifest is not None:
        if manifest.get('completed') is not True or manifest.get('cache_sha256') != sha(cache_path):
            raise RuntimeError('Incomplete or mismatched native build manifest')
        for name, digest in manifest.get('artifacts', {}).items():
            if artifacts.get(name) != digest:
                raise RuntimeError(f'Build manifest artifact mismatch: {name}')
        if manifest.get('artifacts', {}).get('bin/llama-server') != artifacts['bin/llama-server']:
            raise RuntimeError('Build manifest does not pin the executed server')
    source = Path(cache['CMAKE_HOME_DIRECTORY']['value']).resolve()
    source_commit = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
    source_status = subprocess.check_output(['git', '-C', str(source), 'status', '--porcelain'], text=True)
    if source_status:
        raise RuntimeError('llama source must be clean for source-provenance recording')
    if manifest is not None and manifest.get('source_commit') != source_commit:
        raise RuntimeError('Source commit differs from frozen build manifest')
    linked = subprocess.check_output(['otool', '-L', str(server)], text=True)
    return dict(directory=str(directory), server=str(server), cache_sha256=sha(cache_path),
                artifacts=artifacts, source_directory=str(source), source_commit=source_commit,
                source_clean=True, manifest=manifest,
                manifest_sha256=sha(manifest_path) if manifest else None,
                provenance_note='Completed build manifest verified' if manifest else
                    'Historical build has no manifest: current clean source commit is recorded, but binary-to-source linkage is not independently established.',
                build_options={k: v for k, v in cache.items() if k.startswith(('GGML_', 'CMAKE_BUILD_TYPE', 'CMAKE_CXX_COMPILER'))},
                linked_libraries=linked)


def verify_build(record):
    directory = Path(record['directory'])
    if sha(directory / 'CMakeCache.txt') != record['cache_sha256']:
        raise RuntimeError('Native CMake cache changed during evaluation')
    for name, digest in record['artifacts'].items():
        if sha(directory / name) != digest:
            raise RuntimeError(f'Native binary/shared library changed: {name}')
    if record['manifest_sha256'] and sha(directory / 'manifest.json') != record['manifest_sha256']:
        raise RuntimeError('Native build manifest changed during evaluation')
    source = record['source_directory']
    if subprocess.check_output(['git', '-C', source, 'rev-parse', 'HEAD'], text=True).strip() != record['source_commit']:
        raise RuntimeError('Native source commit changed during evaluation')
    if subprocess.check_output(['git', '-C', source, 'status', '--porcelain'], text=True):
        raise RuntimeError('Native source became dirty during evaluation')


def host_snapshot():
    command = ['ps', '-Ao', 'pid,ppid,pcpu,pmem,etime,comm']
    raw = subprocess.check_output(command, text=True).splitlines()
    rows = [line for line in raw[1:] if len(line.split(None, 5)) == 6]
    rows.sort(key=lambda line: float(line.split(None, 5)[2]), reverse=True)
    return dict(timestamp_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                load_average=os.getloadavg(), ps_command=command, busiest_processes=[raw[0], *rows[:20]])


class Phase0NativeEngine(NativeEngine):
    """Reuse NativeEngine.generate/close; only explicit paths/startup differ."""
    def __init__(self, precision, config, build, model, directory):
        # Parent __init__/start hardcode one historical build directory. Keep
        # the inherited HTTP request, EOS, timing and teardown implementation.
        self.precision = precision
        self.threads = config['decode_threads']
        self.batch_threads = config['prefill_threads']
        self.priority = config.get('priority', 0)
        self.poll = config.get('poll', 50)
        if type(self.priority) is not int or self.priority not in (0, 1) or type(self.poll) is not int or not 0 <= self.poll <= 100:
            raise ValueError('Native priority must be0/1 and polling level0..100')
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.model = model
        self.server = Path(build['server'])
        self.process = self.log = None
        self.metadata = dict(weight_storage=precision, arithmetic='llama.cpp native mixed-precision kernels',
                             kv='FP16', cpu_threads=self.threads, prefill_threads=self.batch_threads,
                             requested_priority=self.priority, requested_poll=self.poll,
                             scheduling_note='Same requested priority/poll for decode and prefill; no hard pinning or independent OS scheduling verification.',
                             priority_failure_lines=[],
                             cpu_thread_note='Requested native thread counts, without affinity guarantees; CPU BLAS may use its own threads.')

    def start(self):
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', 0))
            self.port = sock.getsockname()[1]
        self.log_path = self.directory / 'server.log'
        self.log = self.log_path.open('w')
        self.command = [str(self.server), '-m', str(self.model), '--host', '127.0.0.1', '--port', str(self.port),
                        '-t', str(self.threads), '-tb', str(self.batch_threads),
                        '--prio', str(self.priority), '--prio-batch', str(self.priority),
                        '--poll', str(self.poll), '--poll-batch', str(self.poll), '-c', '4096', '-np', '1',
                        '-ngl', '0', '--no-kv-offload', '--no-op-offload', '--no-webui', '--no-warmup',
                        '-ctk', 'f16', '-ctv', 'f16']
        if self.command[self.command.index('-ngl') + 1] != '0' or any(
                flag not in self.command for flag in ('--no-kv-offload', '--no-op-offload')):
            raise RuntimeError('Runtime GPU offload flags missing')
        environment = os.environ.copy()
        self.removed_environment_keys = sorted(k for k in environment if k.startswith(('LLAMA_ARG_', 'GGML_KLEIDIAI_')) or k in
            ('DYLD_LIBRARY_PATH', 'DYLD_FALLBACK_LIBRARY_PATH', 'DYLD_INSERT_LIBRARIES',
             'GGML_BACKEND_PATH', 'GGML_TOTAL_THREADS'))
        for key in self.removed_environment_keys:
            environment.pop(key)
        started = time.perf_counter()
        self.process = subprocess.Popen(self.command, stdout=self.log, stderr=self.log, env=environment)
        while time.perf_counter() - started < 90:
            if self.process.poll() is not None:
                raise RuntimeError(f'Native server failed; see {self.log_path}')
            connection = http.client.HTTPConnection('127.0.0.1', self.port, timeout=1)
            try:
                connection.request('GET', '/health')
                response = connection.getresponse()
                response.read()
                if response.status == 200:
                    self.check_requested_priority()
                    return time.perf_counter() - started
            except OSError:
                pass
            finally:
                connection.close()
            time.sleep(.05)
        raise TimeoutError('Native server startup exceeded90seconds')

    def check_requested_priority(self):
        if self.priority != 1:
            return
        lines = self.log_path.read_text(errors='replace').splitlines()
        failures = [line for line in lines if re.search(r'failed to set (?:process|thread) priority', line, re.IGNORECASE)]
        self.metadata['priority_failure_lines'] = failures
        self.metadata['priority_runtime_check'] = ('application failure logged' if failures else
            'No application failure logged; actual OS scheduling/placement is not independently verified')
        if failures:
            raise RuntimeError('Requested priority1 was not fully applied: ' + ' | '.join(failures))

    def generate(self, *args, **kwargs):
        result = super().generate(*args, **kwargs)
        # Worker failures can be logged after health readiness. Check after all
        # preflight/warm/scored requests, outside the inherited timing interval.
        self.check_requested_priority()
        return result


def parse_configs(args, parser):
    if args.config_file:
        if args.configs or args.decode_threads or args.prefill_threads:
            parser.error('--config-file cannot be combined with thread-list options')
        configs = json.loads(args.config_file.read_text())
        if not isinstance(configs, list) or not configs:
            parser.error('--config-file must contain a nonempty JSON list')
    else:
        if not args.build_dir:
            parser.error('--build-dir is required without --config-file')
        if args.configs:
            if args.decode_threads or args.prefill_threads:
                parser.error('--configs cannot be combined with thread lists')
            try:
                pairs = [tuple(map(int, value.split(':'))) for value in args.configs.split(',')]
                if any(len(pair) != 2 for pair in pairs):
                    raise ValueError('Wrong pair length')
            except ValueError:
                parser.error('--configs requires DECODE:PREFILL,...')
        else:
            try:
                decode = [int(x) for x in (args.decode_threads or '6,8,12,18').split(',')]
                prefill = [int(x) for x in (args.prefill_threads or '6,12,18').split(',')]
                pairs = list(itertools.product(decode, prefill))
            except ValueError:
                parser.error('Thread lists must contain integers')
        configs = [dict(name=f'd{decode}-p{prefill}', build_dir=str(args.build_dir),
                        decode_threads=decode, prefill_threads=prefill) for decode, prefill in pairs]
    normalized = []
    for index, config in enumerate(configs):
        if not isinstance(config, dict) or set(config) - {'name', 'build_dir', 'decode_threads', 'prefill_threads', 'priority', 'poll'}:
            parser.error('Unknown configuration fields')
        if any(type(config.get(k)) is not int or not 1 <= config[k] <= 32 for k in ('decode_threads', 'prefill_threads')):
            parser.error('Decode/prefill thread counts must be integers1..32')
        priority, poll = config.get('priority', 0), config.get('poll', 50)
        if type(priority) is not int or priority not in (0, 1) or type(poll) is not int or not 0 <= poll <= 100:
            parser.error('Optional priority must be0/1 and poll must be0..100')
        build_dir = config.get('build_dir') or args.build_dir
        if not build_dir:
            parser.error('Each configuration requires build_dir')
        name = config.get('name', f'config-{index}')
        if not isinstance(name, str) or not re.fullmatch(r'[A-Za-z0-9_.-]{1,80}', name):
            parser.error('Configuration names must be1..80 letters/digits/underscore/dash/dot')
        normalized.append(dict(name=name, build_dir=str(Path(build_dir).expanduser().resolve()),
                               decode_threads=config['decode_threads'], prefill_threads=config['prefill_threads'],
                               priority=priority, poll=poll))
    if len({c['name'] for c in normalized}) != len(normalized):
        parser.error('Configuration names must be unique')
    signatures = {(c['build_dir'], c['decode_threads'], c['prefill_threads'], c['priority'], c['poll']) for c in normalized}
    if len(signatures) != len(normalized):
        parser.error('Duplicate build/thread/priority/poll configurations are not independent comparisons')
    if args.mode == 'confirmation' and (not (args.config_file or args.configs) or len(normalized) < 2):
        parser.error('Confirmation requires an explicit shortlist with at least2configurations')
    return normalized


def validate_result(result, prompt, count, expected=None):
    timings = result.get('backend_timings')
    if (result.get('output_token_count') != count or result.get('eos_observed')
            or result.get('finish_reason') != 'length' or len(result.get('output_token_ids', [])) != count):
        raise RuntimeError('EOS/output length mismatch; never suppress EOS to retain a score')
    if not isinstance(timings, dict) or timings.get('cache_n') != 0 or timings.get('prompt_n') != len(prompt):
        raise RuntimeError('Native prompt cache or evaluated prompt count differs from fixed workload')
    if expected is not None and result['output_token_ids'] != expected:
        raise RuntimeError('Same-precision output IDs changed across configurations/repetitions')
    if any(not isinstance(result.get(k), (int, float)) or not math.isfinite(result[k]) or result[k] <= 0 for k in METRICS):
        raise RuntimeError('Invalid native elapsed-time measurement')


def distribution(values):
    median = statistics.median(values)
    return dict(count=len(values), median=median, mean=statistics.mean(values),
                minimum=min(values), maximum=max(values),
                standard_deviation=statistics.stdev(values) if len(values) > 1 else 0,
                median_absolute_deviation=statistics.median(abs(value - median) for value in values))


def summarize(report):
    names = [c['name'] for c in report['configurations']]
    summaries = {}
    comparisons = {}
    for profile in report['prompts']:
        summaries[profile] = {}
        comparisons[profile] = {}
        for name in names:
            rows = [visit['profiles'][profile]['scored'] for visit in report['visits'] if visit['config'] == name]
            summaries[profile][name] = {metric: distribution([r[metric] for r in rows]) for metric in METRICS}
        for name in names[1:]:
            pairs = []
            for round_index in range(report['rounds']):
                visits = {v['config']: v for v in report['visits'] if v['round'] == round_index}
                reference = visits[names[0]]['profiles'][profile]['scored']
                candidate = visits[name]['profiles'][profile]['scored']
                pairs.append(dict(round=round_index,
                    reference_before_candidate=visits[names[0]]['order_index'] < visits[name]['order_index'],
                    decode_speedup=candidate['steady_decode_tokens_per_second'] / reference['steady_decode_tokens_per_second'],
                    ttft_speedup=reference['first_token_seconds'] / candidate['first_token_seconds'],
                    request_speedup=reference['request_seconds'] / candidate['request_seconds']))
            comparisons[profile][name] = dict(reference=names[0], pairs=pairs,
                ratios={metric: distribution([p[metric] for p in pairs]) for metric in
                        ('decode_speedup', 'ttft_speedup', 'request_speedup')})
    report['summaries'] = summaries
    report['paired_comparisons'] = comparisons
    report['best_observed_decode'] = {profile: max(names, key=lambda name: summaries[profile][name]['steady_decode_tokens_per_second']['median'])
                                      for profile in summaries}
    report['best_observed_ttft'] = {profile: min(names, key=lambda name: summaries[profile][name]['first_token_seconds']['median'])
                                    for profile in summaries}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('discovery', 'confirmation'), required=True)
    parser.add_argument('--build-dir', type=Path)
    parser.add_argument('--config-file', type=Path)
    parser.add_argument('--configs', help='Explicit DECODE:PREFILL,... shortlist within --build-dir')
    parser.add_argument('--decode-threads')
    parser.add_argument('--prefill-threads')
    parser.add_argument('--precision', choices=('f16', 'f32', 'q4_k_m'), required=True)
    parser.add_argument('--models-dir', type=Path, default=ROOT / 'build/bench')
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--profiles', default='short,long')
    parser.add_argument('--rounds', '--pairs', dest='rounds', type=int)
    parser.add_argument('--tokens', type=int, default=32)
    parser.add_argument('--warmups', type=int, default=1)
    parser.add_argument('--guard-state-dir', type=Path, default=ROOT / 'build/benchmark-status')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.rounds = args.rounds if args.rounds is not None else 2 if args.mode == 'discovery' else 10
    if (args.mode == 'discovery' and not 1 <= args.rounds <= 3) or (args.mode == 'confirmation' and not 10 <= args.rounds <= 100):
        parser.error('Discovery uses1..3rounds; confirmation requires10..100paired rounds')
    profiles = args.profiles.split(',')
    if not profiles or len(set(profiles)) != len(profiles) or any(p not in EXPECTED_PROMPT_LENGTHS for p in profiles):
        parser.error('Profiles must be short,long without duplicates')
    if args.mode == 'confirmation' and set(profiles) != set(EXPECTED_PROMPT_LENGTHS):
        parser.error('Confirmation requires both85-token and452-token prompts')
    if not 2 <= args.tokens <= 128 or not 1 <= args.warmups <= 3:
        parser.error('Use2..128output tokens and1..3warmups')
    configs = parse_configs(args, parser)
    guard_before = guard(args.guard_state_dir)
    if args.output.exists():
        parser.error('Output already exists; retain prior evidence and choose a fresh report path')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    log_root = args.output.parent / (args.output.stem + '-logs')
    if log_root.exists() and any(log_root.iterdir()):
        parser.error('Log directory already contains evidence; choose a fresh output path')
    builds = {path: freeze_build(path) for path in dict.fromkeys(c['build_dir'] for c in configs)}
    model = (args.models_dir / f'qwen3-0.6b-{args.precision}.gguf').resolve()
    checkpoint = locate_model(args.checkpoint)
    protocol = ChatProtocol(checkpoint)
    prompts = {profile: make_prompt(protocol, profile) for profile in profiles}
    if any(len(ids) != EXPECTED_PROMPT_LENGTHS[profile] for profile, ids in prompts.items()):
        raise RuntimeError('Historical matched prompt template/token IDs changed in length')
    source_paths = [Path(__file__), ROOT / 'tools/benchmark_matched.py', ROOT / 'tools/chat_protocol.py',
                    ROOT / 'tools/benchmark_guard.py', ROOT / 'tools/experiment.py']
    tokenization_paths = [checkpoint / name for name in ('tokenizer.json', 'tokenizer_config.json', 'config.json')]
    report = dict(completed=False, mode=args.mode, performance_claim_eligible=False,
                  paired_sampling_complete=False, numerical_quality_review_required=True,
                  interpretation='Exploratory configuration selection only; no performance claim.' if args.mode == 'discovery' else
                                 'At least10paired observations per configuration comparison; observed winners still require uncertainty and quality review.',
                  precision=args.precision, gpu=False, pytorch=False, rounds=args.rounds,
                  tokens=args.tokens, warmups_per_visit_profile=args.warmups,
                  configurations=configs, builds=builds, model=dict(path=str(model), sha256=sha(model)),
                  tokenization={str(p): sha(p) for p in tokenization_paths},
                  source_hashes={str(p): sha(p) for p in source_paths},
                  prompts={name: dict(ids=ids, count=len(ids)) for name, ids in prompts.items()},
                  guard_before=guard_before, host_before=host_snapshot(), platform=platform.platform(),
                  library_thread_environment={k: os.environ.get(k) for k in
                      ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS')},
                  preflight=[], visits=[],
                  limitations=['One server active per visit; startup and per-profile warmup are excluded from scores. Processes restart between visits; address placement is uncontrolled.',
                               'Browser guard excludes browser chat, not unrelated OS applications. No CPU affinity, OS cache eviction, thermal control or OS-wide isolation is claimed.',
                               'Identical prompt/output IDs are required within this precision. Different precision classes must be evaluated in separate runs and are not comparable speedup claims.',
                               'Inherited NativeEngine HTTP/SSE timings include transport. FP16 KV, no prompt reuse, no suffix/speculative drafting, EOS respected.',
                               'This is native baseline tuning, not broad model-quality or full-logit validation of a new arithmetic engine.'])
    expected_ids = {}
    def save():
        args.output.write_text(json.dumps(report, indent=2) + '\n')
    def visit(config, phase, round_index=None, order_index=None):
        guard(args.guard_state_dir)
        suffix = 'preflight' if phase == 'preflight' else f'round-{round_index:03d}'
        engine = Phase0NativeEngine(args.precision, config, builds[config['build_dir']], model,
                                    log_root / suffix / config['name'])
        record = dict(config=config['name'], phase=phase, round=round_index,
                      order_index=order_index, profiles={})
        target = report['preflight'] if phase == 'preflight' else report['visits']
        target.append(record)
        try:
            record['startup_seconds'] = engine.start()
            record.update(command=engine.command, pid=engine.process.pid, metadata=engine.metadata,
                          removed_environment_keys=engine.removed_environment_keys,
                          environment_policy='Automatic KleidiAI dispatch; inherited KleidiAI debug/thread overrides and library/backend injection paths removed.')
            for profile in profiles:
                ids = prompts[profile]
                row = dict(warmups=[])
                record['profiles'][profile] = row
                for _ in range(args.warmups):
                    result = engine.generate(ids, args.tokens, fixed_count=True)
                    validate_result(result, ids, args.tokens, expected_ids.get(profile))
                    expected_ids.setdefault(profile, result['output_token_ids'])
                    row['warmups'].append(result)
                if phase != 'preflight':
                    result = engine.generate(ids, args.tokens, fixed_count=True)
                    validate_result(result, ids, args.tokens, expected_ids[profile])
                    row['scored'] = result
        finally:
            if hasattr(engine, 'command'):
                record['command'] = engine.command
            record['metadata'] = engine.metadata
            engine.close()
            if hasattr(engine, 'log_path') and engine.log_path.is_file():
                record['runtime_log'] = dict(path=str(engine.log_path), sha256=sha(engine.log_path))
            save()
    try:
        save()
        # Establish same-precision trajectory identity for every configuration
        # before collecting any scored sample; warm each later visit again.
        for config in configs:
            visit(config, 'preflight')
        report['preflight_same_precision_ids_passed'] = True
        report['expected_output_ids'] = expected_ids
        for round_index in range(args.rounds):
            offset = (round_index // 2) % len(configs)
            order = list(range(len(configs)))[offset:] + list(range(len(configs)))[:offset]
            if round_index % 2:
                order.reverse()  # Every adjacent pair of rounds reverses all pairwise ordering.
            for order_index, index in enumerate(order):
                visit(configs[index], 'scored', round_index, order_index)
                scored = report['visits'][-1]['profiles']
                print(json.dumps(dict(round=round_index, configuration=configs[index]['name'],
                    results={profile: {metric: scored[profile]['scored'][metric] for metric in METRICS}
                             for profile in profiles})), flush=True)
        summarize(report)
        for record in builds.values():
            verify_build(record)
        if sha(model) != report['model']['sha256'] or any(sha(path) != digest for path, digest in report['source_hashes'].items()):
            raise RuntimeError('Model or benchmark source changed during evaluation')
        if any(sha(path) != digest for path, digest in report['tokenization'].items()):
            raise RuntimeError('Tokenization sources changed during evaluation')
        report['guard_after'] = guard(args.guard_state_dir)
        report['host_after'] = host_snapshot()
        report['paired_sampling_complete'] = args.mode == 'confirmation'
        # This helper establishes workload parity and paired sampling, not the
        # separate full-logit/quality gate or uncertainty review required for a claim.
        report['performance_claim_eligible'] = False
        report['completed'] = True
    except BaseException as error:
        report['error'] = dict(type=type(error).__name__, message=str(error))
        raise
    finally:
        save()


if __name__ == '__main__':
    main()
