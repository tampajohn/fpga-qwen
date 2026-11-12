const $ = (id) => document.getElementById(id);
let session = crypto.randomUUID();
let messages = [];
let ready = false;
let busy = false;
let requestId = null;
let backends = [];
let benchmarkRunning = false;
let checkingStatus = false;

function connectionLabel() {
  if (selectedBackend()?.simulation === true) {
    if (selectedBackend()?.fixed_workers) return 'Local simulation · 12 prefill / 8 decode workers';
    const count = Number($('workers').value);
    return count ? `Local simulation · ${count} worker${count === 1 ? '' : 's'}` : 'Local simulation · serial baseline';
  }
  return `Local engine · ${selectedBackend()?.cpu_threads ?? 6} CPU threads`;
}
function selectedBackend() { return backends.find(b => b.id === $('backend').value); }
function suffixAllowed() {
  const backend = selectedBackend();
  return Boolean(backend?.suffix_supported && (!backend.suffix_greedy_only || $('decoding').value === 'greedy'));
}
function refreshBackendOptions() {
  const select = $('backend');
  // Native select popups can close when even an unchanged option is rewritten.
  // Focus also covers keyboard interaction; apply pending metadata after blur.
  if (document.activeElement === select) return;
  const selected = select.value;
  for (const option of select.options) {
    const backend = backends.find(item => item.id === option.value);
    if (!backend) continue;
    const label = backend.label + (backend.available ? '' : ' (not installed)');
    const disabled = !backend.available;
    if (option.textContent !== label) option.textContent = label;
    if (option.disabled !== disabled) option.disabled = disabled;
  }
  if (select.value !== selected) select.value = selected;
}
function backendChanged() {
  const backend = selectedBackend();
  const rtl = selectedBackend()?.simulation === true;
  $('workers-row').hidden = !rtl;
  for (const option of $('workers').options) option.disabled = backend?.fixed_workers
    ? Number(option.value) !== backend.fixed_workers : Number(option.value) < (backend?.min_workers ?? 0);
  if (backend?.fixed_workers) $('workers').value = String(backend.fixed_workers);
  if (Number($('workers').value) < (backend?.min_workers ?? 0)) $('workers').value = '8';
  $('suffix').checked = false;
  renderBackendDetails();
  $('connection').textContent = connectionLabel();
  ['rate', 'first-token', 'reused', 'accepted', 'rounds'].forEach(id => $(id).textContent = '—');
  $('context').textContent = '0 / 4,096 context tokens';
  controls();
}
function renderBackendDetails() {
  const backend = selectedBackend();
  const rtl = backend?.simulation === true;
  $('suffix-note').textContent = backend?.suffix_supported
    ? (backend.suffix_greedy_only
      ? ($('decoding').value === 'greedy' ? 'Optional · one verified n-gram draft token; off by default' : 'Greedy only · choose Greedy to enable verified drafts')
      : $('backend').value.startsWith('rtl-fp32')
      ? `Optional · up to ${backend.max_draft} verified tokens; helps repeated text, little benefit on prose`
      : `Verified n-gram drafts · up to ${backend.max_draft} tokens; sampled and greedy`)
    : 'Unavailable in this backend';
  $('backend-tag').textContent = rtl ? 'Full SV simulation'
    : backend?.id === 'native-cpu-f16' ? `Native CPU · ${backend.build_id ?? 'experimental'}` : 'llama.cpp · CPU only';
  $('backend-about').textContent = backend?.id === 'native-cpu-f16'
    ? `Original C++ CPU engine with FP16 matrix weights, FP32 activations and normalization weights, and FP16 KV storage.${backend.norm_accumulator_precision === 'f64' ? ' Normalization sums accumulate in FP64.' : ''} Layer weights also have a widened FP32 cache for prefill. Uses ${backend.cpu_threads} decode / ${backend.prefill_workers} prefill threads; no GPU or PyTorch. ${backend.prefix_reuse_enabled
      ? 'Reuses completed 512-token prefill blocks in the same conversation; partial tails are recomputed.'
      : backend.prefix_reuse_supported
      ? 'This engine supports completed 512-token prefill blocks, but prefix reuse is disabled.'
      : 'Prefix reuse is not enabled for the current engine; messages are fully recomputed. Supporting engines reuse completed 512-token prefill blocks and recompute partial tails.'} ${backend.suffix_supported ? 'Optional one-token n-gram verification is available in Greedy mode and is off by default.' : 'Suffix drafting is unavailable in this engine.'}`
    : $('backend').value === 'rtl-fp32-fused-phase'
    ? 'Experimental fused FP32 prefill with 12 workers; separate FP32 decode and verification with 8 workers. Attention and KV remain FP64. Reuses only complete fused prefill blocks. Both pools stay resident; only one runs at a time. Optional suffix drafting is off by default. CPU only; not cycle-accurate FPGA timing.'
    : backend?.fixed_workers
    ? 'Same FP32 hybrid math and canonical prefix reuse, with 12 workers for prefill and 8 for decoding and verification. Both pools stay resident; only one runs at a time. Worker counts are fixed. Optional suffix drafting is off by default. CPU only; not cycle-accurate FPGA timing.'
    : $('backend').value === 'rtl-fp32'
    ? 'SystemVerilog uses FP32 scalar projections for decoding and FP64 for prefill, attention and KV storage. It reuses complete FP64 prefill blocks. Optional FP32 suffix verification helped repeated code in testing; ordinary prose was essentially unchanged. Drafting is off by default. CPU only; not cycle-accurate FPGA timing.'
    : rtl
    ? 'All model math and KV data run in a behavioral SystemVerilog model via Verilator. It uses floating-point real arithmetic and is not synthesizable or cycle-accurate. Python handles tokenization, suffix proposals, sampling random numbers, and HTTP; the model verifies every proposed token.'
    : 'llama.cpp runs Qwen3-0.6B on CPU with native prefix caching. This comparison backend does not use SystemVerilog. GPU support is disabled in the build.';
}
$('backend').addEventListener('change', backendChanged);
$('backend').addEventListener('blur', () => queueMicrotask(refreshBackendOptions));
$('workers').addEventListener('change', backendChanged);
$('decoding').addEventListener('change', () => {
  if (!suffixAllowed()) $('suffix').checked = false;
  renderBackendDetails();
  controls();
});

function showError(message) {
  $('error').textContent = message;
  $('error').hidden = !message;
}
function controls() {
  $('send').disabled = !ready || busy || benchmarkRunning || !$('prompt').value.trim();
  $('prompt').disabled = !ready || busy;
  $('new-chat').disabled = busy;
  $('suffix').disabled = !ready || busy || !suffixAllowed();
  const backendDisabled = busy || !ready;
  if ($('backend').disabled !== backendDisabled) $('backend').disabled = backendDisabled;
  $('workers').disabled = busy || !ready || Boolean(selectedBackend()?.fixed_workers);
  $('reply-limit').disabled = busy;
  $('decoding').disabled = busy;
  $('send').hidden = busy;
  $('stop').hidden = !busy;
  document.querySelectorAll('[data-prompt]').forEach((b) => b.disabled = !ready || busy);
}
function scrollDown(force = false) {
  const box = $('conversation');
  if (force || box.scrollHeight - box.scrollTop - box.clientHeight < 160) box.scrollTop = box.scrollHeight;
}
function bubble(role, text = '') {
  const wrapper = document.createElement('article');
  wrapper.className = `message ${role}`;
  const label = document.createElement('div');
  label.className = 'label';
  label.textContent = role === 'user' ? 'You' : 'Qwen';
  const content = document.createElement('div');
  content.className = 'content';
  content.textContent = text;
  const meta = document.createElement('div');
  meta.className = 'reply-meta';
  wrapper.append(label, content, meta);
  $('messages').append(wrapper);
  return { wrapper, content, meta };
}
function updateMetrics(m) {
  if (!m) return;
  $('rate').textContent = m.output_tokens ? m.decode_tokens_per_second.toFixed(1) : '—';
  $('first-token').textContent = Number.isFinite(m.first_token_ms) ? (m.first_token_ms / 1000).toFixed(2) : '—';
  $('reused').textContent = m.reused_prefix_tokens.toLocaleString();
  $('uncached').textContent = Number.isFinite(m.prompt_tokens) && Number.isFinite(m.reused_prefix_tokens)
    ? Math.max(0, m.prompt_tokens - m.reused_prefix_tokens).toLocaleString() : '—';
  $('accepted').textContent = selectedBackend()?.suffix_supported === false ? '—' : `${m.accepted_tokens} / ${m.proposed_tokens}`;
  $('rounds').textContent = m.rounds ? m.tokens_per_round.toFixed(2) : '—';
  $('context').textContent = `${m.kv_tokens.toLocaleString()} / 4,096 context tokens`;
}

async function checkStatus() {
  if (checkingStatus) return;
  checkingStatus = true;
  try {
    const response = await fetch('/api/status');
    if (!response.ok) throw new Error('Local engine is unavailable');
    const status = await response.json();
    if (status.device !== 'cpu' || status.gpu_inference !== false) throw new Error('CPU-only backend check failed');
    ready = status.ready;
    benchmarkRunning = status.benchmark?.running === true;
    $('benchmark-banner').hidden = !benchmarkRunning;
    $('benchmark-banner').textContent = benchmarkRunning
      ? `Benchmark running: ${status.benchmark.name || 'CPU benchmark'}. Sending is paused; your draft is kept.` : '';
    const lastRun = status.benchmark?.last_run;
    $('benchmark-last').hidden = benchmarkRunning || !lastRun;
    $('benchmark-last').textContent = lastRun
      ? `Last benchmark: ${lastRun.name || 'CPU benchmark'} · ${lastRun.finished_at
        ? `ended ${new Date(lastRun.finished_at).toLocaleTimeString()}` : 'no longer running'}` : '';
    if (!backends.length && status.backends) {
      const releases = ['native-cpu-f16', 'rtl-fp32-fused-phase', 'rtl-fp32-phase', 'rtl-fp32', 'rtl'];
      const rank = b => releases.includes(b.id) ? releases.indexOf(b.id) : releases.length;
      backends = [...status.backends].sort((a, b) =>
        Number(b.available) - Number(a.available) || rank(a) - rank(b));
      $('workers').value = String(status.simulation_workers ?? 8);
      $('backend').replaceChildren(...backends.map(b => {
        const option = document.createElement('option');
        option.value = b.id; option.textContent = b.label + (b.available ? '' : ' (not installed)');
        option.disabled = !b.available;
        return option;
      }));
      const latest = backends.find(b => b.available && b.default_eligible !== false);
      if (latest) $('backend').value = latest.id;
      backendChanged();
    } else if (status.backends) {
      // Refresh capabilities after startup without changing the user's engine,
      // suffix selection, worker controls, or the currently displayed metrics.
      const current = new Map(status.backends.map(backend => [backend.id, backend]));
      backends = backends.map(backend => current.get(backend.id) ?? backend);
      refreshBackendOptions();
      if (!suffixAllowed()) $('suffix').checked = false;
      renderBackendDetails();
    }
    $('connection').textContent = connectionLabel();
    $('connection-dot').classList.remove('offline');
    if (!busy) $('activity').textContent = benchmarkRunning ? 'Benchmark running · waiting to send'
      : status.busy ? 'Another reply is running' : 'Ready · CPU only';
  } catch (error) {
    ready = false;
    $('connection').textContent = 'Local engine disconnected';
    $('connection-dot').classList.add('offline');
    if (!busy) $('activity').textContent = 'Engine offline — start it with make chat';
  }
  checkingStatus = false;
  controls();
}

async function send(event) {
  event.preventDefault();
  const prompt = $('prompt').value.trim();
  if (!prompt || busy || !ready || benchmarkRunning) return;
  showError('');
  busy = true;
  requestId = crypto.randomUUID();
  $('welcome').hidden = true;
  bubble('user', prompt);
  const reply = bubble('assistant');
  reply.wrapper.classList.add('pending');
  $('prompt').value = '';
  $('prompt').style.height = '';
  $('stop').disabled = false;
  $('activity').textContent = 'Preparing context…';
  controls();
  scrollDown(true);
  const submitted = [...messages, { role: 'user', content: prompt }];
  let finished = false;
  let userStopped = false;
  try {
    const response = await fetch('/api/chat', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ request_id: requestId, session, backend: $('backend').value, simulation_workers: Number($('workers').value), decoding: $('decoding').value, messages: submitted,
        max_new_tokens: Number($('reply-limit').value), max_draft: $('suffix').checked && suffixAllowed() ? (selectedBackend()?.max_draft ?? 0) : 0 })
    });
    if (!response.ok) {
      const detail = await response.json();
      throw new Error(detail.error || `Request failed (${response.status})`);
    }
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    function receive(line) {
      if (!line.trim()) return;
      const event = JSON.parse(line);
      if (event.type === 'error') throw new Error(event.error);
      updateMetrics(event.metrics);
      if (event.type === 'loading') $('activity').textContent = event.message;
      if (event.type === 'start') $('activity').textContent = 'Preparing context…';
      if (event.type === 'prefill') $('activity').textContent = `Reading context · ${event.processed_tokens} tokens`;
      if (event.type === 'update' || event.type === 'done') {
        reply.content.textContent = event.text;
        $('activity').textContent = `Generating · ${event.metrics.output_tokens} tokens`;
        scrollDown();
      }
      if (event.type === 'done') {
        finished = true;
        userStopped = event.finish_reason === 'stopped';
        const m = event.metrics;
        const end = userStopped ? 'Stopped' : event.finish_reason === 'length' ? 'Reply limit reached' : 'Complete';
        const suffixLabel = (m.effective_max_draft || m.suffix_enabled) ? ` · ${m.accepted_tokens}/${m.proposed_tokens} drafts accepted` : '';
        const workerLabel = m.prefill_workers ? ` · ${m.prefill_workers} prefill / ${m.decode_workers} decode workers` : m.simulation_workers === undefined ? '' : m.simulation_workers ? ` · ${m.simulation_workers} workers` : ' · serial';
        reply.meta.textContent = `${selectedBackend()?.label ?? "SystemVerilog"}${workerLabel} · ${$('decoding').value === 'chat' ? 'Sampled' : 'Greedy'} · ${end} · ${m.output_tokens} tokens · ${m.elapsed_seconds.toFixed(2)} s · ${m.reused_prefix_tokens} prefix tokens reused${suffixLabel}`;
        if (event.text.trim()) messages = [...submitted, { role: 'assistant', content: event.text }];
        else {
          reply.content.textContent = userStopped ? 'Stopped before a reply was generated.' : 'The model returned an empty reply.';
          $('prompt').value = prompt;
        }
      }
    }
    while (true) {
      const { value, done } = await reader.read();
      buffer += decoder.decode(value, { stream: !done });
      const lines = buffer.split('\n');
      buffer = lines.pop();
      lines.forEach(receive);
      if (done) break;
    }
    if (buffer.trim()) receive(buffer);
    if (!finished) throw new Error('Connection ended before the reply finished. You can retry your message.');
  } catch (error) {
    showError(error.message);
    reply.wrapper.classList.add('failed');
    reply.meta.textContent = 'Not added to conversation context. Retry your message.';
    if (!reply.content.textContent) reply.content.textContent = 'The reply could not be completed.';
    $('prompt').value = prompt;
  } finally {
    reply.wrapper.classList.remove('pending');
    busy = false;
    requestId = null;
    $('activity').textContent = userStopped ? 'Stopped · ready for your next message' : 'Ready · CPU only';
    controls();
    $('prompt').focus();
    scrollDown();
  }
}

$('chat-form').addEventListener('submit', send);
$('prompt').addEventListener('input', () => {
  $('prompt').style.height = 'auto';
  $('prompt').style.height = Math.min($('prompt').scrollHeight, 180) + 'px';
  controls();
});
$('prompt').addEventListener('keydown', (event) => {
  if (event.key === 'Enter' && !event.shiftKey && !event.isComposing) {
    event.preventDefault();
    $('chat-form').requestSubmit();
  }
});
$('stop').addEventListener('click', async () => {
  if (!requestId) return;
  $('stop').disabled = true;
  $('activity').textContent = 'Stopping…';
  try {
    const response = await fetch('/api/stop', { method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ request_id: requestId }) });
    if (!response.ok) throw new Error('Could not stop the request');
  } catch (error) { showError(error.message); $('stop').disabled = false; }
});
$('new-chat').addEventListener('click', () => {
  if (busy) return;
  session = crypto.randomUUID();
  messages = [];
  $('messages').replaceChildren();
  $('welcome').hidden = false;
  $('prompt').value = '';
  ['rate', 'first-token', 'reused', 'accepted', 'rounds'].forEach((id) => $(id).textContent = '—');
  $('context').textContent = '0 / 4,096 context tokens';
  showError('');
  controls();
  $('prompt').focus();
});
document.querySelectorAll('[data-prompt]').forEach((button) => button.addEventListener('click', () => {
  $('prompt').value = button.dataset.prompt;
  controls();
  $('prompt').focus();
}));
controls();
checkStatus();
setInterval(checkStatus, 1000);
