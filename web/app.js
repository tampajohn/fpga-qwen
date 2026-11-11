const $ = (id) => document.getElementById(id);
let session = crypto.randomUUID();
let messages = [];
let ready = false;
let busy = false;
let requestId = null;
let backends = [];

function connectionLabel() {
  if ($('backend').value === 'rtl') {
    const count = Number($('workers').value);
    return count ? `Local simulation · ${count} worker${count === 1 ? '' : 's'}` : 'Local simulation · serial baseline';
  }
  return `Local engine · ${selectedBackend()?.cpu_threads ?? 6} CPU threads`;
}
function selectedBackend() { return backends.find(b => b.id === $('backend').value); }
function backendChanged() {
  const backend = selectedBackend();
  const rtl = $('backend').value === 'rtl';
  $('workers-row').hidden = !rtl;
  $('suffix').checked = false;
  $('suffix-note').textContent = backend?.suffix_supported
    ? `Verified n-gram drafts · up to ${backend.max_draft} tokens; sampled and greedy`
    : 'Unavailable in this backend';
  $('backend-tag').textContent = rtl ? 'Full SV simulation' : 'llama.cpp · CPU only';
  $('backend-about').textContent = rtl
    ? 'All model math and KV data run in a behavioral SystemVerilog model via Verilator. It uses floating-point real arithmetic and is not synthesizable or cycle-accurate. Python handles tokenization, suffix proposals, sampling random numbers, and HTTP; the model verifies every proposed token.'
    : 'llama.cpp runs Qwen3-0.6B on CPU with native prefix caching. This comparison backend does not use SystemVerilog. GPU support is disabled in the build.';
  $('connection').textContent = connectionLabel();
  ['rate', 'reused', 'accepted', 'rounds'].forEach(id => $(id).textContent = '—');
  $('context').textContent = '0 / 4,096 context tokens';
  controls();
}
$('backend').addEventListener('change', backendChanged);
$('workers').addEventListener('change', backendChanged);

function showError(message) {
  $('error').textContent = message;
  $('error').hidden = !message;
}
function controls() {
  $('send').disabled = !ready || busy || !$('prompt').value.trim();
  $('prompt').disabled = !ready || busy;
  $('new-chat').disabled = busy;
  $('suffix').disabled = !ready || busy || !selectedBackend()?.suffix_supported;
  $('backend').disabled = busy || !ready;
  $('workers').disabled = busy || !ready;
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
  $('reused').textContent = m.reused_prefix_tokens.toLocaleString();
  $('accepted').textContent = selectedBackend()?.suffix_supported === false ? '—' : `${m.accepted_tokens} / ${m.proposed_tokens}`;
  $('rounds').textContent = m.rounds ? m.tokens_per_round.toFixed(2) : '—';
  $('context').textContent = `${m.kv_tokens.toLocaleString()} / 4,096 context tokens`;
}

async function checkStatus() {
  try {
    const response = await fetch('/api/status');
    if (!response.ok) throw new Error('Local engine is unavailable');
    const status = await response.json();
    if (status.device !== 'cpu' || status.gpu_inference !== false) throw new Error('CPU-only backend check failed');
    ready = status.ready;
    if (!backends.length && status.backends) {
      backends = status.backends;
      $('workers').value = String(status.simulation_workers ?? 8);
      $('backend').replaceChildren(...backends.map(b => {
        const option = document.createElement('option');
        option.value = b.id; option.textContent = b.label + (b.available ? '' : ' (not installed)');
        option.disabled = !b.available;
        return option;
      }));
      backendChanged();
    }
    $('connection').textContent = connectionLabel();
    $('connection-dot').classList.remove('offline');
    if (!busy) $('activity').textContent = status.busy ? 'Another reply is running' : 'Ready · CPU only';
  } catch (error) {
    ready = false;
    $('connection').textContent = 'Local engine disconnected';
    $('connection-dot').classList.add('offline');
    if (!busy) $('activity').textContent = 'Engine offline — start it with make chat';
  }
  controls();
}

async function send(event) {
  event.preventDefault();
  const prompt = $('prompt').value.trim();
  if (!prompt || busy || !ready) return;
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
        max_new_tokens: Number($('reply-limit').value), max_draft: $('suffix').checked ? 7 : 0 })
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
        const suffixLabel = m.effective_max_draft ? ` · ${m.accepted_tokens}/${m.proposed_tokens} drafts accepted` : '';
        const workerLabel = m.simulation_workers === undefined ? '' : m.simulation_workers ? ` · ${m.simulation_workers} workers` : ' · serial';
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
  ['rate', 'reused', 'accepted', 'rounds'].forEach((id) => $(id).textContent = '—');
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
setInterval(checkStatus, 10000);
