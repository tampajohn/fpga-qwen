"""Opt-in real browser native chat checks against a localhost server.

Start the existing chat server separately after native asset promotion, then run
this file outside benchmark_guard (chat itself acquires the shared guard). The
suite briefly owns the serving guard to verify its actual browser polling.
No server, compiler, exporter or inference starts merely by importing this file.
"""
import argparse
import json
from pathlib import Path
import sys
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.benchmark_guard import BenchmarkGate
from playwright.sync_api import sync_playwright, expect


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default='http://127.0.0.1:8766')
    parser.add_argument('--live', action='store_true', help='Explicitly allow deployment checks on localhost:8765')
    parser.add_argument('--expect-prefix-reuse', action='store_true')
    parser.add_argument('--expect-greedy-suffix', action='store_true',
                        help='Expect a capability-aware B2 engine; old build12 remains the no-flag case')
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--guard-state-dir', type=Path, default=ROOT/'build/benchmark-status')
    parser.add_argument('--chrome', default='/Applications/Google Chrome.app/Contents/MacOS/Google Chrome')
    args = parser.parse_args()
    url = urlparse(args.url)
    if url.scheme != 'http' or url.hostname not in ('127.0.0.1','localhost') or url.port != (8765 if args.live else 8766):
        parser.error('Use localhost:8766, or explicitly pass --live for localhost:8765')
    gate = BenchmarkGate(args.guard_state_dir)
    if gate.status()['running']:
        raise RuntimeError('Wait for the active benchmark before browser inference')
    args.directory.mkdir(parents=True, exist_ok=False)
    report = dict(completed=False, gpu_inference=False, inference_interface='Actual localhost browser chat',
                  url=args.url, live_deployment_check=args.live, expect_greedy_suffix=args.expect_greedy_suffix,
                  checks={}, replies=[], console_errors=[])
    def save():
        (args.directory/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    save()
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True,args=['--disable-gpu'],executable_path=args.chrome)
        try:
            page = browser.new_page(viewport={'width':1280,'height':900})
            page.on('pageerror',lambda error:report['console_errors'].append(str(error)))
            page.goto(args.url)
            expect(page.locator('#prompt')).to_be_enabled(timeout=30000)
            status = page.request.get(args.url+'/api/status').json()
            entry = next(item for item in status['backends'] if item['id']=='native-cpu-f16')
            assert entry['available'] is True
            if entry.get('default_eligible') is False:
                assert page.locator('#backend').input_value() != 'native-cpu-f16'
            else:
                expect(page.locator('#backend')).to_have_value('native-cpu-f16')
                assert page.locator('#backend option').first.get_attribute('value') == 'native-cpu-f16'
                page.reload()
                expect(page.locator('#prompt')).to_be_enabled(timeout=30000)
                expect(page.locator('#backend')).to_have_value('native-cpu-f16')
                report['checks']['latest_release_first_and_default_after_reload'] = True
            report['initial_selection'] = page.locator('#backend').input_value()
            # Native OS popups are not reliably driven in headless Chrome. Check
            # the DOM mutation that dismisses them while the select has focus.
            page.evaluate("""() => {
                window.backendMutations = 0;
                window.backendObserver = new MutationObserver(rows => window.backendMutations += rows.length);
                window.backendObserver.observe(document.querySelector('#backend'),
                    {subtree: true, childList: true, attributes: true, characterData: true});
            }""")
            available = page.locator('#backend option:not([disabled])').evaluate_all('(options) => options.map(o => o.value)')
            page.locator('#backend').focus()
            page.wait_for_timeout(3200)
            assert page.evaluate('window.backendMutations') == 0, 'Status polling mutated focused backend options'
            page.locator('#backend').select_option(available[1])
            expect(page.locator('#backend')).to_have_value(available[1])
            page.locator('#prompt').focus()
            page.wait_for_timeout(2200)
            expect(page.locator('#backend')).to_have_value(available[1])
            page.evaluate('window.backendObserver.disconnect()')
            report['checks']['focused_backend_stable_during_status_polling'] = True
            page.locator('#backend').select_option('native-cpu-f16')
            expect(page.locator('#workers-row')).to_be_hidden()
            expect(page.locator('#decoding')).to_have_value('chat')
            expect(page.locator('#suffix')).to_be_disabled()
            expect(page.locator('#suffix')).not_to_be_checked()
            expect(page.locator('#backend-tag')).to_contain_text('Native CPU')
            if not args.expect_greedy_suffix:
                expect(page.locator('#backend-about')).to_contain_text('Suffix drafting is unavailable')
            report['checks']['catalog_and_controls'] = True
            page.locator('#decoding').select_option('greedy')
            page.locator('#reply-limit').select_option('128')
            def send(text):
                page.locator('#prompt').fill(text)
                with page.expect_response(lambda response: response.request.method == 'POST' and urlparse(response.url).path == '/api/chat') as received, page.expect_request(lambda request: request.method == 'POST' and urlparse(request.url).path == '/api/chat') as sent:
                    page.locator('#send').click()
                payload = sent.value.post_data_json
                expect(page.locator('#prompt')).to_be_enabled(timeout=180000)
                expect(page.locator('#error')).to_be_hidden()
                reply = page.locator('.assistant .content').last.inner_text()
                assert reply.strip()
                events = [json.loads(line) for line in received.value.body().decode().splitlines() if line.strip()]
                completed = [event for event in events if event.get('type') == 'done']
                assert len(completed) == 1
                done = completed[0]
                report['replies'].append(dict(prompt=text,text=reply,
                    requested_max_new_tokens=payload.get('max_new_tokens'),
                    token_ids=done.get('token_ids'), finish_reason=done.get('finish_reason'), metrics=done.get('metrics'),
                    request=dict(backend=payload.get('backend'), decoding=payload.get('decoding'), max_draft=payload.get('max_draft')),
                    metadata=page.locator('.assistant .reply-meta').last.inner_text()))
                save()
                return reply
            send('Answer with the single word blue.')
            send('What word did you just say? Answer with that word only.')
            # Record behavior, without turning a weak model response into a new quality rubric.
            expect(page.locator('#reused')).to_have_text('0')
            status = page.request.get(args.url+'/api/status').json()
            assert status['active_backend']=='native-cpu-f16' and status['device']=='cpu'
            assert status['gpu_inference'] is False and status['suffix_supported'] is args.expect_greedy_suffix
            assert status['prefix_reuse_enabled'] is args.expect_prefix_reuse
            if not args.expect_prefix_reuse:
                assert status['cached_tokens']==0
            report['checks']['fresh_context_status'] = True
            if args.expect_greedy_suffix:
                native = next(item for item in status['backends'] if item['id'] == 'native-cpu-f16')
                assert native['suffix_supported'] is True and native['suffix_greedy_only'] is True
                assert native['max_draft'] == 1 and status['max_verified_draft'] == 1
                expect(page.locator('#suffix')).to_be_enabled(timeout=15000)
                expect(page.locator('#suffix')).not_to_be_checked()
                expect(page.locator('#suffix-note')).to_contain_text('one verified n-gram draft token')
                page.locator('#suffix').check()
                page.locator('#decoding').select_option('chat')
                expect(page.locator('#suffix')).not_to_be_checked()
                expect(page.locator('#suffix')).to_be_disabled()
                expect(page.locator('#suffix-note')).to_contain_text('Greedy only')
                send('Answer with the single word blue.')
                assert report['replies'][-1]['request'] == dict(backend='native-cpu-f16', decoding='chat', max_draft=0)
                page.locator('#decoding').select_option('greedy')
                expect(page.locator('#suffix')).to_be_enabled()
                expect(page.locator('#suffix')).not_to_be_checked()
                # Status polling must not silently re-enable the earlier opt-in.
                page.wait_for_timeout(2200)
                expect(page.locator('#suffix')).not_to_be_checked()
                report['checks']['greedy_suffix_mode_gating_and_chat_payload'] = True
                page.locator('#reply-limit').select_option('128')
                counting_prompt = 'Count from 1 to 40, separated by commas, without commentary.'
                page.locator('#new-chat').click()
                page.locator('#suffix').uncheck()
                send(counting_prompt)
                baseline = report['replies'][-1]
                assert baseline['request']['max_draft'] == 0
                assert baseline['requested_max_new_tokens'] == 128
                assert baseline['metrics']['reused_prefix_tokens'] == 0
                page.locator('#new-chat').click()
                page.locator('#suffix').check()
                send(counting_prompt)
                candidate = report['replies'][-1]
                assert candidate['request'] == dict(backend='native-cpu-f16', decoding='greedy', max_draft=1)
                assert candidate['requested_max_new_tokens'] == 128
                assert candidate['metrics']['reused_prefix_tokens'] == 0
                assert candidate['metrics']['suffix_enabled'] is True
                assert candidate['metrics']['proposed_tokens'] > 0
                assert candidate['metrics']['accepted_tokens'] > 0
                for field in ('text', 'token_ids', 'finish_reason'):
                    assert candidate[field] == baseline[field], field
                assert candidate['token_ids'] and len(candidate['token_ids']) <= 128
                assert candidate['finish_reason'] in ('eos', 'length')
                if candidate['finish_reason'] == 'length':
                    assert len(candidate['token_ids']) == 128
                report['checks']['greedy_suffix_on_browser_request'] = dict(
                    reply_cap=128, cap_control='production UI selection',
                    proposed_tokens=candidate['metrics']['proposed_tokens'],
                    accepted_tokens=candidate['metrics']['accepted_tokens'],
                    exact_visible_ids_text_and_stop=True, fresh_sessions=True,
                    timing_claim=False)
                page.locator('#suffix').uncheck()
                page.locator('#reply-limit').select_option('128')
            else:
                expect(page.locator('#suffix')).to_be_disabled()
                report['checks']['old_binary_suffix_unavailable'] = True
            if args.expect_prefix_reuse:
                expect(page.locator('#backend-about')).to_contain_text('Reuses completed 512-token prefill blocks', timeout=15000)
                page.locator('#new-chat').click()
                send('Background notes:\n' + 'This is filler. ' * 180 + '\nAnswer with the single word blue.')
                expect(page.locator('#reused')).to_have_text('0')
                reply = send('Now answer with the single word green.')
                assert reply.strip().lower().rstrip('.') == 'green', reply
                reused = int(page.locator('#reused').inner_text().replace(',', ''))
                assert reused >= 512 and reused % 512 == 0
                report['checks']['canonical_prefix_reuse_and_changed_question'] = dict(reused_tokens=reused)
                page.screenshot(path=str(args.directory/'prefix-reuse.png'))
            # Guard visibility is real: hold the same lock, let the browser poll,
            # and verify the server also rejects an attempted bypass before inference.
            handle = gate.acquire(exclusive=True)
            if handle is None: raise RuntimeError('Another chat/benchmark acquired the guard')
            try:
                gate.describe('Browser native CPU guard visibility test')
                page.locator('#prompt').fill('Pending until the benchmark finishes')
                expect(page.locator('#benchmark-banner')).to_be_visible(timeout=15000)
                expect(page.locator('#send')).to_be_disabled()
                response = page.request.post(args.url+'/api/chat',data=dict(
                    request_id='browser-native-guard-check',session='guard-check',backend='native-cpu-f16',
                    messages=[dict(role='user',content='hi')],max_new_tokens=1,max_draft=0))
                assert response.status==409
                report['checks']['guard_visible_and_server_blocks']=True
            finally:
                gate.finish(0);handle.close()
            expect(page.locator('#send')).to_be_enabled(timeout=15000)
            page.locator('#prompt').fill('Count from one to five hundred, one number per line.')
            page.locator('#reply-limit').select_option('256')
            page.locator('#send').click()
            page.wait_for_function("async () => (await (await fetch('/api/status')).json()).busy === true", timeout=30000)
            page.locator('#stop').click()
            expect(page.locator('#prompt')).to_be_enabled(timeout=180000)
            expect(page.locator('.assistant .reply-meta').last).to_contain_text('Stopped')
            page.locator('#new-chat').click()
            page.locator('#reply-limit').select_option('128')
            send('Answer with the single word green.')
            report['checks']['cancel_then_fresh_request']=True
            llama = next((item for item in status['backends'] if item['id']=='llama-f16' and item['available']),None)
            if llama is None: raise RuntimeError('Install existing llama-f16 comparison assets for switch test')
            page.locator('#backend').select_option('llama-f16')
            page.locator('#new-chat').click()
            send('Answer with the single word blue.')
            page.locator('#backend').select_option('native-cpu-f16')
            page.locator('#new-chat').click()
            send('Answer with the single word blue.')
            expect(page.locator('#reused')).to_have_text('0')
            report['checks']['backend_switch_roundtrip']=True
            page.screenshot(path=str(args.directory/'native-chat.png'))
            assert not report['console_errors'],report['console_errors']
            report['completed']=True
        except BaseException as error:
            report['error']=dict(type=type(error).__name__,message=str(error));raise
        finally:
            save();browser.close()
    print(args.directory/'report.json')


if __name__=='__main__':main()
