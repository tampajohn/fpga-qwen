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
                  url=args.url, live_deployment_check=args.live,
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
            page.locator('#backend').select_option('native-cpu-f16')
            expect(page.locator('#workers-row')).to_be_hidden()
            expect(page.locator('#suffix')).to_be_disabled()
            expect(page.locator('#suffix')).not_to_be_checked()
            expect(page.locator('#backend-tag')).to_contain_text('Native CPU')
            expect(page.locator('#backend-about')).to_contain_text('prefix reuse and suffix drafting are unavailable')
            report['checks']['catalog_and_controls'] = True
            page.locator('#decoding').select_option('greedy')
            page.locator('#reply-limit').select_option('128')
            def send(text):
                page.locator('#prompt').fill(text)
                page.locator('#send').click()
                expect(page.locator('#prompt')).to_be_enabled(timeout=180000)
                expect(page.locator('#error')).to_be_hidden()
                reply = page.locator('.assistant .content').last.inner_text()
                assert reply.strip()
                report['replies'].append(dict(prompt=text,text=reply,
                    metadata=page.locator('.assistant .reply-meta').last.inner_text()))
                save()
                return reply
            send('Answer with the single word blue.')
            send('What word did you just say? Answer with that word only.')
            # Record behavior, without turning a weak model response into a new quality rubric.
            expect(page.locator('#reused')).to_have_text('0')
            status = page.request.get(args.url+'/api/status').json()
            assert status['active_backend']=='native-cpu-f16' and status['device']=='cpu'
            assert status['gpu_inference'] is False and status['prefix_reuse_enabled'] is False
            assert status['cached_tokens']==0 and status['suffix_supported'] is False
            report['checks']['fresh_context_status'] = True
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
