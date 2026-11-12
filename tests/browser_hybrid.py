"""Opt-in real browser check on an isolated localhost server (default port 8766)."""
import json
import os
from pathlib import Path
from playwright.sync_api import sync_playwright, expect


def main():
    url=os.environ.get('CHAT_URL','http://127.0.0.1:8766')
    output=Path(__file__).resolve().parents[1]/'build/browser-hybrid'
    output.mkdir(parents=True,exist_ok=True)
    with sync_playwright() as p:
        browser=p.chromium.launch(headless=True,args=['--disable-gpu'],
            executable_path='/Applications/Google Chrome.app/Contents/MacOS/Google Chrome')
        try:
            page=browser.new_page(viewport={'width':1280,'height':900})
            errors=[]
            page.on('pageerror',lambda error:errors.append(str(error)))
            page.goto(url)
            expect(page.locator('#prompt')).to_be_enabled()
            expect(page.locator('#backend')).to_have_value('rtl')
            page.locator('#decoding').select_option('greedy')
            page.locator('#reply-limit').select_option('128')
            def send(text):
                page.locator('#prompt').fill(text)
                page.locator('#send').click()
                expect(page.locator('#prompt')).to_be_enabled(timeout=120000)
                assert not page.locator('#error').is_visible()
                reply=page.locator('.assistant .content').last.inner_text()
                assert reply.strip()
                return reply
            replies=[send('Answer with the single word blue.')]
            page.locator('#workers').select_option('0')
            page.locator('#backend').select_option('rtl-fp32')
            expect(page.locator('#workers')).to_have_value('8')
            expect(page.locator('#workers option[value="0"]')).to_be_disabled()
            expect(page.locator('#suffix')).to_be_enabled()
            expect(page.locator('#suffix')).not_to_be_checked()
            assert 'up to 3' in page.locator('#suffix-note').inner_text()
            replies.append(send('What word did you just say? Answer with that word only.'))
            expect(page.locator('#reused')).to_have_text('0')
            assert 'FP32 hybrid' in page.locator('.assistant .reply-meta').last.inner_text()
            first_status=page.request.get(url+'/api/status').json()
            assert first_status['precision_verified'] is True
            assert first_status['prefix_reuse_enabled'] is True
            assert first_status['canonical_prefix_policy']=='fp64_b8_aligned_v1'
            assert first_status['precision_profile']=='fp32_b1_fp64_prefill'
            assert first_status['verification_projection']=='fp32'
            assert first_status['max_verified_draft']==3
            page.locator('#workers').select_option('2')
            replies.append(send('Repeat that word once.'))
            reused=int(page.locator('#reused').inner_text().replace(',',''))
            assert reused>0 and reused%8==0
            assert '2 workers' in page.locator('.assistant .reply-meta').last.inner_text()
            page.screenshot(path=str(output/'hybrid.png'))
            page.locator('#backend').select_option('rtl')
            expect(page.locator('#workers option[value="0"]')).to_be_enabled()
            expect(page.locator('#suffix')).to_be_enabled()
            replies.append(send('Repeat that word once.'))
            assert page.request.get(url+'/api/status').json()['precision_profile']=='fp64'
            page.locator('#backend').select_option('rtl-fp32')
            page.locator('#prompt').fill('Count from one to five hundred, one number per line.')
            page.locator('#send').click()
            page.locator('#stop').click()
            expect(page.locator('#prompt')).to_be_enabled(timeout=120000)
            assert 'Stopped' in page.locator('.assistant .reply-meta').last.inner_text()
            replies.append(send('Answer with the single word blue.'))
            expect(page.locator('#reused')).to_have_text('0')
            # Explicitly enable drafting only for this repetition comparison.
            pattern='Repeat this code exactly three times, without an introduction:\ndef add(a, b):\n    return a + b'
            page.locator('#new-chat').click()
            expect(page.locator('#suffix')).not_to_be_checked()
            off=send(pattern)
            page.locator('#new-chat').click()
            page.locator('#suffix').check()
            on=send(pattern)
            assert on==off
            accepted=page.locator('#accepted').inner_text()
            assert int(accepted.split('/')[0].strip())>0
            assert 'drafts accepted' in page.locator('.assistant .reply-meta').last.inner_text()
            page.screenshot(path=str(output/'hybrid-suffix.png'))
            page.locator('#backend').select_option('rtl')
            expect(page.locator('#suffix')).not_to_be_checked()
            assert not errors,errors
            report=dict(replies=replies,default_backend='rtl',hybrid_precision_verified=True,
                        hybrid_second_turn_reused_tokens=reused,canonical_prefix_policy=first_status["canonical_prefix_policy"],suffix_enabled=True,suffix_default_off=True,suffix_limit=3,serial_disabled=True,
                        suffix_on_off_text_equal=True,suffix_acceptance=accepted,
                        worker_change=True,switch_back=True,cancel_restart=True,console_errors=errors)
            (output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
            print(json.dumps(report,indent=2))
        finally:browser.close()


if __name__=='__main__':main()
