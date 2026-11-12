"""Opt-in fixed-phase browser smoke against an isolated server; no timing claim."""
import json
import os
from pathlib import Path
from playwright.sync_api import sync_playwright,expect


def main():
    url=os.environ.get('CHAT_URL','http://127.0.0.1:8766')
    output=Path(__file__).resolve().parents[1]/'build/browser-phase'
    output.mkdir(parents=True,exist_ok=True)
    with sync_playwright() as p:
        browser=p.chromium.launch(headless=True,args=['--disable-gpu'],executable_path='/Applications/Google Chrome.app/Contents/MacOS/Google Chrome')
        try:
            page=browser.new_page();errors=[]
            page.on('pageerror',lambda error:errors.append(str(error)))
            page.goto(url);expect(page.locator('#prompt')).to_be_enabled()
            expect(page.locator('#backend')).to_have_value('rtl')
            page.locator('#workers').select_option('2')
            page.locator('#backend').select_option('rtl-fp32-phase')
            expect(page.locator('#workers')).to_have_value('8')
            expect(page.locator('#workers')).to_be_disabled()
            expect(page.locator('#suffix')).to_be_enabled()
            expect(page.locator('#suffix')).not_to_be_checked()
            page.locator('#decoding').select_option('greedy')
            def send(text):
                page.locator('#prompt').fill(text);page.locator('#send').click()
                expect(page.locator('#prompt')).to_be_enabled(timeout=120000)
                assert not page.locator('#error').is_visible()
                return page.locator('.assistant .content').last.inner_text()
            first=send('Answer with the single word blue.')
            second=send('Repeat that same word once.')
            reused=int(page.locator('#reused').inner_text().replace(',',''))
            assert first.strip() and second.strip() and reused>0 and reused%8==0
            status=page.request.get(url+'/api/status').json()
            assert status['phase_verified'] and status['precision_verified']
            assert status['prefill_workers']==12 and status['decode_workers']==8
            assert status['canonical_prefix_policy']=='fp64_b8_aligned_v1'
            assert '12 prefill / 8 decode' in page.locator('.assistant .reply-meta').last.inner_text()
            page.locator('#suffix').check()
            third=send('Repeat blue three times.')
            assert third.strip()
            page.locator('#backend').select_option('rtl-fp32')
            expect(page.locator('#workers')).to_be_enabled()
            page.locator('#workers').select_option('2')
            fourth=send('Answer with blue only.')
            generic=page.request.get(url+'/api/status').json()
            assert generic['active_backend']=='rtl-fp32' and generic['simulation_workers']==2
            assert not generic.get('phase_verified',False)
            page.locator('#backend').select_option('rtl')
            expect(page.locator('#workers')).to_be_enabled()
            assert not errors
            page.screenshot(path=str(output/'phase.png'))
            (output/'result.json').write_text(json.dumps(dict(passed=True,replies=[first,second,third,fourth],reused_prefix_tokens=reused,status=status,generic_status=generic,javascript_errors=errors,timing_claim=False),indent=2)+'\n')
        finally:browser.close()

if __name__=='__main__':main()
