"""Real localhost UI smoke for an explicitly block-enabled candidate server."""
import json
import os
from pathlib import Path
from playwright.sync_api import sync_playwright,expect

ROOT=Path(__file__).resolve().parents[1]


def main():
    output=ROOT/'build/browser-suffix'
    output.mkdir(parents=True,exist_ok=True)
    with sync_playwright() as p:
        browser=p.chromium.launch(headless=True,args=['--disable-gpu'],
            executable_path='/Applications/Google Chrome.app/Contents/MacOS/Google Chrome')
        page=browser.new_page(viewport={'width':1280,'height':900})
        errors=[]
        page.on('pageerror',lambda error:errors.append(str(error)))
        page.goto(os.environ.get('CHAT_URL','http://127.0.0.1:8766'))
        expect(page.locator('#prompt')).to_be_enabled()
        expect(page.locator('#suffix')).to_be_enabled()
        expect(page.locator('#suffix-note')).to_contain_text('up to 3')
        page.locator('#suffix').check()
        page.locator('#decoding').select_option('greedy')
        page.locator('#reply-limit').select_option('128')
        page.locator('#prompt').fill('Continue this sequence for another ten repetitions. Output only the sequence: red blue red blue red blue red blue red blue')
        page.locator('#send').click()
        expect(page.locator('#suffix')).to_be_disabled()
        expect(page.locator('#prompt')).to_be_enabled(timeout=120000)
        assert not page.locator('#error').is_visible()
        accepted=page.locator('#accepted').inner_text()
        accepted_count,proposed_count=map(int,accepted.split('/'))
        assert 0<accepted_count<=proposed_count,accepted
        first=page.locator('.assistant .content').last.inner_text()
        assert first.strip()
        assert 'drafts accepted' in page.locator('.assistant .reply-meta').last.inner_text()
        page.locator('#decoding').select_option('chat')
        page.locator('#prompt').fill('What two colors did you just say? Answer briefly.')
        page.locator('#send').click()
        expect(page.locator('#prompt')).to_be_enabled(timeout=120000)
        assert not page.locator('#error').is_visible()
        reused=int(page.locator('#reused').inner_text().replace(',',''))
        assert reused>0
        page.screenshot(path=str(output/'conversation.png'))
        page.locator('#new-chat').click()
        page.locator('#reply-limit').select_option('512')
        page.locator('#prompt').fill('Repeat red blue green 100 times.')
        page.locator('#send').click()
        page.locator('#stop').click()
        expect(page.locator('#prompt')).to_be_enabled(timeout=120000)
        assert 'Stopped' in page.locator('.assistant .reply-meta').last.inner_text()
        expect(page.locator('#suffix')).to_be_enabled()
        page.locator('#new-chat').click()
        page.locator('#prompt').fill('x '*5500)
        page.locator('#send').click()
        expect(page.locator('#prompt')).to_be_enabled(timeout=120000)
        assert page.locator('#error').is_visible()
        page.locator('#new-chat').click()
        page.set_viewport_size({'width':390,'height':844})
        assert page.evaluate('document.documentElement.scrollWidth<=innerWidth')
        page.screenshot(path=str(output/'mobile.png'))
        assert not errors,errors
        report=dict(first_response=first,accepted=accepted_count,proposed=proposed_count,
                    reused_prefix_tokens=reused,sampled_followup=True,stop=True,context_error=True,
                    mobile_no_overflow=True,console_errors=errors)
        (output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report,indent=2))
        browser.close()


if __name__=='__main__': main()
