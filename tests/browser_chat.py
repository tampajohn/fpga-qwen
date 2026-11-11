"""Browser smoke test against a running localhost chat. No model/network mocks."""
import json
import os
from pathlib import Path

from playwright.sync_api import sync_playwright, expect

ROOT = Path(__file__).resolve().parents[1]


def main():
    output = ROOT / 'build/browser'
    output.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        chrome = Path('/Applications/Google Chrome.app/Contents/MacOS/Google Chrome')
        options = {'executable_path': str(chrome)} if chrome.is_file() else {}
        browser = p.chromium.launch(headless=True, args=['--disable-gpu'], **options)
        page = browser.new_page(viewport={'width': 1280, 'height': 900}, device_scale_factor=1)
        errors = []
        page.on('pageerror', lambda e: errors.append(str(e)))
        page.goto(os.environ.get('CHAT_URL', 'http://127.0.0.1:8765'))
        expect(page.locator('#prompt')).to_be_enabled()
        page.screenshot(path=str(output / 'welcome.png'))
        page.locator('#reply-limit').select_option('128')
        expect(page.locator('#workers')).to_have_value('8')
        expect(page.locator('#decoding')).to_have_value('chat')
        page.locator('#decoding').select_option('greedy')
        page.locator('#prompt').fill('Answer with the single word blue.')
        page.locator('#send').click()
        expect(page.locator('#prompt')).to_be_enabled(timeout=120000)
        first = page.locator('.assistant .content').last.inner_text()
        assert first.strip(), 'Empty first response'
        assert not page.locator('#error').is_visible(), 'First reply failed'
        first_rate = page.locator('#rate').inner_text()
        page.locator('#workers').select_option('2')
        page.locator('#prompt').fill('What word did you just say?')
        page.locator('#prompt').press('Enter')
        expect(page.locator('#prompt')).to_be_enabled(timeout=120000)
        second = page.locator('.assistant .content').last.inner_text()
        assert second.strip(), 'Empty second response'
        reused = int(page.locator('#reused').inner_text().replace(',', ''))
        assert reused > 0, 'Second turn did not reuse its prefix'
        assert '2 workers' in page.locator('.assistant .reply-meta').last.inner_text()
        page.locator('#workers').select_option('8')
        page.screenshot(path=str(output / 'conversation.png'))
        for backend in ['llama-q4', 'llama-q4', 'llama-f32', 'rtl']:
            page.locator('#backend').select_option(backend)
            expect(page.locator('#suffix')).to_be_disabled()
            page.locator('#prompt').fill('What word did you just say? Answer with that word only.')
            page.locator('#send').click()
            expect(page.locator('#backend')).to_be_disabled()
            expect(page.locator('#prompt')).to_be_enabled(timeout=120000)
            assert not page.locator('#error').is_visible(), backend
            assert 'blue' in page.locator('.assistant .content').last.inner_text().lower(), backend
        page.locator('#new-chat').click()
        page.locator('#reply-limit').select_option('512')
        page.locator('#backend').select_option('llama-q4')
        page.locator('#prompt').fill(('The color is blue. ' * 300) + 'Count from 1 to 500, one number per line.')
        page.locator('#send').click()
        page.locator('#stop').click()
        expect(page.locator('#prompt')).to_be_enabled(timeout=120000)
        assert 'Stopped' in page.locator('.assistant .reply-meta').last.inner_text()
        page.locator('#backend').select_option('rtl')

        page.locator('#new-chat').click()
        assert page.locator('#welcome').is_visible()
        page.locator('#reply-limit').select_option('512')
        page.locator('#prompt').fill('Repeat the phrase red blue green 100 times, with no explanation.')
        page.locator('#send').click()
        page.locator('#stop').click()
        expect(page.locator('#prompt')).to_be_enabled(timeout=120000)
        assert 'Stopped' in page.locator('.assistant .reply-meta').last.inner_text()
        page.locator('#new-chat').click()
        page.locator('#prompt').fill('x ' * 5500)
        page.locator('#send').click()
        expect(page.locator('#prompt')).to_be_enabled(timeout=120000)
        assert page.locator('#error').is_visible(), 'Overlong prompt did not show an error'
        assert 'Context' in page.locator('#error').inner_text()
        page.locator('#new-chat').click()
        page.set_viewport_size({'width': 390, 'height': 844})
        page.locator('#prompt').fill('Say hello in one short sentence.')
        page.locator('#send').click()
        expect(page.locator('#prompt')).to_be_enabled(timeout=120000)
        assert not page.locator('#error').is_visible(), 'Mobile reply failed'
        assert page.evaluate('() => document.documentElement.scrollWidth <= innerWidth'), 'Mobile horizontal overflow'
        page.screenshot(path=str(output / 'mobile.png'))
        assert not errors, errors
        report = dict(first_response=first, second_response=second, reused_prefix_tokens=reused,
                      first_reply_tokens_per_second=first_rate, worker_change_preserved_cache=True,
                      stop_passed=True, context_error_passed=True, mobile_passed=True,
                      backends_tested=['rtl', 'llama-q4', 'llama-f32'],
                      console_errors=errors)
        (output / 'report.json').write_text(json.dumps(report, indent=2)+'\n')
        print(json.dumps(report, indent=2))
        browser.close()


if __name__ == '__main__':
    main()
