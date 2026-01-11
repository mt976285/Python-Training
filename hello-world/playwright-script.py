import asyncio
import random
from playwright.async_api import async_playwright
from Humanization import Humanization, HumanizationConfig

# --- Compatibility wrapper for playwright_stealth ---
# Try the async API first (stealth_async). If not available, fall back to
# a sync `stealth` entrypoint. If neither exist, provide a no-op so the
# script can still run without failing on import.
try:
    from playwright_stealth import stealth_async as _stealth_async

    async def apply_stealth(page):
        await _stealth_async(page)
except Exception:
    try:
        from playwright_stealth import stealth as _stealth_sync

        async def apply_stealth(page):
            # Some implementations of `stealth` might return a coroutine,
            # while others run synchronously and return None. Handle both.
            result = _stealth_sync(page)
            if asyncio.iscoroutine(result):
                await result
    except Exception:
        async def apply_stealth(page):
            # stealth not available — no-op to avoid breaking the script
            return None

async def run_human_simulation():
    async with async_playwright() as p:
        # 1. Launch a real Chrome instance for better stealth than Chromium
        browser = await p.chromium.launch(headless=False, channel="chrome")
        
        # 2. Configure realistic browser context
        context = await browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            viewport={'width': 1920, 'height': 1080},
            device_scale_factor=1,
        )

        # 3. Configure human-like behavior settings
        config = HumanizationConfig(
            humanize=True,
            characters_per_minute=350, # Average human typing speed
            backspace_cpm=700,          # Faster backspacing to simulate fixing typos
            stealth_mode=True
        )
        
        # Initialize humanization helper
        human = await Humanization.undetected_launch(context=context, config=config)
        page = human.page

        # 4. Apply stealth to mask 'navigator.webdriver' flags (compat)
        await apply_stealth(page)

        try:
            # Navigate with a realistic wait
            await page.goto("https://www.google.com", wait_until="domcontentloaded")
            
            # 5. Natural Mouse Movement & Click
            # Moves in curved Bezier paths rather than straight lines
            search_input = page.locator("textarea[name='q']")
            await human.move_to(search_input)
            await human.click_at(search_input)

            # 6. Humanized Typing
            # Introduces variable delays and occasional hesitations
            await human.type_at(search_input, "How to automate like a human")
            await page.keyboard.press("Enter")

            # 7. Random Scrolling (Inertia-based)
            await asyncio.sleep(random.uniform(2, 4)) # Human "reading" time
            await human.scroll_to(delta_y=random.randint(400, 800))
            
            # 8. Random Human Wait
            await human.human_wait(min_sec=3, max_sec=6)

        finally:
            await browser.close()

if __name__ == "__main__":
    asyncio.run(run_human_simulation())
