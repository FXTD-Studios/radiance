/**
 * Reach: what a user can get to, from the keys, the menus, the live panel and
 * a mouse, trackpad, pen or finger (findings H8, M4, L3, L4 and the menu half
 * of M18 in the October 2026 review).
 *
 * Whole viewers, built the way ComfyUI builds them (sessionharness.html), with
 * real key presses from Playwright where the physical key matters.
 *
 * Skips when Playwright is unavailable.
 *
 * Run: node --test js/tests/viewer_reach.test.mjs
 */
import test, { after } from 'node:test';
import assert from 'node:assert/strict';
import { startSession } from './session_support.mjs';

const session = await startSession();
const skip = session ? false : 'Playwright is not installed: the viewers cannot be driven.';
after(() => session?.close());

/** A fresh page with one Advanced viewer of `n` frames, hovered so it owns the keys. */
async function viewerPage(n = 12) {
    const page = await session.open();
    await page.evaluate(async (count) => {
        const node = __make('advanced');
        window.v = node.radianceViewer;
        window.node = node;
        node.onExecuted(__frames(count, 96, 54));
        await __until(() => v.hdrData && v._frameWindow?.isWindowReady(), 30000);
        await __sleep(200);
    }, n);
    const box = await page.evaluate(() => {
        const r = v.canvas.getBoundingClientRect();
        return { x: r.left + r.width / 2, y: r.top + r.height / 2 };
    });
    await page.mouse.move(box.x, box.y);
    await page.waitForTimeout(50);
    page.centre = box;
    return page;
}

// ── L3: the viewer bar f/ and gamma boxes ──────────────────────────────────

test('f/ and gamma clamp to their ranges and a double-click selects the text', { skip }, async () => {
    const page = await viewerPage(4);
    try {
        const ev = await page.evaluate(() => v._viewEvInput.getBoundingClientRect().toJSON());
        await page.mouse.click(ev.x + 10, ev.y + ev.height / 2);
        await page.keyboard.press('Control+a');
        await page.keyboard.type('100');
        await page.keyboard.press('Enter');
        const g = await page.evaluate(() => v._viewGammaInput.getBoundingClientRect().toJSON());
        await page.mouse.click(g.x + 10, g.y + g.height / 2);
        await page.keyboard.press('Control+a');
        await page.keyboard.type('40');
        await page.keyboard.press('Enter');
        let r = await page.evaluate(() => ({ ev: v.viewExposure, evText: v._viewEvInput.value, g: v.viewGamma, gText: v._viewGammaInput.value }));
        assert.deepEqual(r, { ev: 16, evText: '16.0', g: 4, gText: '4.00' });
        await page.mouse.click(g.x + 10, g.y + g.height / 2);
        await page.keyboard.press('Control+a');
        await page.keyboard.type('0');
        await page.keyboard.press('Enter');
        assert.equal(await page.evaluate(() => v.viewGamma), 0.1, 'gamma 0 was accepted');
        // Double-click selects, it does not reset.
        await page.mouse.dblclick(ev.x + 10, ev.y + ev.height / 2);
        r = await page.evaluate(() => {
            const i = v._viewEvInput;
            return { ev: v.viewExposure, selected: i.selectionStart === 0 && i.selectionEnd === i.value.length && i.value.length > 0 };
        });
        assert.deepEqual(r, { ev: 16, selected: true }, 'double-click reset the f-stop instead of selecting it');
        // Alt+click resets.
        await page.keyboard.down('Alt');
        await page.mouse.click(ev.x + 10, ev.y + ev.height / 2);
        await page.keyboard.up('Alt');
        assert.equal(await page.evaluate(() => v.viewExposure), 0, 'Alt+click did not reset');
        assert.deepEqual(page.errors, []);
    } finally { await page.close(); }
});
