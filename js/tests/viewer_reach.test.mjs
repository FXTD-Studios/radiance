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

// ── L4: wheel, trackpad, pen and touch ──────────────────────────────────────

test('the wheel honours line and page modes, a trackpad scroll pans and a pinch zooms', { skip }, async () => {
    const page = await viewerPage(4);
    try {
        const r = await page.evaluate(() => {
            const c = v.canvas, rect = c.getBoundingClientRect();
            const at = { clientX: rect.left + rect.width / 2, clientY: rect.top + rect.height / 2, bubbles: true, cancelable: true };
            const zoomBy = (init) => {
                v.setZoom(1);
                const z = v.zoom, px = v.panX, py = v.panY;
                c.dispatchEvent(new WheelEvent('wheel', { ...at, ...init }));
                return { ratio: v.zoom / z, dx: v.panX - px, dy: v.panY - py };
            };
            return {
                pixel: zoomBy({ deltaMode: 0, deltaY: -100, wheelDeltaY: 120 }),
                line: zoomBy({ deltaMode: 1, deltaY: -3 }),
                page: zoomBy({ deltaMode: 2, deltaY: -1 }),
                trackpad: zoomBy({ deltaMode: 0, deltaX: 6, deltaY: 14, wheelDeltaX: -18, wheelDeltaY: -42 }),
                pinch: zoomBy({ deltaMode: 0, deltaY: -10, ctrlKey: true, wheelDeltaY: 30 }),
            };
        });
        assert.ok(r.pixel.ratio > 1.05, `a wheel notch did not zoom in: ${r.pixel.ratio}`);
        assert.ok(Math.abs(r.line.ratio / r.pixel.ratio - 1) < 0.1,
            `a line-mode notch zoomed ${r.line.ratio} against ${r.pixel.ratio} for a pixel-mode one`);
        assert.ok(r.page.ratio > r.pixel.ratio, `a page-mode scroll zoomed ${r.page.ratio}`);
        assert.equal(r.trackpad.ratio, 1, 'a two-finger scroll zoomed instead of panning');
        assert.ok(r.trackpad.dx < 0 && r.trackpad.dy < 0, `a two-finger scroll did not pan: ${JSON.stringify(r.trackpad)}`);
        assert.ok(r.pinch.ratio > 1.05, `a pinch (ctrl+wheel) did not zoom: ${r.pinch.ratio}`);
        assert.deepEqual(page.errors, []);
    } finally { await page.close(); }
});

test('touch drags pan, two fingers pinch-zoom, and a pen pans like a mouse', { skip }, async () => {
    const page = await viewerPage(4);
    try {
        const r = await page.evaluate(() => {
            const c = v.canvas, rect = c.getBoundingClientRect();
            const cx = rect.left + rect.width / 2, cy = rect.top + rect.height / 2;
            const fire = (type, id, x, y, extra = {}) => c.dispatchEvent(new PointerEvent(type, {
                pointerId: id, pointerType: 'touch', isPrimary: id === 1, clientX: x, clientY: y,
                button: type === 'pointermove' ? -1 : 0, buttons: type === 'pointerup' ? 0 : 1, bubbles: true, cancelable: true, ...extra,
            }));
            const out = {};
            v.setZoom(1);
            let px = v.panX, py = v.panY;
            fire('pointerdown', 1, cx, cy);
            fire('pointermove', 1, cx + 40, cy + 25);
            fire('pointerup', 1, cx + 40, cy + 25);
            out.drag = { dx: v.panX - px, dy: v.panY - py };
            v.setZoom(1);
            const z = v.zoom;
            fire('pointerdown', 1, cx - 50, cy);
            fire('pointerdown', 2, cx + 50, cy);
            fire('pointermove', 1, cx - 100, cy);
            fire('pointermove', 2, cx + 100, cy);
            fire('pointerup', 1, cx - 100, cy);
            fire('pointerup', 2, cx + 100, cy);
            out.pinch = v.zoom / z;
            v.setZoom(1);
            px = v.panX; py = v.panY;
            const pen = { pointerType: 'pen', shiftKey: true };
            fire('pointerdown', 7, cx, cy, pen);
            fire('pointermove', 7, cx - 30, cy + 10, pen);
            fire('pointerup', 7, cx - 30, cy + 10, pen);
            out.pen = { dx: v.panX - px, dy: v.panY - py };
            out.touchAction = getComputedStyle(c).touchAction;
            return out;
        });
        const scale = await page.evaluate(() => v.canvas.width / v.canvas.getBoundingClientRect().width);
        assert.ok(Math.abs(r.drag.dx - 40 * scale) < 1 && Math.abs(r.drag.dy - 25 * scale) < 1, `a touch drag did not pan: ${JSON.stringify(r.drag)}`);
        assert.ok(Math.abs(r.pinch - 2) < 0.05, `spreading two fingers to twice the distance zoomed ${r.pinch}`);
        assert.ok(Math.abs(r.pen.dx + 30 * scale) < 1 && Math.abs(r.pen.dy - 10 * scale) < 1, `a pen Shift+drag did not pan: ${JSON.stringify(r.pen)}`);
        assert.equal(r.touchAction, 'none', 'the browser would take the touch gestures for itself');
        assert.deepEqual(page.errors, []);
    } finally { await page.close(); }
});
