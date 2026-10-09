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
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { startSession, ROOT } from './session_support.mjs';
import { KEYMAP, keyEntry, shortcutLabel } from '../radiance_keymap.js';

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

// ── M4: one keymap ──────────────────────────────────────────────────────────

test('every menu item shows the key that runs it, and only keys that are bound', { skip }, async () => {
    const page = await viewerPage(4);
    try {
        const items = await page.evaluate(async () => {
            const out = [];
            const buttons = [...v.proMenuBar.querySelectorAll('.radiance-pro-menu-items > button')];
            for (const b of buttons) {
                b.click();
                await __sleep(10);
                for (const it of document.querySelectorAll('.radiance-pro-dropdown .radiance-pro-menu-item')) {
                    out.push({ menu: b.textContent, id: it.dataset.action || null,
                        label: it.children[0].textContent, key: it.children[1].textContent });
                }
                document.body.dispatchEvent(new MouseEvent('mousedown', { bubbles: true }));
            }
            return out;
        });
        assert.ok(items.length > 25, `only ${items.length} menu items`);
        assert.ok(items.filter((i) => i.menu === 'File').length >= 4, 'File is still thin');
        for (const it of items) {
            assert.ok(it.id && keyEntry(it.id), `"${it.menu} > ${it.label}" is not in the keymap`);
            assert.equal(it.key, shortcutLabel(it.id), `"${it.menu} > ${it.label}" shows "${it.key}"`);
        }
        // Masks, curves, OCIO and presets are reachable from the menus (M18).
        for (const id of ['panel.masks', 'panel.curves', 'panel.ocio', 'panel.presets']) {
            assert.ok(items.some((i) => i.id === id), `no menu entry for ${id}`);
        }
        // Every tooltip that ends in a key names a bound one (the timeline
        // tools omitted the Shift their keys need).
        const bound = new Set(KEYMAP.flatMap((e) => [shortcutLabel(e.id), shortcutLabel(e.id, { all: true })]).filter(Boolean));
        const titles = await page.evaluate(() => [...v.container.querySelectorAll('[title]')]
            .map((el) => el.title).filter((t) => /\(([^)]{1,14})\)\s*$/.test(t)));
        assert.ok(titles.length >= 9, `only ${titles.length} tooltips name a key`);
        for (const t of titles) {
            const key = /\(([^)]{1,14})\)\s*$/.exec(t)[1];
            assert.ok(bound.has(key), `tooltip "${t}" names "${key}", which is not a binding`);
        }
        // The shortcuts the menus show do what the menus say.
        await page.keyboard.press('2');
        assert.equal(await page.evaluate(() => v.zoom), 2, '2 did not zoom to 200%');
        await page.evaluate(() => { window.__fs = 0; v.toggleFullscreen = () => { window.__fs++; }; });
        await page.keyboard.press('Shift+F');
        assert.equal(await page.evaluate(() => window.__fs), 1, 'Shift+F did not toggle full screen');
        // Running a menu item runs the same action as its key.
        const viaMenu = await page.evaluate(async () => {
            const b = [...v.proMenuBar.querySelectorAll('.radiance-pro-menu-items > button')].find((x) => x.textContent === 'Color');
            b.click(); await __sleep(10);
            [...document.querySelectorAll('.radiance-pro-dropdown .radiance-pro-menu-item')]
                .find((x) => x.dataset.action === 'panel.masks').click();
            await __sleep(20);
            return v._referenceRightTab;
        });
        assert.equal(viaMenu, 'masks');
        assert.deepEqual(page.errors, []);
    } finally { await page.close(); }
});

test('H opens help, and help lists only bound keys with their real meaning', { skip }, async () => {
    const page = await viewerPage(4);
    try {
        await page.keyboard.press('h');
        const text = await page.evaluate(() => v.helpPanel?.textContent || '');
        assert.ok(text.length > 100, 'H did not open help');
        assert.doesNotMatch(text, /Histogram/, 'help still says H is Histogram');
        assert.doesNotMatch(text, /Prompt editor/, 'help still lists P');
        for (const e of KEYMAP.filter((x) => x.keys.length && x.section)) {
            assert.ok(text.includes(e.help || e.label), `help is missing "${e.help || e.label}"`);
        }
        await page.keyboard.press('Escape');
        assert.equal(await page.evaluate(() => !!v.helpPanel), false);
        // P is unbound: it used to hide the whole right panel on the second press.
        await page.keyboard.press('p');
        await page.keyboard.press('p');
        assert.notEqual(await page.evaluate(() => v.rightControlPanel.style.display), 'none');
        assert.deepEqual(page.errors, []);
    } finally { await page.close(); }
});

test('Alt+2 opens the Grade tab and the arrows still step frames', { skip }, async () => {
    const page = await viewerPage(12);
    try {
        await page.keyboard.press('ArrowRight');
        assert.equal(await page.evaluate(() => v.currentFrame), 1);
        await page.keyboard.press('Alt+2');
        assert.equal(await page.evaluate(() => v._referenceRightTab), 'grade');
        await page.keyboard.press('ArrowRight');
        await page.keyboard.press('ArrowRight');
        assert.equal(await page.evaluate(() => v.currentFrame), 3, 'after Alt+2 the arrows stopped stepping frames');
        // An Option shortcut as macOS sends it: the code is KeyX, the character is not "x".
        const cleared = await page.evaluate(() => {
            v.setInPoint(1); v.setOutPoint(5);
            document.body.dispatchEvent(new KeyboardEvent('keydown', { code: 'KeyX', key: '≈', altKey: true, bubbles: true }));
            return [v.inPoint, v.outPoint];
        });
        assert.deepEqual(cleared, [null, null], 'Option+X did not clear in and out');
        assert.deepEqual(page.errors, []);
    } finally { await page.close(); }
});

test('numpad digits zoom, and leave the grade alone', { skip }, async () => {
    const page = await viewerPage(4);
    try {
        await page.keyboard.press('Numpad4');
        const r = await page.evaluate(() => ({ zoom: v.zoom, offset: [...v.offset] }));
        assert.equal(r.zoom, 4);
        assert.deepEqual(r.offset, [0, 0, 0], 'a numpad digit still nudges a hidden grade offset');
        await page.keyboard.press('Numpad0');
        assert.equal(await page.evaluate(() => v._viewIsFit), true, 'Numpad 0 did not fit');
        assert.deepEqual(page.errors, []);
    } finally { await page.close(); }
});

test('Space and J/K/L work after clicking the scrubber; the arrows step frames past a focused select', { skip }, async () => {
    const page = await viewerPage(12);
    try {
        const range = await page.$('.radiance-pro-sequence-range');
        const b = await range.boundingBox();
        await page.mouse.click(b.x + b.width * 0.25, b.y + b.height / 2);
        assert.equal(await page.evaluate(() => document.activeElement === v.sequenceRange), true);
        await page.keyboard.press('Space');
        assert.equal(await page.evaluate(() => v.isPlaying), true, 'Space after the scrubber did not play');
        await page.keyboard.press('k');
        assert.equal(await page.evaluate(() => v.isPlaying), false, 'K after the scrubber did not stop');
        await page.keyboard.press('l');
        assert.equal(await page.evaluate(() => v.isPlaying), true, 'L after the scrubber did not play');
        await page.keyboard.press('k');
        // The scrubber's own arrow step moves one frame, not two.
        const before = await page.evaluate(() => v.currentFrame);
        await page.keyboard.press('ArrowRight');
        assert.equal(await page.evaluate(() => v.currentFrame), before + 1);

        // With the play button focused after a click, Space toggles once, not
        // once from the key and again from the button.
        const play = await (await page.$('.radiance-pro-sequence-controls button:nth-child(2)')).boundingBox();
        await page.mouse.click(play.x + play.width / 2, play.y + play.height / 2);
        assert.equal(await page.evaluate(() => v.isPlaying), true);
        await page.keyboard.press('Space');
        assert.equal(await page.evaluate(() => v.isPlaying), false, 'Space on the focused play button toggled twice');

        // A select keeps the frame rate; the arrows step frames.
        const fps = await page.evaluate(() => { v.setFrame(2); v._fpsSelect.focus(); return v._fpsSelect.value; });
        await page.keyboard.press('ArrowRight');
        const after = await page.evaluate(() => ({ frame: v.currentFrame, fps: v._fpsSelect.value }));
        assert.deepEqual(after, { frame: 3, fps }, 'with the fps select focused the arrows changed the frame rate');
        // A select gives the keys back once a value is picked.
        const blurred = await page.evaluate(() => {
            v._fpsSelect.focus();
            v._fpsSelect.value = '30';
            v._fpsSelect.dispatchEvent(new Event('change', { bubbles: true }));
            return document.activeElement !== v._fpsSelect;
        });
        assert.equal(blurred, true, 'the select kept focus after a pick');
        assert.deepEqual(page.errors, []);
    } finally { await page.close(); }
});

test('keys the viewer handled do not reach ComfyUI; others do', { skip }, async () => {
    const page = await viewerPage(4);
    try {
        await page.evaluate(() => {
            window.__comfy = [];
            window.addEventListener('keydown', (e) => window.__comfy.push(e.code));
        });
        await page.keyboard.press('f');
        await page.keyboard.press('KeyU');
        await page.keyboard.press('Control+z');
        const seen = await page.evaluate(() => window.__comfy);
        assert.ok(!seen.includes('KeyF'), 'F reached the page after the viewer fitted');
        assert.ok(!seen.includes('KeyZ'), 'Ctrl+Z reached the page after the viewer undid');
        assert.ok(seen.includes('KeyU'), 'an unbound key was swallowed');
        // Typing in a text field is never a shortcut.
        const typed = await page.evaluate(() => {
            const t = document.createElement('input'); t.type = 'text';
            v.container.appendChild(t); t.focus();
            const z = v.zoom;
            return { t, z };
        }).then(() => page.keyboard.type('2f'))
            .then(() => page.evaluate(() => ({ zoom: v.zoom, value: document.activeElement.value })));
        assert.equal(typed.value, '2f');
        assert.notEqual(typed.zoom, 2, 'typing "2" in a text field zoomed the viewer');
        assert.deepEqual(page.errors, []);
    } finally { await page.close(); }
});

// ── H8: the live panel reaches masks, qualifiers, scope scale, OCIO ─────────

test('the Masks tab drives the mask and qualifier the shader reads, with undo', { skip }, async () => {
    const page = await viewerPage(4);
    try {
        const r = await page.evaluate(async () => {
            const panel = v.controlsPanel;
            panel.querySelector('[data-tab-id="masks"]').click();
            await __sleep(20);
            const out = { tab: v._referenceRightTab };
            const type = panel.querySelector('select[data-radiance-param="mask_type"]');
            type.value = '1';
            type.dispatchEvent(new Event('change', { bubbles: true }));
            const cx = panel.querySelector('input[type="range"][data-radiance-param="mask_center_x"]');
            cx.dispatchEvent(new PointerEvent('pointerdown', { bubbles: true }));
            cx.value = '0.25';
            cx.dispatchEvent(new Event('input', { bubbles: true }));
            const q = panel.querySelector('[data-radiance-param="qualifier_enabled"]');
            q.click();
            const hue = panel.querySelector('input[type="range"][data-radiance-param="qualifier_hue"]');
            hue.value = '0.6';
            hue.dispatchEvent(new Event('input', { bubbles: true }));
            v.render();
            const R = v.renderer;
            out.state = { type: v.maskState.type, cx: v.maskState.center[0], qOn: v.qualifierState.enabled, h: v.qualifierState.h };
            out.renderer = { type: R.maskType, cx: R.maskCenter?.[0], qOn: !!R.qualifierEnabled, h: R.qualifier?.h };
            v.undo(); v.undo(); v.undo(); v.undo();
            out.undone = { type: v.maskState.type, cx: v.maskState.center[0], qOn: v.qualifierState.enabled };
            return out;
        });
        assert.equal(r.tab, 'masks');
        assert.deepEqual(r.state, { type: 1, cx: 0.25, qOn: true, h: 0.6 });
        assert.deepEqual(r.renderer, { type: 1, cx: 0.25, qOn: true, h: 0.6 }, 'the renderer did not get the mask and qualifier');
        assert.deepEqual(r.undone, { type: 0, cx: 0.5, qOn: false }, 'undo did not take the mask and qualifier back');
        assert.deepEqual(page.errors, []);
    } finally { await page.close(); }
});

test('the Scopes tab sets scale and levels, and the rail scope buttons pick the scope they name', { skip }, async () => {
    const page = await viewerPage(4);
    try {
        const r = await page.evaluate(async () => {
            const panel = v.controlsPanel;
            v._setReferenceTab('scopes');
            await __sleep(30);
            const out = {};
            const scale = panel.querySelector('select[data-radiance-param="scope_scale"]');
            out.scales = [...scale.options].map((o) => o.value);
            scale.value = 'percent';
            scale.dispatchEvent(new Event('change', { bubbles: true }));
            out.scale = v.scopeScale;
            const levels = panel.querySelector('select[data-radiance-param="scope_levels"]');
            levels.value = 'video';
            levels.dispatchEvent(new Event('change', { bubbles: true }));
            out.levels = v.scopeLevels;
            out.ticks = v._scopeOpts({ width: 1, height: 1 }).ticks.map((t) => t.label);
            const visible = () => [...v.controlsPanel.querySelectorAll('.radiance-ref-scope-box')]
                .filter((b) => b.offsetParent !== null).map((b) => b.dataset.scope);
            out.rail = {};
            for (const name of ['Histogram', 'Waveform', 'Vectorscope', 'Parade']) {
                const btn = [...v.proSidebar.querySelectorAll('button')].find((b) => b.textContent.startsWith(name));
                btn.click();
                await __sleep(30);
                out.rail[name] = visible();
            }
            return out;
        });
        assert.ok(r.scales.includes('cv10') && r.scales.includes('percent'), JSON.stringify(r.scales));
        assert.equal(r.scale, 'percent');
        assert.equal(r.levels, 'video');
        // Percent in video levels runs -7 to 109 (footroom and headroom).
        assert.deepEqual(r.ticks, ['-7', '0', '25', '50', '75', '100', '109'], `the scope ticks did not follow the scale: ${r.ticks}`);
        assert.deepEqual(r.rail, {
            Histogram: ['histogram'], Waveform: ['waveform'], Vectorscope: ['vectorscope'], Parade: ['parade'],
        }, 'the rail scope buttons all did the same thing');
        assert.deepEqual(page.errors, []);
    } finally { await page.close(); }
});

test('the OCIO config loader lives in the Grade tab and the Color menu opens it', { skip }, async () => {
    const page = await viewerPage(4);
    try {
        const r = await page.evaluate(async () => {
            v._runKeyAction('panel.ocio');
            await __sleep(30);
            const panel = v.controlsPanel;
            return {
                tab: v._referenceRightTab,
                load: [...panel.querySelectorAll('button')].some((b) => /LOAD CONFIG/.test(b.textContent)),
                presets: !!panel.querySelector('select[data-radiance-param="grade_preset"]'),
            };
        });
        assert.deepEqual(r, { tab: 'grade', load: true, presets: true });
        assert.deepEqual(page.errors, []);
    } finally { await page.close(); }
});

test('the timeline has no tool that only redraws itself, and Ref Wipe shows B', { skip }, async () => {
    const page = await viewerPage(24);
    try {
        const r = await page.evaluate(async () => {
            const tools = [...v.sequenceDock.querySelectorAll('.radiance-pro-timeline-tools button')];
            const out = { labels: tools.map((b) => b.textContent), titles: tools.map((b) => b.title) };
            // Place a Ref Wipe block on V2 and play into it.
            v._runKeyAction('tool.reference');
            const lane = v.sequenceTrack.querySelector('.radiance-pro-lane-v2');
            const rect = lane.getBoundingClientRect();
            lane.dispatchEvent(new MouseEvent('mousedown', { bubbles: true, clientX: rect.left + rect.width * 0.5, clientY: rect.top + 5 }));
            const seg = v.v2Segments[0];
            out.seg = seg && { type: seg.type, start: seg.startFrame, end: seg.endFrame };
            v.setFrame(Math.round((seg.startFrame + seg.endFrame) / 2));
            await __sleep(50);
            out.inBlock = { mode: v.compareMode, b: !!v.renderer._compareSrc, frame: v._refWipeFrame };
            // Slip moves B, not just the drawing.
            seg.offset = 3;
            v._refreshSequenceDock();
            out.slipped = v._refWipeFrame - v.currentFrame;
            v.setFrame(0);
            await __sleep(50);
            out.outside = { mode: v.compareMode, b: !!v.renderer._compareSrc };
            return out;
        });
        assert.ok(!r.labels.some((l) => /Blade|Adjust/.test(l)), `still offered: ${r.labels}`);
        for (const t of r.titles.filter((x) => /\(.*\)/.test(x))) assert.match(t, /Shift\+/, `a tool tooltip omits Shift: ${t}`);
        assert.equal(r.seg?.type, 'reference');
        assert.deepEqual(r.inBlock, { mode: 'wipe', b: true, frame: r.inBlock.frame });
        assert.equal(r.slipped, 3, 'slipping the block did not move B');
        assert.deepEqual(r.outside, { mode: 'none', b: false });
        const src = readFileSync(join(ROOT, 'js', 'radiance_viewer.js'), 'utf8');
        assert.doesNotMatch(src, /contrast:\s*1\.15,\s*saturation:\s*1\.08/, 'the Adjust tool hidden grade is still there');
        assert.deepEqual(page.errors, []);
    } finally { await page.close(); }
});

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
