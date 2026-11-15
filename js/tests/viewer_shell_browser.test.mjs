/**
 * The 4.1 shell (radiance_viewer_shell.js) in a real browser: the Simple /
 * Full switch, the workspace tabs, the Simple dock and the preview-exposure
 * label. Uses the session harness; skips when Playwright is not installed.
 * Set RADIANCE_SHELL_SHOTS=<dir> to also save screenshots of both modes.
 *
 * Run: node --test js/tests/viewer_shell_browser.test.mjs
 */
import test, { after } from 'node:test';
import assert from 'node:assert/strict';
import { startSession } from './session_support.mjs';

const session = await startSession();
const skip = session ? false : 'Playwright is not installed: the viewers cannot be driven.';
after(() => session?.close());

async function inPage(fn, arg, shots) {
    const page = await session.open();
    try {
        const out = await page.evaluate(fn, arg);
        if (shots && process.env.RADIANCE_SHELL_SHOTS) await shots(page);
        return { out, errors: page.errors };
    } finally {
        await page.close();
    }
}

test('Simple / Full switches layout, keeps the workspace, and Review is not a Full tab', { skip }, async () => {
    const { out, errors } = await inPage(async () => {
        const n = __make('simple'); const v = n.radianceViewer;
        n.onExecuted(__frames(3, 64, 48));
        await __until(() => v.image, 10000);
        const seg = v.container.querySelector('.rv4-mode');
        const [simpleBtn, fullBtn] = seg.querySelectorAll('button');
        const r = { v4: v.container.classList.contains('radiance-v4'), labels: [simpleBtn.textContent, fullBtn.textContent],
            simplePressed: simpleBtn.getAttribute('aria-pressed') };
        fullBtn.click();
        r.afterFull = [v.uiMode, v.workspace];
        const tabs = [...v.container.querySelectorAll('.radiance-mode-switch button')]
            .filter((b) => getComputedStyle(b).display !== 'none').map((b) => b.textContent);
        r.tabs = tabs;
        v.setWorkspace('qc');
        simpleBtn.click();
        r.afterSimple = [v.uiMode, v.workspace];
        fullBtn.click();
        r.backToLast = v.workspace;
        r.brand = v.container.querySelector('.radiance-pro-brand').textContent.replace(/\s+/g, ' ').trim();
        r.icon = !!v.container.querySelector('.radiance-pro-mark svg');
        __remove(n);
        return r;
    });
    assert.deepEqual(errors, []);
    assert.equal(out.v4, true);
    assert.deepEqual(out.labels, ['Simple', 'Full']);
    assert.equal(out.simplePressed, 'true');
    assert.deepEqual(out.afterFull, ['advanced', 'color']);
    assert.deepEqual(out.tabs, ['Color', 'Inspect', 'QC']);
    assert.deepEqual(out.afterSimple, ['simple', 'review']);
    assert.equal(out.backToLast, 'qc', 'Full goes back to the last Full workspace');
    assert.match(out.brand, /^RADIANCE VIEWER/);
    assert.equal(out.icon, true);
});

test('the Simple dock: two rows, compare, preview exposure labelled, the rest under More', { skip }, async () => {
    const { out, errors } = await inPage(async () => {
        const n = __make('simple'); const v = n.radianceViewer;
        n.onExecuted(__frames(4, 64, 48));
        await __until(() => v.image, 10000);
        const sb = v.simpleBar;
        const r = { rows: sb.querySelectorAll(':scope > .rv4-row').length,
            shown: getComputedStyle(sb).display, dir: getComputedStyle(sb).flexDirection };
        r.compare = [...sb.querySelectorAll('[aria-label="Compare"] button')].map((b) => b.textContent)
            .filter((t) => ['A', 'B', 'Wipe', 'Diff', 'Blink'].includes(t));
        const pop = sb.querySelector('.rv4-more-pop');
        r.popHidden = getComputedStyle(pop).display;
        r.inMore = pop.contains(v._sbOnion) && pop.contains(v._sbPin) && pop.contains(v._sbBFit)
            && !!pop.querySelector('[aria-label="Review"]');
        sb.querySelector('.rv4-more > button').click();
        r.popOpen = getComputedStyle(pop).display;
        v._sbOnion.click();
        r.onion = v.onionEnabled;
        document.body.dispatchEvent(new PointerEvent('pointerdown', { bubbles: true }));
        r.popClosed = getComputedStyle(pop).display;
        const badge = v.container.querySelector('.rv4-ev-badge');
        r.badgeBefore = getComputedStyle(badge).display;
        sb.querySelector('.rv4-ev button[aria-label^="Viewer f-stop up"]').click();
        r.ev = v.viewExposure;
        r.evText = sb.querySelector('.rv4-ev-value').textContent;
        r.badge = [getComputedStyle(badge).display, badge.textContent];
        v.setViewExposure(0);
        r.badgeAfter = getComputedStyle(badge).display;
        v.setFrame(2);
        r.tc = sb.querySelector('.rv4-tc').textContent;
        r.playIcon = !!v._sbPlay.querySelector('svg');
        __remove(n);
        return r;
    });
    assert.deepEqual(errors, []);
    assert.equal(out.rows, 2);
    assert.equal(out.shown, 'flex');
    assert.equal(out.dir, 'column');
    assert.deepEqual(out.compare, ['A', 'B', 'Wipe', 'Diff', 'Blink']);
    assert.equal(out.popHidden, 'none');
    assert.equal(out.inMore, true);
    assert.equal(out.popOpen, 'flex');
    assert.equal(out.onion, true, 'a control moved into More still works');
    assert.equal(out.popClosed, 'none');
    assert.equal(out.badgeBefore, 'none');
    assert.equal(out.ev, 0.5);
    assert.equal(out.evText, '+0.5 EV');
    assert.notEqual(out.badge[0], 'none', 'the label is shown');
    assert.equal(out.badge[1], 'Preview exposure +0.5 EV · not in delivery');
    assert.equal(out.badgeAfter, 'none');
    assert.equal(out.tc, '00:00:00:02');
    assert.equal(out.playIcon, true);
});

test('screenshots of both modes (only with RADIANCE_SHELL_SHOTS)', { skip: skip || !process.env.RADIANCE_SHELL_SHOTS }, async () => {
    const dir = process.env.RADIANCE_SHELL_SHOTS;
    const page = await session.open();
    try {
        await page.setViewportSize({ width: 1400, height: 900 });
        await page.evaluate(async () => {
            const n = __make('simple'); window.__n = n; const v = n.radianceViewer;
            n.__host.style.cssText = 'position:fixed;inset:0;width:1400px;height:900px;';
            n.onExecuted(__frames(24, 640, 360));
            await __until(() => v.image, 15000);
            v.resize?.(); v.fitToView?.(); await __sleep(400);
        });
        await page.screenshot({ path: `${dir}/viewer_simple.png` });
        await page.evaluate(async () => { window.__n.radianceViewer.setWorkspace('color', { grow: false }); await __sleep(600); });
        await page.screenshot({ path: `${dir}/viewer_full_color.png` });
        await page.evaluate(async () => { window.__n.radianceViewer.setWorkspace('qc'); await __sleep(400); });
        await page.screenshot({ path: `${dir}/viewer_full_qc.png` });
        assert.deepEqual(page.errors, []);
    } finally {
        await page.close();
    }
});

test('the Input menu shows the node settings, writes them back, and closes on Escape', { skip }, async () => {
    const { out, errors } = await inPage(async () => {
        const n = __make('simple'); const v = n.radianceViewer;
        const seen = [];
        const w = (name, type, value, options = {}) => ({ name, type, value, options, callback: (x) => seen.push([name, x]) });
        n.widgets = (n.widgets || []).concat([
            w('exposure_bracketing', 'toggle', false),
            w('input_space', 'combo', 'Auto', { values: ['Auto', 'sRGB (ComfyUI IMAGE)', 'ACEScg'] }),
            w('float_precision', 'combo', 'Half (16-bit)', { values: ['Half (16-bit)', 'Full (32-bit)'] }),
            w('fps', 'number', 0, { min: 0, max: 240 }),
        ]);
        v._shell.syncInput(false);
        const box = v.container.querySelector('.rv4-input');
        const btn = box.querySelector(':scope > button');
        const r = { shown: getComputedStyle(box).display !== 'none', label: btn.textContent.replace(/\s+/g, ' ').trim() };
        btn.click();
        const pop = box.querySelector('.rv4-input-pop');
        r.open = getComputedStyle(pop).display;
        r.labels = [...pop.querySelectorAll('.rv4-ip-label')].map((l) => l.textContent).concat(
            [...pop.querySelectorAll('.rv4-switch')].map((s) => s.getAttribute('aria-label')));
        const sel = pop.querySelector('select');
        sel.value = 'ACEScg'; sel.dispatchEvent(new Event('change'));
        r.space = n.widgets.find((x) => x.name === 'input_space').value;
        r.label2 = btn.textContent.replace(/\s+/g, ' ').trim();
        [...pop.querySelectorAll('.rv4-seg button')].find((b) => b.textContent === 'Full (32-bit)').click();
        r.prec = n.widgets.find((x) => x.name === 'float_precision').value;
        pop.querySelector('.rv4-switch').click();
        r.brackets = n.widgets.find((x) => x.name === 'exposure_bracketing').value;
        r.switchState = pop.querySelector('.rv4-switch').getAttribute('aria-checked');
        pop.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape', bubbles: true }));
        r.closed = getComputedStyle(pop).display;
        r.seen = seen;
        __remove(n);
        return r;
    });
    assert.deepEqual(errors, []);
    assert.equal(out.shown, true);
    assert.equal(out.label, 'Input · Auto');
    assert.equal(out.open, 'flex');
    assert.deepEqual(out.labels, ['Input space', 'Float precision', 'Playback rate', 'Exposure brackets']);
    assert.equal(out.space, 'ACEScg');
    assert.equal(out.label2, 'Input · ACEScg');
    assert.equal(out.prec, 'Full (32-bit)');
    assert.equal(out.brackets, true);
    assert.equal(out.switchState, 'true', 'the menu redraws from the widget');
    assert.equal(out.closed, 'none');
    assert.equal(out.seen.length, 3, 'each change runs the widget callback');
});
