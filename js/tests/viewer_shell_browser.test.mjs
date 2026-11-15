/**
 * The V5 shell (radiance_viewer_shell.js) in a real browser: the tasks View /
 * Compare / Check, the floating panel, Pro tools, the bottom bar, preview
 * brightness and the Input settings. Uses the session harness; skips when
 * Playwright is not installed. Set RADIANCE_SHELL_SHOTS=<dir> to also save
 * screenshots of each task and of Pro tools.
 *
 * Run: node --test js/tests/viewer_shell_browser.test.mjs
 */
import test, { after } from 'node:test';
import assert from 'node:assert/strict';
import { startSession } from './session_support.mjs';

const session = await startSession();
const skip = session ? false : 'Playwright is not installed: the viewers cannot be driven.';
after(() => session?.close());

async function inPage(fn, arg) {
    const page = await session.open();
    try {
        const out = await page.evaluate(fn, arg);
        return { out, errors: page.errors };
    } finally {
        await page.close();
    }
}

test('the header: tasks in the middle, Pro tools goes to the Full layout and back', { skip }, async () => {
    const { out, errors } = await inPage(async () => {
        const n = __make('simple'); const v = n.radianceViewer;
        n.onExecuted(__frames(3, 64, 48));
        await __until(() => v.image, 10000);
        const c = v.container;
        const tasks = [...c.querySelectorAll('.rv5-tasks button')];
        const r = { v5: c.classList.contains('radiance-v5'), labels: tasks.map((b) => b.getAttribute('aria-label')),
            pressed: tasks.map((b) => b.getAttribute('aria-pressed')),
            brand: c.querySelector('.rv5-brand-word').textContent,
            icon: !!c.querySelector('.radiance-pro-mark svg'), task: v._shell.task };
        c.querySelector('.rv5-pro-btn').click();
        r.afterPro = [v.uiMode, v.workspace];
        r.proTabs = [...c.querySelectorAll('.radiance-mode-switch button')]
            .filter((b) => getComputedStyle(b).display !== 'none').map((b) => b.textContent);
        r.panelHidden = getComputedStyle(c.querySelector('.rv5-panel')).display;
        v.setWorkspace('qc');
        tasks[0].click();
        r.afterView = [v.uiMode, v.workspace, v._shell.task];
        c.querySelector('.rv5-pro-btn').click();
        r.backToLast = v.workspace;
        __remove(n);
        return r;
    });
    assert.deepEqual(errors, []);
    assert.equal(out.v5, true);
    assert.deepEqual(out.labels, ['View', 'Compare', 'Check']);
    assert.deepEqual(out.pressed, ['true', 'false', 'false']);
    assert.equal(out.task, 'view');
    assert.match(out.brand, /^Radiance$/);
    assert.equal(out.icon, true);
    assert.deepEqual(out.afterPro, ['advanced', 'color']);
    assert.deepEqual(out.proTabs, ['Color', 'Inspect', 'QC']);
    assert.equal(out.panelHidden, 'none', 'the floating panel is a Simple-layout thing');
    assert.deepEqual(out.afterView, ['simple', 'review', 'view']);
    assert.equal(out.backToLast, 'qc', 'Pro tools goes back to the last Full workspace');
});

test('the panel follows the task; the viewer\'s own compare and review buttons live in it and still work', { skip }, async () => {
    const { out, errors } = await inPage(async () => {
        const n = __make('simple'); const v = n.radianceViewer;
        n.onExecuted(__frames(4, 64, 48));
        await __until(() => v.image, 10000);
        const c = v.container;
        const panel = c.querySelector('.rv5-panel');
        const shown = () => [...panel.querySelectorAll('.rv5-body')].findIndex((b) => getComputedStyle(b).display !== 'none');
        const r = { title0: panel.querySelector('.rv5-panel-head b').textContent, body0: shown(),
            panelOn: getComputedStyle(panel).display };
        r.reviewInView = !!panel.querySelector('.rv5-body.is-on [aria-label="Review"]') && panel.querySelector('.rv5-body.is-on').contains(v._sbOnion);
        v._sbOnion.click();
        r.onion = v.onionEnabled;
        r.tools = [...panel.querySelectorAll('[data-tool]')].map((b) => b.dataset.tool);

        c.querySelector('.rv5-tasks button[aria-label="Compare"]').click();
        r.title1 = panel.querySelector('.rv5-panel-head b').textContent;
        r.body1 = shown();
        r.cmpMode = v.compareMode;
        r.cmpButtons = [...panel.querySelectorAll('.rv5-cmp button')].map((b) => b.textContent);
        r.bInPanel = panel.contains(v._sbBLabel) && panel.contains(v._sbPin);
        v._sbCompare.get('difference').click();
        r.diff = v.compareMode;
        r.diffOpts = getComputedStyle(v._sbDiffGain).display !== 'none';

        c.querySelector('.rv5-tasks button[aria-label="Check"]').click();
        r.leftCompare = v.compareMode;
        r.body2 = shown();
        r.overlay = [v.falseColor, getComputedStyle(c.querySelector('.rv5-ovbar')).display !== 'none'];
        r.legend = getComputedStyle(c.querySelector('.rv5-legend')).display !== 'none';
        [...c.querySelectorAll('.rv5-ovbar button')].find((b) => b.textContent === 'Clipping').click();
        r.clip = [v.clippingMonitor, v.falseColor];
        r.legendAfter = getComputedStyle(c.querySelector('.rv5-legend')).display;

        c.querySelector('.rv5-tasks button[aria-label="View"]').click();
        r.overlaysOff = [v.clippingMonitor, v.falseColor, getComputedStyle(c.querySelector('.rv5-ovbar')).display];
        c.querySelector('.rv5-tasks button[aria-label="Compare"]').click();
        r.cmpRemembered = v.compareMode;

        panel.querySelector('.rv5-panel-head button').click();
        r.closed = [getComputedStyle(panel).display, n.properties.radiance_viewer_panel];
        c.querySelector('.rv5-ib[aria-label="Details panel"]').click();
        r.reopened = getComputedStyle(panel).display;
        __remove(n);
        return r;
    });
    assert.deepEqual(errors, []);
    assert.equal(out.panelOn, 'flex');
    assert.equal(out.title0, 'Details');
    assert.equal(out.body0, 0);
    assert.equal(out.reviewInView, true);
    assert.equal(out.onion, true, 'a moved control still works');
    assert.deepEqual(out.tools, ['scopes', 'inspect', 'grade', 'qc']);
    assert.equal(out.title1, 'Compare versions');
    assert.equal(out.body1, 1);
    assert.equal(out.cmpMode, 'wipe', 'Compare opens on Split');
    assert.deepEqual(out.cmpButtons, ['Split', 'Difference', 'Flip', 'B only', 'A only']);
    assert.equal(out.bInPanel, true);
    assert.equal(out.diff, 'difference');
    assert.equal(out.diffOpts, true);
    assert.equal(out.leftCompare, 'none', 'leaving Compare turns compare off');
    assert.equal(out.body2, 2);
    assert.deepEqual(out.overlay, [true, true], 'Check opens on the exposure map');
    assert.equal(out.legend, true);
    assert.deepEqual(out.clip, [true, false], 'one overlay at a time');
    assert.equal(out.legendAfter, 'none');
    assert.deepEqual(out.overlaysOff, [false, false, 'none'], 'leaving Check turns the overlays off');
    assert.equal(out.cmpRemembered, 'difference', 'Compare comes back the way it was left');
    assert.deepEqual(out.closed, ['none', false]);
    assert.equal(out.reopened, 'flex');
});

test('turning compare on another way (the X key, a menu) moves to the Compare task', { skip }, async () => {
    const { out, errors } = await inPage(async () => {
        const n = __make('simple'); const v = n.radianceViewer;
        n.onExecuted(__frames(3, 64, 48));
        await __until(() => v.image, 10000);
        v._shell.setTask('check');
        const r = { before: [v._shell.task, v.falseColor] };
        v._runKeyAction('view.compare');
        r.after = [v._shell.task, v.compareMode, v.falseColor];
        __remove(n);
        return r;
    });
    assert.deepEqual(errors, []);
    assert.deepEqual(out.before, ['check', true]);
    assert.deepEqual(out.after, ['compare', 'wipe', false], 'and the Check overlays go');
});

test('Check > Rebuilt by AI: the regions input draws over the picture and is measured', { skip }, async () => {
    const { out, errors } = await inPage(async () => {
        const n = __make('simple'); const v = n.radianceViewer;
        const msg = __frames(2, 64, 48);
        const c = v.container;
        n.onExecuted(__frames(2, 64, 48));
        await __until(() => v.image, 10000);
        const r = { without: [v.hasRegions(), getComputedStyle([...c.querySelectorAll('.rv5-ovbar button')].find((b) => b.textContent === 'Rebuilt by AI')).display] };
        for (let i = 0; i < 2; i++) msg.radiance_images.push({ filename: 'r_64x48_0.png', subfolder: '', type: 'temp', is_regions: true, frame: i });
        n.onExecuted(msg);
        await __until(() => v.image, 10000);
        r.frames = v.totalFrames;
        v._shell.setTask('check');
        await __until(() => v.regionShares(), 10000);
        v._shell.syncChecks();
        const shares = v.regionShares();
        r.shares = [Math.round(shares.reconstructed), Math.round(shares.generated * 10) / 10, Math.round(shares.estimated)];
        const item = c.querySelector('.rv5-check[data-check="ai"]');
        r.item = [item?.dataset.tone, item?.querySelector('span')?.textContent];
        const btn = [...c.querySelectorAll('.rv5-ovbar button')].find((b) => b.textContent === 'Rebuilt by AI');
        r.btnShown = getComputedStyle(btn).display !== 'none';
        btn.click();
        await __sleep(50);
        r.flags = [v.showRegions, v.falseColor];
        r.legend = [...c.querySelectorAll('.rv5-legend')].filter((l) => getComputedStyle(l).display !== 'none').map((l) => l.textContent);
        // The regions picture is on the overlay where the picture is: its right quarter is cyan.
        const ctx = v.overlayCtx;
        const px = (fx, fy) => [...ctx.getImageData(Math.round(v.panX + fx * 64 * v.zoom), Math.round(v.panY + fy * 48 * v.zoom), 1, 1).data];
        r.right = px(0.9, 0.5);
        r.mid = px(0.5, 0.75);
        v._shell.setTask('view');
        r.off = v.showRegions;
        __remove(n);
        return r;
    });
    assert.deepEqual(errors, []);
    assert.deepEqual(out.without, [false, 'none'], 'not offered without the regions input');
    assert.equal(out.frames, 2, 'regions frames are not timeline frames');
    assert.deepEqual(out.shares, [25, 12.5, 0]);
    assert.equal(out.item[0], 'info');
    assert.match(out.item[1], /^RUDRA rebuilt 37\.5% of the frame: 25\.0% reconstructed, 0\.0% estimated, 12\.5% generated by AI\.$/);
    assert.equal(out.btnShown, true);
    assert.deepEqual(out.flags, [true, false]);
    assert.deepEqual(out.legend, ['Rebuilt by AIReconstructedEstimatedGenerated by AI']);
    assert.deepEqual(out.right.slice(0, 3), [72, 202, 220]);
    assert.deepEqual(out.mid.slice(0, 3), [70, 70, 70]);
    assert.equal(out.off, false, 'leaving Check turns it off');
});

test('the bottom bar: transport, timecode, filmstrip scrubber, preview brightness labelled not saved', { skip }, async () => {
    const { out, errors } = await inPage(async () => {
        const n = __make('simple'); const v = n.radianceViewer;
        n.onExecuted(__frames(4, 64, 48));
        await __until(() => v.image && v.frameImages?.filter(Boolean).length >= 4, 10000);
        const sb = v.simpleBar;
        const c = v.container;
        const r = { shown: getComputedStyle(sb).display, kids: [...sb.children].map((k) => k.className || k.tagName) };
        v.setFrame(2);
        r.tc = sb.querySelector('.rv5-tc').textContent;
        r.frame = v._sbFrame.textContent;
        r.playIcon = !!v._sbPlay.querySelector('svg');
        r.strip = sb.querySelectorAll('.rv5-strip img').length;
        const badge = c.querySelector('.rv5-badge');
        r.badgeBefore = getComputedStyle(badge).display;
        const range = sb.querySelector('.rv5-bright input');
        range.value = '1.5'; range.dispatchEvent(new Event('input')); range.dispatchEvent(new Event('change'));
        r.ev = v.viewExposure;
        r.out = sb.querySelector('.rv5-bright output').textContent;
        r.badge = [getComputedStyle(badge).display, badge.textContent.trim()];
        badge.querySelector('button').click();
        r.after = [v.viewExposure, getComputedStyle(badge).display, range.value];
        __remove(n);
        return r;
    });
    assert.deepEqual(errors, []);
    assert.equal(out.shown, 'flex');
    assert.deepEqual(out.kids, ['rv5-transport', 'rv5-time', 'rv5-strip-wrap', 'rv5-bright', 'rv5-sr', 'rv5-zoom']);
    assert.equal(out.tc, '00:00:00:02');
    assert.equal(out.frame, 'Frame 3 of 4');
    assert.equal(out.playIcon, true);
    assert.equal(out.strip, 4, 'one preview per frame, up to 12');
    assert.equal(out.badgeBefore, 'none');
    assert.equal(out.ev, 1.5);
    assert.equal(out.out, '+1.5');
    assert.notEqual(out.badge[0], 'none');
    assert.equal(out.badge[1], 'Preview brightness +1.5 · not savedReset');
    assert.deepEqual(out.after, [0, 'none', '0']);
});

test('the Look menu names the view transforms in words and sets the viewer\'s view', { skip }, async () => {
    const { out, errors } = await inPage(async () => {
        const n = __make('simple'); const v = n.radianceViewer;
        n.onExecuted(__frames(2, 64, 48));
        await __until(() => v.image, 10000);
        const c = v.container;
        const btn = c.querySelector('.rv5-look > button');
        btn.click();
        const menu = c.querySelector('.rv5-look .rv5-menu');
        const r = { open: getComputedStyle(menu).display, names: [...menu.querySelectorAll('.rv5-mi b')].map((b) => b.textContent) };
        menu.querySelector('[data-look="srgb"]').click();
        r.mode = v.viewMode;
        r.closed = getComputedStyle(menu).display;
        r.label = btn.textContent.replace(/\s+/g, ' ').trim();
        __remove(n);
        return r;
    });
    assert.deepEqual(errors, []);
    assert.equal(out.open, 'flex');
    assert.ok(out.names.includes('Film') && out.names.includes('Plain') && out.names.includes('Automatic'), out.names.join());
    assert.equal(out.mode, 'srgb');
    assert.equal(out.closed, 'none');
    assert.equal(out.label, 'Look Plain');
});

test('Input settings in the Details panel show the node settings and write them back', { skip }, async () => {
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
        const fold = v.container.querySelector('.rv5-panel details.rv5-fold');
        fold.open = true;
        fold.dispatchEvent(new Event('toggle'));
        const r = { labels: [...fold.querySelectorAll('.rv5-field > label, .rv5-field > .rv5-flabel')].map((l) => l.textContent) };
        const sel = fold.querySelector('select');
        sel.value = 'ACEScg'; sel.dispatchEvent(new Event('change'));
        r.space = n.widgets.find((x) => x.name === 'input_space').value;
        [...fold.querySelectorAll('.rv5-seg button')].find((b) => b.textContent === 'Full (32-bit)').click();
        r.prec = n.widgets.find((x) => x.name === 'float_precision').value;
        fold.querySelector('.rv5-switch').click();
        r.brackets = n.widgets.find((x) => x.name === 'exposure_bracketing').value;
        r.switchState = fold.querySelector('.rv5-switch').getAttribute('aria-checked');
        r.seen = seen.length;
        __remove(n);
        return r;
    });
    assert.deepEqual(errors, []);
    assert.deepEqual(out.labels, ['Input space', 'Float precision', 'Playback rate', 'Exposure brackets']);
    assert.equal(out.space, 'ACEScg');
    assert.equal(out.prec, 'Full (32-bit)');
    assert.equal(out.brackets, true);
    assert.equal(out.switchState, 'true', 'redrawn from the widget');
    assert.equal(out.seen, 3, 'each change runs the widget callback');
});

test('screenshots of each task and Pro tools (only with RADIANCE_SHELL_SHOTS)', { skip: skip || !process.env.RADIANCE_SHELL_SHOTS }, async () => {
    const dir = process.env.RADIANCE_SHELL_SHOTS;
    const page = await session.open();
    try {
        for (const [w, h, tag] of [[1440, 900, ''], [640, 480, '_node']]) {
            await page.setViewportSize({ width: w, height: h });
            await page.evaluate(async ([W, H]) => {
                document.querySelectorAll('.__host').forEach((x) => x.remove());
                const n = __make('simple'); window.__n = n; const v = n.radianceViewer;
                n.__host.classList.add('__host');
                n.__host.style.cssText = `position:fixed;inset:0;width:${W}px;height:${H}px;`;
                n.onExecuted(__frames(24, 640, 360));
                await __until(() => v.image, 15000);
                v.resize?.(); v.fitToView?.(); await __sleep(500);
            }, [w, h]);
            await page.screenshot({ path: `${dir}/v5_view${tag}.png` });
            if (tag) continue;
            await page.evaluate(async () => { window.__n.radianceViewer._shell.setTask('compare'); await __sleep(400); });
            await page.screenshot({ path: `${dir}/v5_compare.png` });
            await page.evaluate(async () => { window.__n.radianceViewer._shell.setTask('check'); await __sleep(400); });
            await page.screenshot({ path: `${dir}/v5_check.png` });
            await page.evaluate(async () => { const v = window.__n.radianceViewer; v._shell.setTask('view'); v.container.querySelector('.rv5-look > button').click(); await __sleep(300); });
            await page.screenshot({ path: `${dir}/v5_look.png` });
            await page.evaluate(async () => { window.__n.radianceViewer.setWorkspace('color', { grow: false }); await __sleep(600); });
            await page.screenshot({ path: `${dir}/v5_pro_color.png` });
        }
        assert.deepEqual(page.errors, []);
    } finally {
        await page.close();
    }
});
