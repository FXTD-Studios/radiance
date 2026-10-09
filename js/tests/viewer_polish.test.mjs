/**
 * The Advanced viewer's layout, type and accessibility, measured in a browser.
 *
 * Each of these was seen in the October 2026 review screenshots or reproduced
 * with a real viewer in headless Chromium:
 *
 *   M15  Effects toggles were empty buttons, slider labels were not linked to
 *        their sliders, focus outlines were removed, transport buttons were
 *        23x21 px, and Inspector channel rows could not be reached by keyboard.
 *   M17  Fit centred the picture in the whole canvas, under the transport bar,
 *        and the two zoom readouts disagreed (54% and 88% at the same moment).
 *   M18  10 to 11 px grey on near-black, a rail cut off with no hint, a rail
 *        highlight that never moved, "Exposure" and the gear button that only
 *        hid the panel, a "Pixel Grid" that was Grid, and an empty depth
 *        preview half the height of the Effects tab.
 *   L1   The header said v3.5 on a 4.0.0 package.
 *   L2   "A001C010", "EXR" and "32-bit (float)" were fixed text, and the proxy
 *        badge described a tone map the node no longer writes.
 *   L6   Rail tools did not show whether they were on; depth-of-field
 *        controls did nothing without a depth map and said nothing about it.
 *
 * Viewers are built the way ComfyUI builds them (sessionharness.html).
 * Skips when Playwright is unavailable.
 *
 * Run: node --test js/tests/viewer_polish.test.mjs
 */
import test, { after } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { startSession, ROOT } from './session_support.mjs';

const session = await startSession();
const skip = session ? false : 'Playwright is not installed: the viewers cannot be driven.';
after(() => session?.close());

const PACKAGE_VERSION = JSON.parse(readFileSync(join(ROOT, 'package.json'), 'utf8')).version;

/** Run `fn` in a fresh page; page errors are returned for the test to check. */
async function inPage(fn, arg) {
    const page = await session.open();
    try {
        await page.evaluate(installHelpers);
        const out = await page.evaluate(fn, arg);
        return { out, errors: page.errors, page };
    } finally {
        await page.close();
    }
}

/**
 * Page-side helpers: an Advanced viewer with a frame, and the type and
 * contrast of every piece of text inside an element.
 */
function installHelpers() {
    window.__advanced = async (W = 640, H = 360, extra = {}) => {
        const n = __make('advanced');
        await __sleep(300);
        const v = n.radianceViewer;
        n.onExecuted(__frames(1, W, H, extra));
        await __until(() => v.image && v.imageWidth === W && v.hdrData, 15000);
        await __sleep(300);
        v.resize();
        await __sleep(100);
        return { n, v };
    };

    const parse = (c) => {
        const m = /rgba?\(([^)]+)\)/.exec(c || '');
        if (!m) return null;
        const p = m[1].split(/[ ,/]+/).filter(Boolean).map(Number);
        return [p[0], p[1], p[2], p.length > 3 ? p[3] : 1];
    };
    const over = (top, under) => {
        const a = top[3];
        return [0, 1, 2].map((i) => top[i] * a + under[i] * (1 - a)).concat(1);
    };
    const lum = (c) => {
        const [r, g, b] = c.slice(0, 3).map((v) => {
            v /= 255;
            return v <= 0.04045 ? v / 12.92 : ((v + 0.055) / 1.055) ** 2.4;
        });
        return 0.2126 * r + 0.7152 * g + 0.0722 * b;
    };
    /** The colours in a CSS gradient, averaged: a gradient's mean tone. */
    const gradientMean = (img) => {
        const cols = [...img.matchAll(/rgba?\([^)]+\)/g)].map((m) => parse(m[0])).filter(Boolean);
        if (!cols.length) return null;
        const s = [0, 0, 0, 0];
        cols.forEach((c) => c.forEach((v, i) => { s[i] += v; }));
        return s.map((v) => v / cols.length);
    };
    /** The opaque colour behind an element: its backgrounds, composited down. */
    const backgroundOf = (el, fallback) => {
        const layers = [];
        for (let e = el; e && e.nodeType === 1; e = e.parentElement) {
            const cs = getComputedStyle(e);
            const bg = parse(cs.backgroundColor);
            if (bg && bg[3] > 0) layers.push(bg);
            if (bg && bg[3] >= 1) break;
            if (cs.backgroundImage && cs.backgroundImage !== 'none') {
                const g = gradientMean(cs.backgroundImage);
                if (g && g[3] > 0) layers.push(g);
                if (g && g[3] >= 0.95) break;
            }
        }
        let c = fallback;
        for (let i = layers.length - 1; i >= 0; i--) c = over(layers[i], c);
        return c;
    };
    const opacityOf = (el) => {
        let o = 1;
        for (let e = el; e && e.nodeType === 1; e = e.parentElement) o *= Number(getComputedStyle(e).opacity);
        return o;
    };
    const visible = (el) => {
        const r = el.getBoundingClientRect();
        if (r.width < 1 || r.height < 1) return false;
        for (let e = el; e && e.nodeType === 1; e = e.parentElement) {
            const cs = getComputedStyle(e);
            if (cs.display === 'none' || cs.visibility === 'hidden') return false;
        }
        return true;
    };
    const HEADING = '.radiance-pro-sidebar-title, .radiance-ref-title, .radiance-ref-tab, '
        + '.radiance-ref-wheel-label, .radiance-ref-scope-head, [role="heading"]';

    /**
     * Every visible piece of text under `root`: its size, whether it is an
     * uppercase, letter-spaced section heading, and its contrast ratio
     * against the colour behind it. Disabled controls are left out (WCAG
     * exempts inactive components), as are native option lists.
     */
    window.__textReport = (root, base = [10, 12, 16, 1]) => {
        const els = new Set();
        const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT);
        let t;
        while ((t = walker.nextNode())) {
            if (t.nodeValue.trim() && t.parentElement) els.add(t.parentElement);
        }
        root.querySelectorAll('select, input[type="text"], input[type="number"], input:not([type])').forEach((e) => els.add(e));
        const out = [];
        for (const el of els) {
            if (el.closest('option, optgroup, script, style, canvas')) continue;
            if (!visible(el)) continue;
            if (el.closest(':disabled, [aria-disabled="true"], .is-disabled')) continue;
            const cs = getComputedStyle(el);
            const text = (el.value !== undefined && el.tagName !== 'BUTTON' && el.tagName !== 'LI'
                ? (el.tagName === 'SELECT' ? el.selectedOptions[0]?.textContent : el.value) : el.textContent) || '';
            if (!text.trim()) continue;
            const size = parseFloat(cs.fontSize);
            const upper = cs.textTransform === 'uppercase' || (text === text.toUpperCase() && /[A-Z]/.test(text));
            const spaced = parseFloat(cs.letterSpacing) > 0;
            const heading = upper && spaced && !!el.closest(HEADING);
            const bg = backgroundOf(el, base);
            const fg0 = parse(cs.color);
            const fg = over([fg0[0], fg0[1], fg0[2], fg0[3] * opacityOf(el)], bg);
            const L1 = lum(fg), L2 = lum(bg);
            const ratio = (Math.max(L1, L2) + 0.05) / (Math.min(L1, L2) + 0.05);
            out.push({ text: text.trim().slice(0, 40), size, heading, ratio: Math.round(ratio * 100) / 100,
                cls: (el.className && typeof el.className === 'string' ? el.className : el.tagName).slice(0, 60) });
        }
        return out;
    };

    /** The zoom in each of the two readouts, as numbers. */
    window.__readouts = (v) => ({
        model: Math.round(v.zoom * 100),
        statusBar: Number((/ZOOM:\s*(\d+)%/.exec(v.infoRight.textContent) || [])[1]),
        bottomBar: Number((/(\d+)%/.exec(v.zoomInfo.textContent) || [])[1]),
    });

    /** The picture's rectangle on screen, in CSS pixels. */
    window.__imageBox = (v) => {
        const c = v.canvas.getBoundingClientRect();
        const sx = c.width / v.canvas.width, sy = c.height / v.canvas.height;
        return { left: c.left + v.panX * sx, top: c.top + v.panY * sy,
            right: c.left + (v.panX + v.imageWidth * v.zoom) * sx,
            bottom: c.top + (v.panY + v.imageHeight * v.zoom) * sy, canvas: c.toJSON() };
    };
    window.__rail = (v, id) => v.proSidebar.querySelector(`[data-rail-id="${id}"]`);
}

const offenders = (report, pred) => report.filter(pred).map((r) => `${r.text} [${r.cls}] ${r.size}px ${r.ratio}:1`);
const tooSmall = (r) => r.size < (r.heading ? 11 : 12);


test('rail, Inspector and every panel tab: 12 px text (11 px headings) at 4.5:1 or better', { skip }, async () => {
    const { out, errors } = await inPage(async () => {
        const { v } = await __advanced();
        const res = { rail: __textReport(v.proSidebar), header: __textReport(v.proToolbar),
            status: __textReport(v.bottomInfoBar).concat(__textReport(v.statusBar)),
            dock: __textReport(v.sequenceDock).concat(__textReport(v.viewerBar)) };
        for (const tab of ['inspector', 'grade', 'effects', 'scopes', 'analysis']) {
            v._setReferenceTab(tab);
            await __sleep(150);
            res[tab] = __textReport(v.controlsPanel);
        }
        return res;
    });
    assert.deepEqual(errors, []);
    for (const [where, report] of Object.entries(out)) {
        assert.ok(report.length > 0, `${where}: no text found`);
        if (!['header', 'status', 'dock'].includes(where)) {
            assert.deepEqual(offenders(report, tooSmall), [], `${where}: text below 12 px (11 px for uppercase headings)`);
        }
        assert.deepEqual(offenders(report, (r) => r.ratio < 4.5), [], `${where}: text below 4.5:1 contrast`);
    }
});

test('colours and sizes come from the theme tokens, and high contrast changes them', { skip }, async () => {
    const { out, errors } = await inPage(async () => {
        const { v } = await __advanced();
        const css = (name) => getComputedStyle(v.container).getPropertyValue(name).trim().toLowerCase();
        const r = {
            theme: { accent: v.theme.accent.toLowerCase(), textDim: v.theme.textDim.toLowerCase() },
            vars: { accent: css('--radiance-accent'), textDim: css('--radiance-text-dim'), fs: css('--radiance-fs-body') },
        };
        v.setHighContrast(true);
        r.hc = { textDim: css('--radiance-text-dim'), cls: v.container.classList.contains('radiance-high-contrast'),
            stored: localStorage.getItem('radiance_high_contrast') };
        v.setHighContrast(false);
        r.off = { textDim: css('--radiance-text-dim'), stored: localStorage.getItem('radiance_high_contrast') };
        return r;
    });
    assert.deepEqual(errors, []);
    assert.equal(out.vars.accent, out.theme.accent, 'the CSS accent is not the theme accent');
    assert.equal(out.vars.textDim, out.theme.textDim, 'the CSS dim text is not the theme dim text');
    assert.equal(out.vars.fs, '12px');
    assert.ok(out.hc.cls && out.hc.stored === '1', JSON.stringify(out.hc));
    assert.notEqual(out.hc.textDim, out.vars.textDim, 'high contrast leaves the dim text as it was');
    assert.equal(out.off.textDim, out.vars.textDim);
    assert.equal(out.off.stored, '0');
});

test('the rail scrolls when it does not fit, with a hint that more is below', { skip }, async () => {
    const { out, errors } = await inPage(async () => {
        const { v } = await __advanced();
        const s = v.proSidebar;
        const cs = getComputedStyle(s);
        const r = { overflowY: cs.overflowY, scrollable: s.scrollHeight > s.clientHeight + 4,
            hintTop: s.classList.contains('has-more-below') };
        s.scrollTop = s.scrollHeight;
        s.dispatchEvent(new Event('scroll'));
        await __sleep(50);
        r.hintBottom = s.classList.contains('has-more-below');
        r.mask = getComputedStyle(s).webkitMaskImage || getComputedStyle(s).maskImage;
        return r;
    });
    assert.deepEqual(errors, []);
    assert.ok(['auto', 'scroll'].includes(out.overflowY));
    assert.ok(out.scrollable, 'the test host should be short enough for the rail to scroll');
    assert.equal(out.hintTop, true, 'no hint that the rail continues below');
    assert.equal(out.hintBottom, false, 'the hint stays after scrolling to the end');
});

test('the rail highlight follows the real state, and toggles say on or off', { skip }, async () => {
    const { out, errors } = await inPage(async () => {
        const { v } = await __advanced();
        const st = (id) => {
            const b = __rail(v, id);
            return b ? { active: b.classList.contains('is-active'), pressed: b.getAttribute('aria-pressed') } : null;
        };
        const r = {};
        v.fitToView();
        r.fit = { fit: st('fit'), z100: st('zoom-100') };
        v.setZoom(1);
        r.z100 = { fit: st('fit'), z100: st('zoom-100'), z200: st('zoom-200') };
        __rail(v, 'channel-r').click();
        r.chanR = { r: st('channel-r'), rgb: st('channel-rgb'), channel: v.channel };
        // A change made anywhere else (here: the state itself) shows on the rail.
        v.channel = 'b'; v.render();
        await __sleep(30);
        r.chanB = { b: st('channel-b'), r: st('channel-r') };
        r.fcOff = st('false-color');
        __rail(v, 'false-color').click();
        r.fcOn = { st: st('false-color'), state: v.falseColor };
        __rail(v, 'false-color').click();
        r.fcOff2 = st('false-color');
        v.zebra = true; v.render();
        await __sleep(30);
        r.zebra = st('zebra');
        __rail(v, 'safe-areas').click();
        r.safe = { st: st('safe-areas'), mode: v.safeAreaMode, label: __rail(v, 'safe-areas').textContent };
        __rail(v, 'center-cross').click();
        r.cross = st('center-cross');
        // Every toggle declares itself one.
        r.toggles = ['false-color', 'zebra', 'hdr-heatmap', 'safe-areas', 'grid', 'pixel-grid', 'center-cross', 'depth']
            .map((id) => [id, __rail(v, id)?.hasAttribute('aria-pressed')]);
        return r;
    });
    assert.deepEqual(errors, []);
    assert.deepEqual(out.fit, { fit: { active: true, pressed: 'true' }, z100: { active: false, pressed: 'false' } });
    assert.equal(out.z100.fit.active, false, 'Fit stays highlighted at 100%');
    assert.equal(out.z100.z100.active, true, '100% is not highlighted at 100%');
    assert.equal(out.z100.z200.active, false);
    assert.equal(out.chanR.channel, 'r');
    assert.equal(out.chanR.r.active, true);
    assert.equal(out.chanR.rgb.active, false, 'RGB stays highlighted after picking R');
    assert.equal(out.chanB.b.active, true, 'the rail does not follow a channel change made elsewhere');
    assert.equal(out.chanB.r.active, false);
    assert.equal(out.fcOff.pressed, 'false');
    assert.deepEqual(out.fcOn, { st: { active: true, pressed: 'true' }, state: true });
    assert.equal(out.fcOff2.pressed, 'false');
    assert.equal(out.zebra.pressed, 'true', 'Zebra turned on elsewhere is not shown on the rail');
    assert.equal(out.safe.mode, 'action');
    assert.equal(out.safe.st.pressed, 'true');
    assert.match(out.safe.label, /action/i, 'the rail does not say which safe area is showing');
    assert.equal(out.cross.pressed, 'false', 'Center Cross starts on, so one click turns it off');
    for (const [id, has] of out.toggles) assert.ok(has, `${id} has no aria-pressed`);
});

test('rail Exposure opens the Grade tab on Exposure; Pixel Grid is not Grid', { skip }, async () => {
    const { out, errors } = await inPage(async () => {
        const { v } = await __advanced(64, 48);
        const r = {};
        __rail(v, 'exposure').click();
        await __sleep(100);
        const a = document.activeElement;
        r.exposure = { tab: v._referenceRightTab, focused: a?.dataset?.radianceParam || a?.tagName,
            panelShown: getComputedStyle(v.rightControlPanel).display !== 'none' };

        const grid0 = v.gridMode;
        __rail(v, 'pixel-grid').click();
        r.pixel = { on: v.pixelGrid, gridModeUnchanged: v.gridMode === grid0, pressed: __rail(v, 'pixel-grid').getAttribute('aria-pressed') };
        const inked = () => {
            v.renderOverlay();
            // One marked pixel after the redraw: Chromium can hand back the
            // previous read of a canvas that was only cleared since.
            v.overlayCtx.fillStyle = '#f00';
            v.overlayCtx.fillRect(0, 0, 1, 1);
            const d = v.overlayCtx.getImageData(0, 0, v.overlayCanvas.width, v.overlayCanvas.height).data;
            let n = -1;
            for (let i = 3; i < d.length; i += 4) if (d[i] > 0) n++;
            return n;
        };
        // What the pixel grid adds, over whatever else the overlay draws.
        const added = () => {
            const on = inked();
            v.pixelGrid = false;
            const off = inked();
            v.pixelGrid = true;
            return on - off;
        };
        // From a fitted view, so 1600% is centred on the picture.
        v.fitToView();
        v.setZoom(16);
        r.pixel.view = { zoom: v.zoom, panX: Math.round(v.panX), panY: Math.round(v.panY), w: v.overlayCanvas.width, h: v.overlayCanvas.height };
        r.pixel.atZoom16 = added();
        v.fitToView();
        r.pixel.atFit = added();
        __rail(v, 'pixel-grid').click();
        r.pixel.offPressed = __rail(v, 'pixel-grid').getAttribute('aria-pressed');
        return r;
    });
    assert.deepEqual(errors, []);
    assert.equal(out.exposure.tab, 'grade');
    assert.equal(out.exposure.focused, 'exposure', 'Exposure on the rail does not land on the Exposure slider');
    assert.equal(out.exposure.panelShown, true, 'Exposure on the rail hid the panel');
    assert.equal(out.pixel.on, true);
    assert.equal(out.pixel.gridModeUnchanged, true, 'Pixel Grid changed the composition grid');
    assert.equal(out.pixel.pressed, 'true');
    assert.ok(out.pixel.atZoom16 > 1000, `no pixel grid drawn at 1600% (${out.pixel.atZoom16} px inked, view ${JSON.stringify(out.pixel.view)})`);
    assert.equal(out.pixel.atFit, 0, 'the pixel grid draws at Fit too');
    assert.equal(out.pixel.offPressed, 'false');
});

test('the gear opens real settings; hiding the panel is its own labelled button', { skip }, async () => {
    const { out, errors } = await inPage(async () => {
        const { v } = await __advanced();
        const gear = v.proToolbar.querySelector('.radiance-pro-settings-btn');
        const panelBtn = v.proToolbar.querySelector('.radiance-pro-panel-btn');
        const r = { gear: gear && { label: gear.getAttribute('aria-label'), popup: gear.getAttribute('aria-haspopup') } };
        gear.click();
        await __sleep(50);
        const pop = document.querySelector('.radiance-settings-popover');
        r.pop = pop && { role: pop.getAttribute('role'), text: pop.textContent,
            expanded: gear.getAttribute('aria-expanded'),
            panelStill: getComputedStyle(v.rightControlPanel).display !== 'none',
            controls: [...pop.querySelectorAll('[data-setting]')].map((e) => e.dataset.setting) };
        const hc = pop?.querySelector('[data-setting="high-contrast"]');
        hc?.click();
        r.hc = { stored: localStorage.getItem('radiance_high_contrast'), cls: v.container.classList.contains('radiance-high-contrast'),
            pressed: hc?.getAttribute('aria-checked') || hc?.getAttribute('aria-pressed') };
        hc?.click();
        pop?.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape', bubbles: true }));
        await __sleep(30);
        r.closed = !document.querySelector('.radiance-settings-popover');
        r.panelBtn = panelBtn && { text: panelBtn.textContent.trim(), expanded: panelBtn.getAttribute('aria-expanded') };
        panelBtn?.click();
        await __sleep(50);
        r.afterHide = { shown: getComputedStyle(v.rightControlPanel).display !== 'none', text: panelBtn?.textContent.trim(),
            expanded: panelBtn?.getAttribute('aria-expanded') };
        panelBtn?.click();
        r.afterShow = getComputedStyle(v.rightControlPanel).display !== 'none';
        return r;
    });
    assert.deepEqual(errors, []);
    assert.deepEqual(out.gear, { label: 'Settings', popup: 'dialog' });
    assert.ok(out.pop, 'the gear did not open a settings popover');
    assert.equal(out.pop.role, 'dialog');
    assert.equal(out.pop.expanded, 'true');
    assert.equal(out.pop.panelStill, true, 'the gear still hides the panel');
    for (const s of ['backend', 'precision', 'magnify', 'high-contrast']) {
        assert.ok(out.pop.controls.includes(s), `settings has no ${s} control (has ${out.pop.controls})`);
    }
    assert.deepEqual(out.hc, { stored: '1', cls: true, pressed: 'true' });
    assert.ok(out.closed, 'Escape does not close the settings');
    assert.match(out.panelBtn.text, /hide panel/i);
    assert.equal(out.panelBtn.expanded, 'true');
    assert.deepEqual(out.afterHide, { shown: false, text: 'Show Panel', expanded: 'false' });
    assert.equal(out.afterShow, true);
});

test('Effects without a depth map: one line, and the depth controls disabled with the reason', { skip }, async () => {
    const { out, errors } = await inPage(async () => {
        const { v } = await __advanced();
        v._setReferenceTab('effects');
        await __sleep(100);
        const col = v.controlsPanel.querySelector('.radiance-ref-col');
        const preview = col.querySelector('.radiance-ref-depth-preview');
        const row = (label) => [...col.querySelectorAll('.radiance-ref-slider')]
            .find((el) => el.querySelector('label')?.textContent.trim() === label);
        const dofToggle = col.querySelector('[data-radiance-toggle="dof"]');
        return {
            previewHeight: preview ? preview.getBoundingClientRect().height : 0,
            previewText: preview?.textContent.trim() || '',
            sliders: ['Focus', 'Aperture', 'Blades', 'Angle', 'Anamorphic', 'Highlight', 'Rim', 'Cat Eye']
                .map((l) => [l, row(l)?.querySelector('input[type="range"]')?.disabled]),
            dof: dofToggle && { disabled: dofToggle.disabled, title: dofToggle.title || dofToggle.getAttribute('aria-describedby') },
            reason: row('Focus')?.title || '',
        };
    });
    assert.deepEqual(errors, []);
    assert.equal(out.previewText, 'No depth map: connect zdepth to enable depth of field');
    assert.ok(out.previewHeight < 40, `the empty depth preview is ${out.previewHeight}px tall`);
    for (const [l, disabled] of out.sliders) assert.equal(disabled, true, `${l} is live without a depth map`);
    assert.ok(out.dof?.disabled, 'Depth Of Field can be switched on without a depth map');
    assert.match(out.dof.title, /depth map/i);
    assert.match(out.reason, /depth map/i, 'a disabled slider does not say why');
});

test('Effects with a depth map: the depth controls work, and Blades skips 1 and 2', { skip }, async () => {
    const { out, errors } = await inPage(async () => {
        const { v } = await __advanced();
        const map = document.createElement('canvas');
        map.width = 64; map.height = 48;
        map.getContext('2d').fillRect(0, 0, 32, 48);
        v.zdepthImage = map;
        v._setReferenceTab('effects');
        await __sleep(100);
        const col = v.controlsPanel.querySelector('.radiance-ref-col');
        const blades = col.querySelector('input[type="range"][data-radiance-param="blades"]');
        const set = (n) => { blades.value = String(n); blades.dispatchEvent(new Event('input', { bubbles: true })); return v.apertureBlades; };
        return {
            preview: col.querySelector('.radiance-ref-depth-preview')?.classList.contains('is-empty'),
            disabled: blades.disabled,
            dof: col.querySelector('[data-radiance-toggle="dof"]').disabled,
            blades: [set(0), set(1), set(2), set(3), set(7)],
        };
    });
    assert.deepEqual(errors, []);
    assert.equal(out.preview, false);
    assert.equal(out.disabled, false);
    assert.equal(out.dof, false);
    assert.deepEqual(out.blades, [0, 0, 3, 3, 7], 'one and two blades draw the round disc');
});

test('Effects toggles are named buttons with aria-pressed, and their pills follow a click', { skip }, async () => {
    const { out, errors } = await inPage(async () => {
        const { v } = await __advanced();
        v._setReferenceTab('effects');
        await __sleep(100);
        const col = v.controlsPanel.querySelector('.radiance-ref-col');
        const name = (b) => b.getAttribute('aria-label')
            || (b.getAttribute('aria-labelledby') || '').split(/\s+/).map((id) => document.getElementById(id)?.textContent || '').join(' ').trim();
        const toggles = [...col.querySelectorAll('.radiance-ref-toggle')].map((b) => ({ name: name(b), pressed: b.getAttribute('aria-pressed') }));
        const grain = col.querySelector('[data-radiance-toggle="grain-animate"]');
        const pill = () => grain.closest('.radiance-ref-toggle-row').querySelector('.radiance-ref-status-pill')?.textContent;
        const before = { pressed: grain.getAttribute('aria-pressed'), pill: pill() };
        grain.click();
        await __sleep(30);
        const live = col.querySelector('[data-radiance-toggle="grain-animate"]') || grain;
        const after = { pressed: live.getAttribute('aria-pressed'), pill: live.closest('.radiance-ref-toggle-row').querySelector('.radiance-ref-status-pill')?.textContent };
        return { toggles, before, after };
    });
    assert.deepEqual(errors, []);
    assert.ok(out.toggles.length >= 3);
    for (const t of out.toggles) {
        assert.ok(t.name, 'a toggle has no accessible name');
        assert.ok(['true', 'false'].includes(t.pressed), `${t.name} has no aria-pressed`);
    }
    assert.deepEqual(out.before, { pressed: 'false', pill: 'STATIC' });
    assert.deepEqual(out.after, { pressed: 'true', pill: 'LIVE' });
});

test('slider labels are linked to their sliders, in Grade and Effects', { skip }, async () => {
    const { out, errors } = await inPage(async () => {
        const { v } = await __advanced();
        const res = {};
        for (const tab of ['grade', 'effects']) {
            v._setReferenceTab(tab);
            await __sleep(100);
            res[tab] = [...v.controlsPanel.querySelectorAll('.radiance-ref-slider')].map((row) => {
                const l = row.querySelector('label'), i = row.querySelector('input[type="range"]');
                return { label: l?.textContent.trim(), linked: !!l && l.control === i };
            });
        }
        return res;
    });
    assert.deepEqual(errors, []);
    for (const [tab, rows] of Object.entries(out)) {
        assert.ok(rows.length > 5, `${tab}: ${rows.length} sliders`);
        assert.deepEqual(rows.filter((r) => !r.linked).map((r) => r.label), [], `${tab}: labels not linked`);
    }
});

test('keyboard focus is visible, in the accent colour', { skip }, async () => {
    const page = await session.open();
    try {
        await page.evaluate(installHelpers);
        await page.evaluate(async () => {
            const { v } = await __advanced();
            window.__v = v;
            v._setReferenceTab('grade');
            await __sleep(100);
        });
        const targets = {
            rail: '.radiance-pro-sidebar button',
            slider: '.radiance-ref-slider input[type="range"]',
            tab: '.radiance-ref-tab',
            transport: '.radiance-pro-sequence-controls button',
            action: '.radiance-pro-actions button',
            viewerBar: '.radiance-pro-viewer-bar button',
        };
        const res = {};
        for (const [k, sel] of Object.entries(targets)) {
            // Focus after a key press, as a keyboard user's focus is.
            await page.keyboard.press('Shift');
            res[k] = await page.evaluate((s) => {
                const el = document.querySelector(s);
                el.focus();
                const cs = getComputedStyle(el);
                return { visible: el.matches(':focus-visible'), style: cs.outlineStyle, width: parseFloat(cs.outlineWidth), color: cs.outlineColor,
                    accent: getComputedStyle(window.__v.container).getPropertyValue('--radiance-accent').trim() };
            }, sel);
        }
        assert.deepEqual(page.errors, []);
        const hex = (rgb) => '#' + (/\d+,\s*\d+,\s*\d+/.exec(rgb)?.[0] || '').split(/,\s*/).map((n) => Number(n).toString(16).padStart(2, '0')).join('');
        for (const [k, r] of Object.entries(res)) {
            assert.ok(r.visible, `${k}: focus is not focus-visible`);
            assert.notEqual(r.style, 'none', `${k}: no outline on keyboard focus`);
            assert.ok(r.width >= 2, `${k}: outline ${r.width}px`);
            assert.equal(hex(r.color), r.accent.toLowerCase(), `${k}: focus outline is not the accent`);
        }
    } finally {
        await page.close();
    }
});

test('transport, track and wheel controls and the dock scrubber are at least 28 x 28 px', { skip }, async () => {
    const { out, errors } = await inPage(async () => {
        const { v } = await __advanced();
        v._setReferenceTab('grade');
        await __sleep(100);
        const box = (el) => { const r = el.getBoundingClientRect(); return [Math.round(r.width), Math.round(r.height)]; };
        const all = (sel, root = document) => [...root.querySelectorAll(sel)].filter((e) => e.offsetParent).map((e) => [sel, e.textContent.trim() || e.getAttribute('aria-label') || e.type, ...box(e)]);
        return [
            ...all('.radiance-pro-sequence-controls button', v.sequenceDock),
            ...all('.radiance-pro-track-btn', v.sequenceDock),
            ...all('.radiance-pro-tool-btn', v.sequenceDock),
            ...all('.radiance-pro-sequence-range', v.sequenceDock),
            ...all('.radiance-ref-wheel-channel input[type="range"]', v.controlsPanel),
        ];
    });
    assert.deepEqual(errors, []);
    assert.ok(out.length >= 10, `only ${out.length} controls found`);
    const small = out.filter(([sel, , w, h]) => h < 28 || (w < 28 && !/range/.test(sel)));
    assert.deepEqual(small, [], 'controls under 28 px');
});

test('Inspector channel rows can be reached and chosen by keyboard', { skip }, async () => {
    const { out, errors } = await inPage(async () => {
        const { v } = await __advanced();
        v._setReferenceTab('inspector');
        await __sleep(100);
        const rows = [...v.controlsPanel.querySelectorAll('.radiance-ref-channel')];
        const red = rows.find((r) => /^Red/.test(r.textContent.trim()));
        const r = { n: rows.length, role: red?.getAttribute('role'), tabindex: red?.getAttribute('tabindex'),
            pressed: red?.getAttribute('aria-pressed') ?? red?.getAttribute('aria-selected') };
        red.focus();
        r.focused = document.activeElement === red;
        const ev = new KeyboardEvent('keydown', { key: 'Enter', bubbles: true, cancelable: true });
        red.dispatchEvent(ev);
        await __sleep(50);
        r.channelAfterEnter = v.channel;
        r.prevented = ev.defaultPrevented;
        const green = [...v.controlsPanel.querySelectorAll('.radiance-ref-channel')].find((x) => /^Green/.test(x.textContent.trim()));
        green.dispatchEvent(new KeyboardEvent('keydown', { key: ' ', bubbles: true, cancelable: true }));
        await __sleep(50);
        r.channelAfterSpace = v.channel;
        r.playing = !!v.isPlaying;
        return r;
    });
    assert.deepEqual(errors, []);
    assert.ok(out.n >= 4);
    assert.equal(out.role, 'button');
    assert.equal(out.tabindex, '0');
    assert.ok(['true', 'false'].includes(out.pressed));
    assert.ok(out.focused);
    assert.equal(out.channelAfterEnter, 'r');
    assert.ok(out.prevented, 'Enter on a row also reaches the viewer shortcuts');
    assert.equal(out.channelAfterSpace, 'g');
    assert.equal(out.playing, false, 'Space on a channel row started playback');
});

test('Fit leaves the whole picture clear of the transport and the panels', { skip }, async () => {
    const { out, errors } = await inPage(async () => {
        const { v } = await __advanced(640, 360);
        const res = {};
        // The canvas follows its column through a ResizeObserver; under
        // software GL a rendering step can take a while to come round.
        const relaid = async (w0) => { await __until(() => v.canvas.width !== w0, 10000); await __sleep(100); };
        const check = () => {
            const img = __imageBox(v);
            const bars = [v.viewerBar, v.sequenceDock].filter((e) => e.offsetParent).map((e) => e.getBoundingClientRect().top);
            return { img, barTop: Math.min(...bars), panelLeft: v.rightControlPanel.offsetParent ? v.rightControlPanel.getBoundingClientRect().left : Infinity,
                railRight: v.proSidebar.getBoundingClientRect().right, readouts: __readouts(v) };
        };
        v.fitToView();
        res.fit = check();
        let w0 = v.canvas.width;
        v.toggleControls();
        await relaid(w0);
        res.panelHidden = check();
        w0 = v.canvas.width;
        v.toggleControls();
        await relaid(w0);
        res.panelBack = check();
        return res;
    });
    assert.deepEqual(errors, []);
    for (const [k, r] of Object.entries(out)) {
        assert.ok(r.img.bottom <= r.barTop + 0.5, `${k}: picture bottom ${r.img.bottom.toFixed(1)} under the transport at ${r.barTop.toFixed(1)}`);
        assert.ok(r.img.top >= r.img.canvas.top - 0.5, `${k}: picture top cut off`);
        assert.ok(r.img.left >= r.railRight - 0.5, `${k}: picture under the rail`);
        assert.ok(r.img.right <= Math.min(r.panelLeft, r.img.canvas.right) + 0.5, `${k}: picture under the panel`);
        assert.equal(r.readouts.statusBar, r.readouts.model, `${k}: status bar zoom`);
        assert.equal(r.readouts.bottomBar, r.readouts.model, `${k}: bottom bar zoom`);
    }
    assert.ok(out.panelHidden.readouts.model > out.fit.readouts.model, 'hiding the panel did not refit larger');
});

test('both zoom readouts agree after fit, zoom, wheel and resize', { skip }, async () => {
    const { out, errors } = await inPage(async () => {
        const { v, n } = await __advanced();
        const r = {};
        v.fitToView(); r.fit = __readouts(v);
        v.setZoom(1); r.z100 = __readouts(v);
        v.setZoom(2); r.z200 = __readouts(v);
        const c = v.canvas.getBoundingClientRect();
        v.canvas.dispatchEvent(new WheelEvent('wheel', { deltaY: -300, clientX: c.left + 50, clientY: c.top + 50, bubbles: true, cancelable: true }));
        await __sleep(50);
        r.wheel = __readouts(v);
        v.fitToView();
        const w0 = v.canvas.width;
        n.__host.style.width = '1500px';
        await __until(() => v.canvas.width !== w0, 10000);
        await __sleep(100);
        r.resize = __readouts(v);
        r.resizedTo = v.canvas.width;
        return r;
    });
    assert.deepEqual(errors, []);
    for (const [k, r] of Object.entries(out)) {
        assert.equal(r.statusBar, r.model, `${k}: status bar says ${r.statusBar}% at ${r.model}%`);
        assert.equal(r.bottomBar, r.model, `${k}: bottom bar says ${r.bottomBar}% at ${r.model}%`);
    }
    assert.equal(out.z100.model, 100);
});
