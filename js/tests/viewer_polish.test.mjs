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
