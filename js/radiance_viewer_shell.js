/**
 * Radiance Viewer V5 shell: the picture first, three tasks in plain words.
 *
 * Implements the approved "Radiance Viewer V5 — built for the end user"
 * mockup on top of the existing viewer. Like the 4.1 shell it rearranges the
 * DOM the viewer already builds and every control keeps calling the viewer
 * method it called before, so compare, B, onion, review, QC and the report
 * behave exactly as they did.
 *
 *  - Header: Radiance icon and name, the shot, the tasks View / Compare /
 *    Check in the middle, then Look, Save frame, full screen, the details
 *    panel, Pro tools and help.
 *  - The picture fills the node. A glass panel floats over its right side
 *    instead of shrinking it; what it holds follows the task.
 *  - View: what the frame is, how bright it gets, the node's input settings,
 *    review notes and the way into the Pro tools (the Full layout).
 *  - Compare: B, how to show it (Split, Difference, Flip, B only, A only)
 *    and the difference options. Leaving Compare turns compare off.
 *  - Check: the measured QC checks and the picture's own brightness in
 *    sentences, each with a Show button for its overlay (exposure map, clipping,
 *    out of gamut). Leaving Check turns the overlays off.
 *  - Bottom bar: transport, timecode, a filmstrip scrubber, preview brightness
 *    (labelled "not saved" on the picture) and zoom.
 *
 * The pure parts (tokens, wording, check items, looks) are exported for Node
 * tests; installShell is the only function that touches the DOM.
 */
import { smpteTimecode } from "./radiance_timecode.js";
import { fullWorkspaceFor } from "./radiance_viewer_modes.js";

/**
 * V5 colours: near-black around the picture, glass panels over it, one blue
 * for state and play, amber for warnings only. Every text colour is at least
 * 4.5:1 on the surfaces it sits on.
 */
export const TOKENS = Object.freeze({
    canvas: '#08080a',
    chrome: '#0e0e11',
    raised: '#18181b',
    card: 'rgba(255,255,255,.04)',
    glass: 'rgba(18,18,22,.84)',
    control: '#27272a',
    line: 'rgba(255,255,255,.08)',
    text: '#fafafa',
    soft: '#d4d4d8',
    muted: '#a1a1aa',
    accent: '#5aa9ff',
    accentInk: '#06101f',
    warn: '#f5b544',
    warnText: '#fde7bd',
    info: '#3be0ff',
    fail: '#ff6b7a',
    pass: '#86efac',
    font: "'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif",
    mono: "'JetBrains Mono', 'SF Mono', 'Cascadia Code', Consolas, 'Liberation Mono', monospace",
});

/** Bundled fonts (SIL OFL, js/fonts). Served by ComfyUI from the extension folder; no network font request. */
export function fontFaceCSS(base = import.meta.url) {
    const u = (f) => { try { return new URL(`./fonts/${f}`, base).href; } catch { return `./fonts/${f}`; } };
    return `
@font-face { font-family: 'Inter'; font-style: normal; font-weight: 100 900; font-display: swap;
  src: url('${u('inter-latin-wght.woff2')}') format('woff2'); }
@font-face { font-family: 'JetBrains Mono'; font-style: normal; font-weight: 400; font-display: swap;
  src: url('${u('jetbrains-mono-latin-400.woff2')}') format('woff2'); }
@font-face { font-family: 'JetBrains Mono'; font-style: normal; font-weight: 500; font-display: swap;
  src: url('${u('jetbrains-mono-latin-500.woff2')}') format('woff2'); }
@font-face { font-family: 'Manrope'; font-style: normal; font-weight: 200 800; font-display: swap;
  src: url('${u('manrope-latin-wght.woff2')}') format('woff2'); }
@font-face { font-family: 'DM Mono'; font-style: normal; font-weight: 400; font-display: swap;
  src: url('${u('dm-mono-latin-400.woff2')}') format('woff2'); }`;
}

/** Preview brightness in stops as the bar shows it: 0.0, +0.5, −1.5 (a real minus sign). */
export function formatEV(stops) {
    const v = Math.round((Number(stops) || 0) * 10) / 10;
    if (v === 0) return '0.0';
    return (v > 0 ? '+' : '−') + Math.abs(v).toFixed(1);
}

/** The label on the picture while preview brightness is not zero, else null. */
export function brightnessBadge(stops) {
    const v = Math.round((Number(stops) || 0) * 10) / 10;
    return v === 0 ? null : `Preview brightness ${formatEV(v)} · not saved`;
}

/** Simple / Full, from the workspace's layout. Full is "Pro tools" in V5. */
export function shellMode(uiMode) {
    return uiMode === 'advanced' ? 'full' : 'simple';
}

/** The three tasks, in header order. */
export const TASKS = Object.freeze([
    { id: 'view', label: 'View', tip: 'Look at the picture and what it is' },
    { id: 'compare', label: 'Compare', tip: 'Put two versions against each other (X)' },
    { id: 'check', label: 'Check', tip: 'What needs a look: measured checks and overlays' },
]);

export function normalizeTask(id) {
    return TASKS.some((t) => t.id === id) ? id : 'view';
}

/**
 * The view transforms (RadianceViewer.VIEW_MODES) in words a person picks
 * from. Only the names change; the ids are the viewer's own.
 */
export const LOOKS = Object.freeze({
    auto: { name: 'Automatic', desc: 'Picks Film for HDR and float frames, Plain for ordinary images.' },
    aces2: { name: 'Film', desc: 'ACES 2.0. Smooth highlights; the usual HDR look.' },
    aces13: { name: 'Film, classic', desc: 'ACES 1.3, for shows that still use it.' },
    srgb: { name: 'Plain', desc: 'No tone map. Bright parts above white clip.' },
    rec709: { name: 'TV (Rec.709)', desc: 'For an external broadcast monitor; looks lighter here.' },
    filmic: { name: 'Filmic', desc: 'A soft filmic curve (approximate).' },
    manual: { name: 'Custom', desc: 'The Output Transform set in Pro tools › Grade.' },
});

export function lookFor(id, fallbackLabel = '') {
    return LOOKS[id] || { name: String(fallbackLabel || id || 'Automatic').replace(/^View:\s*/, ''), desc: '' };
}

/** Overlays the Check task shows, each one of the viewer's own flags. */
export const OVERLAYS = Object.freeze([
    { id: 'off', label: 'Off' },
    { id: 'exposure', label: 'Exposure map', flag: 'falseColor' },
    { id: 'clip', label: 'Clipping', flag: 'clippingMonitor' },
    { id: 'gamut', label: 'Out of gamut', flag: 'gamutWarning' },
    // Only offered when the node's regions input is connected (v.hasRegions()).
    { id: 'ai', label: 'Rebuilt by AI', flag: 'showRegions', needs: 'regions' },
]);
const OVERLAY_FLAGS = ['falseColor', 'clippingMonitor', 'gamutWarning', 'zebra', 'focusPeaking', 'showRegions'];

/**
 * The RUDRA pixel classes the regions picture colours in (model/rudra_analytic.py
 * PIXEL_CLASS_COLOUR, mixed 0.85 over the frame's grey). Preserved and "follows"
 * pixels stay grey.
 */
export const REGION_CLASSES = Object.freeze([
    { id: 'reconstructed', label: 'Reconstructed', rgb: [0.25, 0.78, 0.86] },
    { id: 'estimated', label: 'Estimated', rgb: [0.92, 0.30, 0.86] },
    { id: 'generated', label: 'Generated by AI', rgb: [0.98, 0.78, 0.20] },
]);

/**
 * Share of each class, in percent, from the regions picture's RGBA bytes.
 * A pixel is grey (not rebuilt) when its channels are within 40 of each other;
 * otherwise it takes the class whose colour points the same way.
 */
export function regionShares(rgba) {
    const n = Math.floor((rgba?.length || 0) / 4);
    const out = { reconstructed: 0, estimated: 0, generated: 0, rebuilt: 0 };
    if (!n) return out;
    const dirs = REGION_CLASSES.map((c) => { const l = Math.hypot(...c.rgb); return c.rgb.map((x) => x / l); });
    const counts = [0, 0, 0];
    for (let p = 0; p < n; p++) {
        const r = rgba[p * 4], g = rgba[p * 4 + 1], b = rgba[p * 4 + 2];
        if (Math.max(r, g, b) - Math.min(r, g, b) < 40) continue;
        const l = Math.hypot(r, g, b) || 1;
        let best = 0, bestDot = -1;
        for (let k = 0; k < 3; k++) {
            const d = (r * dirs[k][0] + g * dirs[k][1] + b * dirs[k][2]) / l;
            if (d > bestDot) { bestDot = d; best = k; }
        }
        counts[best]++;
    }
    REGION_CLASSES.forEach((c, k) => { out[c.id] = (counts[k] / n) * 100; });
    out.rebuilt = out.reconstructed + out.estimated + out.generated;
    return out;
}

/** The Rebuilt by AI overlay's key. */
export const REGION_LEGEND = Object.freeze(REGION_CLASSES.map((c) => ({
    label: c.label, c: '#' + c.rgb.map((x) => Math.round(x * 255).toString(16).padStart(2, '0')).join(''),
})));

/** Which overlay the viewer shows now. */
export function currentOverlay(v) {
    const o = OVERLAYS.find((x) => x.flag && v?.[x.flag]);
    return o ? o.id : 'off';
}

/** Show one overlay (or none). Turns the other picture overlays off, as their own buttons do. */
export function setOverlay(v, id) {
    if (!v) return;
    const o = OVERLAYS.find((x) => x.id === id) || OVERLAYS[0];
    for (const f of OVERLAY_FLAGS) v[f] = false;
    if (o.flag) v[o.flag] = true;
    v.render?.();
}

/** The exposure map's bands, as the shader draws them (radiance_webgl.js getFalseColorMap). */
export const EXPOSURE_LEGEND = Object.freeze([
    { c: '#8c00cc', label: 'Crushed' },
    { c: '#1a59ff', label: 'Near black' },
    { c: '#00d933', label: 'Mid grey' },
    { c: '#ff80bf', label: 'Skin' },
    { c: '#ffff00', label: 'Near white' },
    { c: '#ff0000', label: 'Clipped' },
]);

/** The measured QC checks (radiance_qc.js) in sentences. */
const QC_WORDS = Object.freeze({
    nan_inf: { title: 'Broken pixels', ok: 'No empty (NaN) or infinite values.',
        bad: 'Some pixels hold no number (NaN) or an infinite one. They show cyan and orange on the picture.' },
    over_range: { title: 'Above white', ok: 'Values stay within range.',
        bad: 'Parts of the frame are brighter than 1.0. Expected for HDR; a problem for an SDR delivery.' },
    under_range: { title: 'Below black', ok: 'No negative values.',
        bad: 'Some values are below zero (negative light), often from a colour conversion.' },
    alpha_range: { title: 'Alpha', ok: 'Alpha stays between 0 and 1.', bad: 'Alpha goes outside 0 to 1.' },
    alpha_premultiplied: { title: 'Premultiplied alpha', ok: 'Colour never exceeds its alpha.',
        bad: 'Colour exceeds alpha in places: the alpha may not be premultiplied.' },
    flicker: { title: 'Flicker', ok: 'Brightness is steady from frame to frame.',
        bad: 'Brightness jumps between some frames.' },
});

const TONE_OF = { PASS: 'ok', WARN: 'warn', FAIL: 'fail', NOT_RUN: 'skip', NOT_APPLICABLE: 'skip', INDETERMINATE: 'skip' };

/**
 * The Check task's list: the picture's brightness (from the viewer's zone
 * stats, always there for a float frame), what RUDRA rebuilt (when the regions
 * input is connected) and the measured QC checks (once run). Each item:
 * { id, tone: ok|info|warn|fail|skip, title, text, overlay?, frames? }.
 *   stats: v._hdrZoneStats  displayPeak: nits the chosen view shows (100 SDR)
 *   regions: regionShares() result; null = no regions input; 'loading' = still loading
 */
export function checkItems(qcReport, stats, displayPeak = 100, regions = null) {
    const items = [];
    if (regions && regions !== 'loading') {
        const f = (x) => `${x < 0.1 && x > 0 ? '<0.1' : x.toFixed(1)}%`;
        items.push(regions.rebuilt > 0
            ? { id: 'ai', tone: 'info', overlay: 'ai', title: 'Rebuilt by AI',
                text: `RUDRA rebuilt ${f(regions.rebuilt)} of the frame: ${f(regions.reconstructed)} reconstructed, `
                    + `${f(regions.estimated)} estimated, ${f(regions.generated)} generated by AI.` }
            : { id: 'ai', tone: 'ok', overlay: 'ai', title: 'Rebuilt by AI', text: 'Nothing in this frame was rebuilt; it is the source, expanded.' });
    } else if (regions === 'loading') {
        items.push({ id: 'ai', tone: 'skip', overlay: 'ai', title: 'Rebuilt by AI', text: 'Reading the regions picture\u2026' });
    }
    if (stats && Number.isFinite(stats.nitPeak)) {
        const hdrView = displayPeak >= 1000;
        const above = Number(hdrView ? stats.above1000Pct : stats.clippedPct) || 0;
        const nits = Math.round(stats.nitPeak).toLocaleString('en-US');
        items.push(above >= 0.1
            ? { id: 'bright', tone: 'warn', overlay: 'clip', title: hdrView ? `Brighter than ${displayPeak} nits` : 'Brighter than white',
                text: `${above.toFixed(1)}% of the frame is above what ${hdrView ? `a ${displayPeak}-nit HDR screen` : 'this SDR view'} shows (peak about ${nits} nits). Detail there is rolled off or lost.` }
            : { id: 'bright', tone: 'ok', overlay: 'clip', title: 'Highlights', text: `Everything fits the screen (peak about ${nits} nits).` });
        items.push({ id: 'exposure', tone: 'ok', overlay: 'exposure', title: 'Exposure',
            text: `The frame spans ${Number(stats.evRange || 0).toFixed(1)} stops. The exposure map shows where each part sits.` });
    }
    if (qcReport?.checks) {
        for (const c of qcReport.checks) {
            const w = QC_WORDS[c.id] || { title: c.id, ok: 'Passed.', bad: c.reason || 'Needs a look.' };
            const tone = TONE_OF[c.status] || 'skip';
            const frames = Array.isArray(c.frames) ? c.frames : [];
            const text = tone === 'ok' ? w.ok
                : tone === 'skip' ? (c.reason || 'Could not be measured on these frames.')
                : w.bad + (frames.length ? ` ${frames.length} frame${frames.length === 1 ? '' : 's'}.` : '');
            items.push({ id: c.id, tone, title: w.title, text, frames });
        }
    }
    const order = { fail: 0, warn: 1, info: 2, ok: 3, skip: 4 };
    return items.sort((a, b) => order[a.tone] - order[b.tone]);
}

/** How many items need a look (the number on the Check tab). */
export function issueCount(items) {
    return (items || []).filter((i) => i.tone === 'warn' || i.tone === 'fail').length;
}

/** "4K", "HD", "2K"…: a size a person recognises, else the pixels. */
export function sizeName(w, h) {
    if (!w || !h) return '—';
    if (w >= 7600) return '8K';
    if (w >= 5000) return '5K+';
    if (w >= 3800) return '4K';
    if (w >= 2500) return '2.5K';
    if (w >= 1990) return '2K';
    if (w >= 1900) return 'HD';
    if (w >= 1260) return '720p';
    return 'Small';
}

/** "10 s", "1 min 4 s", "1 frame". */
export function lengthName(frames, fps) {
    const n = Number(frames) || 0;
    if (n <= 1) return n === 1 ? 'Still' : '—';
    const s = n / (Number(fps) || 24);
    if (s < 60) return `${s < 10 ? s.toFixed(1).replace(/\.0$/, '') : Math.round(s)} s`;
    return `${Math.floor(s / 60)} min ${Math.round(s % 60)} s`;
}

/**
 * The node's own settings, shown in the Details panel instead of as ComfyUI
 * widget rows above the viewer. Names are the INPUT_TYPES keys in
 * nodes/monitor/viewer.py; the values stay in the node's widgets, so saved
 * workflows, the API format and older Radiance versions are unchanged.
 */
export const INPUT_SETTINGS = Object.freeze([
    { name: 'input_space', label: 'Input space', kind: 'combo',
      hint: 'Auto reads the pixels as linear when any value leaves 0–1, otherwise sRGB.' },
    { name: 'float_precision', label: 'Float precision', kind: 'segmented',
      hint: 'Half is what OpenEXR, RV and Nuke use for display; Full keeps every fp32 bit.' },
    { name: 'fps', label: 'Playback rate', kind: 'number', hint: '0 = the source rate (24 for an image batch).' },
    { name: 'exposure_bracketing', label: 'Exposure brackets', kind: 'switch',
      hint: 'Writes −2 / +2 EV thumbnails for the analysis shelf. Costs disk on long shots.' },
]);
export const INPUT_WIDGET_NAMES = Object.freeze(INPUT_SETTINGS.map((s) => s.name));

/** What each setting holds now: value, choices, and whether a link drives it. */
export function inputSettingsState(node) {
    const widgets = node?.widgets || [];
    return INPUT_SETTINGS.map((def) => {
        const w = widgets.find((x) => x?.name === def.name);
        const linked = node?.inputs?.find?.((i) => i?.widget?.name === def.name)?.link != null;
        const values = w?.options?.values;
        return { ...def, present: !!w, linked, value: w?.value,
            options: Array.isArray(values) ? values.slice() : null };
    });
}

/** Set one of the node's settings as its widget would. Returns false when it cannot. */
export function setInputSetting(node, name, value) {
    const w = node?.widgets?.find((x) => x?.name === name);
    if (!w) return false;
    if (node?.inputs?.find?.((i) => i?.widget?.name === name)?.link != null) return false;
    if (w.type === 'number' || name === 'fps') {
        const n = Number(value);
        if (!Number.isFinite(n)) return false;
        const lo = w.options?.min ?? -Infinity, hi = w.options?.max ?? Infinity;
        value = Math.min(hi, Math.max(lo, n));
    }
    if (Array.isArray(w.options?.values) && !w.options.values.includes(value)) return false;
    w.value = value;
    try { w.callback?.(value, undefined, node); } catch { /* the value is set either way */ }
    node.setDirtyCanvas?.(true, true);
    node.graph?.change?.();
    return true;
}

/** The input space, briefly: "sRGB (ComfyUI IMAGE)" reads "sRGB". */
export function shortSpace(v) {
    return String(v ?? 'Auto').replace(/\s*\(.*\)\s*$/, '') || 'Auto';
}

/** Inline icons (stroke, currentColor). */
export const ICONS = Object.freeze({
    prev: '<path d="M6 5v14M18 6l-9 6 9 6z"/>',
    next: '<path d="M18 5v14M6 6l9 6-9 6z"/>',
    play: '<path d="M8 5v14l11-7z" fill="currentColor" stroke="none"/>',
    pause: '<rect x="6" y="5" width="4" height="14" rx="1" fill="currentColor" stroke="none"/><rect x="14" y="5" width="4" height="14" rx="1" fill="currentColor" stroke="none"/>',
    view: '<path d="M2 12s3.5-7 10-7 10 7 10 7-3.5 7-10 7S2 12 2 12z"/><circle cx="12" cy="12" r="3"/>',
    compare: '<rect x="3" y="4" width="18" height="16" rx="2"/><path d="M12 2v20"/>',
    check: '<path d="M12 3 4 6v6c0 5 3.5 8 8 9 4.5-1 8-4 8-9V6z"/><path d="m8.5 12 2.5 2.5 4.5-5"/>',
    camera: '<path d="M4 8h3l2-3h6l2 3h3v11H4z"/><circle cx="12" cy="13" r="3.5"/>',
    fullscreen: '<path d="M4 9V4h5M20 9V4h-5M4 15v5h5M20 15v5h-5"/>',
    panel: '<rect x="3" y="4" width="18" height="16" rx="2"/><path d="M15 4v16"/>',
    pro: '<path d="M4 7h10M18 7h2M4 17h4M12 17h8"/><circle cx="16" cy="7" r="2"/><circle cx="10" cy="17" r="2"/>',
    help: '<circle cx="12" cy="12" r="9"/><path d="M9.5 9.5a2.5 2.5 0 1 1 3.5 2.3c-.6.3-1 .8-1 1.5v.7M12 17h.01"/>',
    close: '<path d="M6 6l12 12M18 6 6 18"/>',
    chevron: '<path d="m6 9 6 6 6-6"/>',
    right: '<path d="m9 6 6 6-6 6"/>',
    sun: '<circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M2 12h2M20 12h2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"/>',
    warn: '<path d="M12 3 2 20h20z"/><path d="M12 10v4M12 17h.01"/>',
    tick: '<path d="m5 12 5 5 9-10"/>',
    swap: '<path d="M7 4 3 8l4 4M3 8h14M17 20l4-4-4-4M21 16H7"/>',
});

export function svgIcon(name, size = 16) {
    return `<svg viewBox="0 0 24 24" width="${size}" height="${size}" fill="none" stroke="currentColor" `
        + `stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${ICONS[name] || ''}</svg>`;
}

/** The Radiance app icon, as on the website (#1C1C1E tile, white ring, blue square). */
export const RADIANCE_ICON_SVG = '<svg viewBox="0 0 100 100" width="26" height="26" aria-hidden="true">'
    + '<rect width="100" height="100" rx="22.5" fill="#1C1C1E"/>'
    + '<path d="M52.2 28.4A24 24 0 1 0 71.6 47.8" fill="none" stroke="#F5F5F7" stroke-width="14"/>'
    + '<rect x="58" y="28" width="14" height="14" rx="3" fill="#0A84FF"/></svg>';

/** Where "More tools" goes: a Pro tools workspace and the panel tab it opens on. */
export const MORE_TOOLS = Object.freeze([
    { id: 'scopes', title: 'Scopes', text: 'Waveform, vectorscope, histogram', workspace: 'inspect', tab: 'scopes' },
    { id: 'inspect', title: 'Channels and pixel probe', text: 'Red, green, blue or alpha alone; exact values', workspace: 'inspect', tab: 'inspector' },
    { id: 'grade', title: 'Grade', text: 'Colour, masks and effects', workspace: 'color', tab: 'grade' },
    { id: 'qc', title: 'Full QC report', text: 'Every check, every flagged frame', workspace: 'qc', tab: 'qc' },
]);

/** Shell styles. Scoped to .radiance-v5 so nothing leaks into ComfyUI. */
export function shellCSS() {
    const T = TOKENS;
    return `
.radiance-v5 { --rv5-accent: ${T.accent}; --rv5-line: ${T.line}; font-family: ${T.font}; }
.radiance-v5 button, .radiance-v5 select, .radiance-v5 input { font-family: inherit; }
.radiance-v5 .rv5-glass { background: ${T.glass}; backdrop-filter: blur(22px) saturate(140%); -webkit-backdrop-filter: blur(22px) saturate(140%);
  border: 1px solid rgba(255,255,255,.09); box-shadow: 0 20px 60px rgba(0,0,0,.55); }
.radiance-v5 .rv5-ib { width: 36px; height: 36px; flex-shrink: 0; border-radius: 10px; border: 0; background: transparent; color: ${T.muted};
  display: inline-flex; align-items: center; justify-content: center; cursor: pointer; padding: 0; }
.radiance-v5 .rv5-ib:hover { background: rgba(255,255,255,.07); color: ${T.text}; }
.radiance-v5 .rv5-ib[aria-pressed="true"] { background: rgba(255,255,255,.08); color: ${T.text}; }
.radiance-v5 button:focus-visible, .radiance-v5 select:focus-visible, .radiance-v5 input:focus-visible { outline: 2px solid #9ccaff; outline-offset: 2px; }
.radiance-v5 .rv5-btn { height: 36px; padding: 0 12px; border-radius: 10px; border: 1px solid rgba(255,255,255,.10); background: transparent;
  color: ${T.text}; font: 500 13px ${T.font}; display: inline-flex; align-items: center; gap: 8px; cursor: pointer; white-space: nowrap; flex-shrink: 0; }
.radiance-v5 .rv5-btn:hover { background: rgba(255,255,255,.06); }
.radiance-v5 .rv5-btn[aria-expanded="true"] { background: rgba(255,255,255,.08); border-color: rgba(255,255,255,.22); }
.radiance-v5 .rv5-sec { font: 600 11px ${T.font}; letter-spacing: .08em; text-transform: uppercase; color: ${T.muted}; }
.radiance-v5 .rv5-mono { font-family: ${T.mono}; }
/* Header. */
.radiance-v5 .radiance-pro-menu { flex: 0 0 56px; gap: 12px; padding: 0 12px 0 16px; background: ${T.chrome};
  border-bottom: 1px solid var(--rv5-line); overflow: visible; position: relative; z-index: 40; }
.radiance-v5 .radiance-pro-brand { min-width: 0; gap: 10px; font: 600 15px ${T.font}; letter-spacing: -.01em; color: ${T.text}; }
.radiance-v5 .radiance-pro-mark.rv5-icon { width: 26px; height: 26px; display: block; gap: 0; }
.radiance-v5 .radiance-pro-mark.rv5-icon svg { display: block; }
.radiance-v5 .radiance-pro-version { display: none; }
.radiance-v5 .rv5-divider { width: 1px; height: 24px; background: rgba(255,255,255,.10); flex-shrink: 0; }
.radiance-v5 .rv5-shot { display: flex; align-items: center; gap: 8px; min-width: 0; }
.radiance-v5 .rv5-shot-name { font: 600 14px ${T.font}; color: ${T.text}; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; max-width: 240px; }
.radiance-v5 .rv5-chip { height: 24px; padding: 0 8px; border-radius: 7px; font: 500 11px ${T.mono}; color: ${T.soft};
  background: rgba(255,255,255,.05); border: 1px solid rgba(255,255,255,.08); display: inline-flex; align-items: center; white-space: nowrap; }
.radiance-v5 .rv5-hdr { height: 22px; padding: 0 8px; border-radius: 6px; font: 600 11px ${T.font}; color: #f7c873;
  background: rgba(245,181,68,.14); display: none; align-items: center; }
.radiance-v5 .rv5-hdr.is-on { display: inline-flex; }
.radiance-v5 .rv5-tasks { display: flex; flex-shrink: 0; gap: 4px; padding: 4px;
  border-radius: 12px; background: rgba(255,255,255,.04); border: 1px solid rgba(255,255,255,.07); }
.radiance-v5 .rv5-tasks button { height: 34px; padding: 0 16px; border: 0; border-radius: 9px; background: transparent; color: ${T.muted};
  font: 600 13px ${T.font}; display: inline-flex; align-items: center; gap: 8px; cursor: pointer; white-space: nowrap; }
.radiance-v5 .rv5-tasks button:hover { color: ${T.text}; }
.radiance-v5 .rv5-tasks button.is-active { background: ${T.control}; color: ${T.text};
  box-shadow: 0 1px 0 rgba(255,255,255,.06) inset, 0 2px 8px rgba(0,0,0,.4); }
.radiance-v5 .rv5-count { min-width: 18px; height: 18px; padding: 0 5px; box-sizing: border-box; border-radius: 9px; background: ${T.warn};
  color: #1a1205; font: 700 11px ${T.font}; display: none; align-items: center; justify-content: center; }
.radiance-v5 .rv5-count.is-on { display: inline-flex; }
.radiance-v5 .rv5-spacer { flex: 1 1 auto; min-width: 8px; }
.radiance-v5 .rv5-right { display: flex; align-items: center; gap: 6px; flex-shrink: 0; }
.radiance-v5 .rv5-look { position: relative; }
.radiance-v5 .rv5-look-dot { width: 10px; height: 10px; border-radius: 5px; background: var(--rv5-accent); }
.radiance-v5 .rv5-look-key { color: ${T.muted}; }
.radiance-v5 .rv5-menu { display: none; position: absolute; top: 44px; right: 0; z-index: 80; width: 340px; box-sizing: border-box; padding: 8px;
  border-radius: 16px; flex-direction: column; gap: 2px; color: ${T.text}; }
.radiance-v5 .rv5-menu.is-open { display: flex; }
.radiance-v5 .rv5-menu .rv5-sec { padding: 8px 10px 4px; }
.radiance-v5 .rv5-mi { width: 100%; text-align: left; border: 0; background: transparent; color: ${T.text}; border-radius: 10px; padding: 9px 10px;
  display: flex; gap: 10px; align-items: flex-start; cursor: pointer; }
.radiance-v5 .rv5-mi:hover { background: rgba(255,255,255,.06); }
.radiance-v5 .rv5-mi .rv5-mi-dot { width: 10px; height: 10px; margin-top: 4px; flex-shrink: 0; border-radius: 5px; background: #52525b; }
.radiance-v5 .rv5-mi[aria-checked="true"] .rv5-mi-dot { background: var(--rv5-accent); }
.radiance-v5 .rv5-mi-txt { flex: 1; display: flex; flex-direction: column; gap: 2px; }
.radiance-v5 .rv5-mi-txt b { font: 600 13px ${T.font}; }
.radiance-v5 .rv5-mi-txt span { font: 12px/1.4 ${T.font}; color: ${T.muted}; }
.radiance-v5 .rv5-mi .rv5-mi-tick { color: var(--rv5-accent); visibility: hidden; }
.radiance-v5 .rv5-mi[aria-checked="true"] .rv5-mi-tick { visibility: visible; }
.radiance-v5 .rv5-menu-note { margin: 6px 4px 2px; padding: 10px; border-radius: 10px; background: rgba(255,255,255,.04);
  font: 12px/1.45 ${T.font}; color: ${T.muted}; }
.radiance-v5 .rv5-split { display: inline-flex; flex-shrink: 0; }
.radiance-v5 .rv5-split > .rv5-btn:first-child { border-top-right-radius: 0; border-bottom-right-radius: 0; }
.radiance-v5 .rv5-split > .rv5-btn:last-child { border-top-left-radius: 0; border-bottom-left-radius: 0; border-left: 0; padding: 0 8px; }
/* Pro tools (the Full layout): the tasks sit inline beside the workspace tabs. */
.radiance-v5:not(.radiance-mode-simple) .rv5-tasks button.is-active { background: transparent; box-shadow: none; color: ${T.muted}; }
.radiance-v5:not(.radiance-mode-simple) .rv5-pro-btn { background: rgba(90,169,255,.14); border-color: rgba(90,169,255,.4); color: ${T.text}; }
.radiance-v5 .rv5-simple-only { display: none !important; }
.radiance-v5.radiance-mode-simple .rv5-simple-only { display: flex !important; }
.radiance-v5 .radiance-mode-switch { align-self: stretch; gap: 2px; padding: 0; border: 0; border-radius: 0; background: transparent; }
.radiance-v5 .radiance-mode-switch button { height: 100%; padding: 0 12px; border-radius: 0; font: 600 13px ${T.font};
  color: ${T.muted}; background: transparent; box-shadow: inset 0 -2px 0 transparent; }
.radiance-v5 .radiance-mode-switch button:hover { color: ${T.text}; }
.radiance-v5 .radiance-mode-switch button.is-active { background: transparent; color: ${T.text}; box-shadow: inset 0 -2px 0 var(--rv5-accent); }
.radiance-v5 .radiance-mode-switch button[data-rv5-hidden] { display: none; }
.radiance-v5.radiance-mode-simple .radiance-mode-switch { display: none; }
.radiance-v5 .radiance-pro-menu-items { gap: 2px; color: ${T.muted}; font: 500 12px ${T.font}; }
/* The stage: the picture and what floats over it. */
.radiance-v5.radiance-mode-simple .radiance-pro-view-column, .radiance-v5.radiance-mode-simple .radiance-pro-view-column > div { background: #000; }
.radiance-v5 .rv5-stage-ui { display: none; }
.radiance-v5.radiance-mode-simple .rv5-stage-ui { display: flex; }
.radiance-v5 .rv5-panel { position: absolute; top: 16px; right: 16px; bottom: 16px; width: 336px; max-width: calc(100% - 32px); box-sizing: border-box;
  border-radius: 18px; flex-direction: column; overflow: hidden; z-index: 35; color: ${T.text}; font: 13px ${T.font}; }
.radiance-v5.rv5-panel-off .rv5-panel { display: none; }
.radiance-v5 .rv5-panel-head { height: 52px; flex-shrink: 0; display: flex; align-items: center; padding: 0 8px 0 18px; border-bottom: 1px solid rgba(255,255,255,.07); }
.radiance-v5 .rv5-panel-head b { flex: 1; font: 600 14px ${T.font}; }
.radiance-v5 .rv5-body { flex: 1; overflow: auto; padding: 16px 18px; display: none; flex-direction: column; gap: 20px; }
.radiance-v5 .rv5-body.is-on { display: flex; }
.radiance-v5 .rv5-tiles { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 8px; }
.radiance-v5 .rv5-tile { padding: 12px; border-radius: 12px; background: ${T.card}; min-width: 0; }
.radiance-v5 .rv5-tile small { display: block; font-size: 11px; color: ${T.muted}; }
.radiance-v5 .rv5-tile b { display: block; font: 600 15px ${T.font}; margin-top: 4px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.radiance-v5 .rv5-tile span { display: block; font: 11px ${T.mono}; color: ${T.muted}; margin-top: 2px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.radiance-v5 .rv5-group { display: flex; flex-direction: column; gap: 10px; }
.radiance-v5 .rv5-stats { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 8px; }
.radiance-v5 .rv5-stat { padding: 10px; border-radius: 10px; background: ${T.card}; }
.radiance-v5 .rv5-stat small { display: block; font-size: 11px; color: ${T.muted}; }
.radiance-v5 .rv5-stat b { display: block; font: 600 14px ${T.font}; margin-top: 3px; }
.radiance-v5 .rv5-note { font: 12px/1.45 ${T.font}; color: ${T.muted}; }
.radiance-v5 .rv5-callout { padding: 12px 14px; border-radius: 12px; background: rgba(245,181,68,.10); border: 1px solid rgba(245,181,68,.28);
  display: none; gap: 10px; color: ${T.warnText}; }
.radiance-v5 .rv5-callout.is-on { display: flex; }
.radiance-v5 .rv5-callout svg { flex-shrink: 0; margin-top: 1px; color: #f7c873; }
.radiance-v5 .rv5-callout div { display: flex; flex-direction: column; gap: 8px; font: 13px/1.45 ${T.font}; }
.radiance-v5 .rv5-soft-btn { align-self: flex-start; height: 28px; padding: 0 10px; border-radius: 8px; border: 1px solid rgba(255,255,255,.12);
  background: transparent; color: ${T.text}; font: 600 12px ${T.font}; cursor: pointer; white-space: nowrap; }
.radiance-v5 .rv5-soft-btn:hover { background: rgba(255,255,255,.08); }
.radiance-v5 .rv5-soft-btn[aria-pressed="true"] { background: rgba(255,255,255,.12); border-color: rgba(255,255,255,.22); }
.radiance-v5 .rv5-callout .rv5-soft-btn { border: 0; background: rgba(245,181,68,.22); color: ${T.warnText}; }
.radiance-v5 .rv5-primary { height: 38px; padding: 0 14px; border-radius: 10px; border: 0; background: var(--rv5-accent); color: ${T.accentInk};
  font: 700 13px ${T.font}; cursor: pointer; }
.radiance-v5 .rv5-primary:hover { filter: brightness(1.08); }
.radiance-v5 .rv5-row { height: 44px; padding: 0 10px; border: 0; border-radius: 10px; background: transparent; color: ${T.text};
  display: flex; align-items: center; gap: 12px; text-align: left; cursor: pointer; width: 100%; }
.radiance-v5 .rv5-row:hover { background: rgba(255,255,255,.05); }
.radiance-v5 .rv5-row span { flex: 1; display: flex; flex-direction: column; min-width: 0; }
.radiance-v5 .rv5-row b { font: 500 13px ${T.font}; }
.radiance-v5 .rv5-row small { font-size: 11px; color: ${T.muted}; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.radiance-v5 .rv5-row svg { color: ${T.muted}; flex-shrink: 0; }
.radiance-v5 details.rv5-fold > summary { list-style: none; cursor: pointer; display: flex; align-items: center; justify-content: space-between; }
.radiance-v5 details.rv5-fold > summary::-webkit-details-marker { display: none; }
.radiance-v5 details.rv5-fold > summary svg { transition: transform .15s; color: ${T.muted}; }
.radiance-v5 details.rv5-fold[open] > summary svg { transform: rotate(180deg); }
.radiance-v5 .rv5-fold-body { display: flex; flex-direction: column; gap: 14px; padding-top: 12px; }
/* Buttons the viewer built, moved into the panel. */
.radiance-v5 .rv5-panel .rsb-group { display: flex; flex-wrap: wrap; gap: 6px; padding: 0; border: 0; background: transparent; }
.radiance-v5 .rv5-panel .rsb-group button, .radiance-v5 .rv5-panel .rv5-moved, .radiance-v5 .rv5-panel .rv5-cmp button { height: 34px; padding: 0 12px; border-radius: 10px;
  border: 1px solid rgba(255,255,255,.10); background: transparent; color: ${T.soft}; font: 600 12px ${T.font}; cursor: pointer; }
.radiance-v5 .rv5-panel .rsb-group button:hover:not(:disabled), .radiance-v5 .rv5-panel .rv5-moved:hover:not(:disabled),
.radiance-v5 .rv5-panel .rv5-cmp button:hover:not(:disabled) { color: ${T.text}; background: rgba(255,255,255,.06); }
.radiance-v5 .rv5-panel .rsb-group button.is-active, .radiance-v5 .rv5-panel .rv5-moved.is-active, .radiance-v5 .rv5-panel .rv5-cmp button.is-active { background: rgba(255,255,255,.10);
  color: ${T.text}; border-color: var(--rv5-accent); }
.radiance-v5 .rv5-panel button:disabled { opacity: .4; cursor: default; }
.radiance-v5 .rv5-panel .rv5-cmp { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 6px; }
.radiance-v5 .rv5-panel .rv5-cmp button { height: 40px; }
.radiance-v5 .rv5-panel .rsb-b { width: 100%; height: 40px; padding: 0 12px; border-radius: 10px; border: 1px solid rgba(255,255,255,.10);
  font: 500 13px ${T.font}; cursor: pointer; text-align: left; color: ${T.text}; background: rgba(255,255,255,.05);
  white-space: nowrap; overflow: hidden; text-overflow: ellipsis; max-width: none; }
.radiance-v5 .rv5-panel .rsb-qc { display: none; }
.radiance-v5 .rv5-checks { display: flex; flex-direction: column; gap: 4px; margin: 0 -6px; }
.radiance-v5 .rv5-check { padding: 12px; border-radius: 12px; display: flex; gap: 12px; align-items: flex-start; }
.radiance-v5 .rv5-check[data-tone="warn"] { background: rgba(245,181,68,.10); }
.radiance-v5 .rv5-check[data-tone="fail"] { background: rgba(255,107,122,.10); }
.radiance-v5 .rv5-check i { width: 24px; height: 24px; flex-shrink: 0; border-radius: 12px; display: flex; align-items: center; justify-content: center;
  font: 700 13px ${T.font}; font-style: normal; background: rgba(74,222,128,.15); color: ${T.pass}; }
.radiance-v5 .rv5-check[data-tone="warn"] i { background: rgba(245,181,68,.22); color: #f7c873; }
.radiance-v5 .rv5-check[data-tone="fail"] i { background: rgba(255,107,122,.2); color: ${T.fail}; }
.radiance-v5 .rv5-check[data-tone="skip"] i { background: rgba(255,255,255,.08); color: ${T.muted}; }
.radiance-v5 .rv5-check > div { flex: 1; min-width: 0; display: flex; flex-direction: column; gap: 3px; }
.radiance-v5 .rv5-check b { font: 600 13px ${T.font}; }
.radiance-v5 .rv5-check span { font: 12px/1.45 ${T.font}; color: ${T.muted}; }
/* Over the picture. */
.radiance-v5 .rv5-float { position: absolute; z-index: 34; align-items: center; color: ${T.text}; font: 12px ${T.font}; }
.radiance-v5 .rv5-hints { left: 16px; bottom: 16px; height: 34px; padding: 0 12px; border-radius: 10px; gap: 14px; color: ${T.muted}; pointer-events: none; }
.radiance-v5 .rv5-hints kbd { font: 11px ${T.mono}; color: ${T.text}; padding: 1px 6px; border-radius: 4px; background: rgba(255,255,255,.08); margin-right: 4px; }
.radiance-v5 .rv5-hints.is-off { display: none; }
.radiance-v5 .rv5-ovbar { top: 16px; left: 50%; transform: translateX(-50%); gap: 4px; padding: 4px; border-radius: 12px; }
.radiance-v5 .rv5-ovbar button { height: 32px; padding: 0 12px; border: 0; border-radius: 8px; background: transparent; color: ${T.muted};
  font: 600 12px ${T.font}; display: inline-flex; align-items: center; gap: 7px; cursor: pointer; white-space: nowrap; }
.radiance-v5 .rv5-ovbar button[aria-pressed="true"] { background: rgba(255,255,255,.14); color: ${T.text}; }
.radiance-v5 .rv5-legend { left: 16px; bottom: 16px; padding: 12px 14px; border-radius: 14px; flex-direction: column; align-items: stretch; gap: 8px; }
.radiance-v5 .rv5-legend-row { display: flex; gap: 2px; border-radius: 6px; overflow: hidden; }
.radiance-v5 .rv5-legend-row > div { width: 70px; display: flex; flex-direction: column; gap: 6px; font-size: 11px; color: ${T.soft}; }
.radiance-v5 .rv5-legend-row i { height: 8px; display: block; }
.radiance-v5 .rv5-badge { left: 50%; bottom: 16px; transform: translateX(-50%); height: 34px; padding: 0 8px 0 12px; border-radius: 10px;
  background: rgba(40,30,8,.86); border: 1px solid rgba(245,181,68,.45); color: #f7c873; font-weight: 600; gap: 10px; }
.radiance-v5 .rv5-badge button { height: 24px; padding: 0 8px; border-radius: 6px; border: 0; background: rgba(245,181,68,.2); color: #f7c873;
  font: 600 12px ${T.font}; cursor: pointer; }
.radiance-v5 .rv5-hide { display: none !important; }
.radiance-v5:not(.rv5-panel-off):not(.rv5-tiny) .rv5-ovbar, .radiance-v5:not(.rv5-panel-off):not(.rv5-tiny) .rv5-badge { left: calc((100% - 368px) / 2); }
.radiance-v5.rv5-narrow:not(.rv5-panel-off):not(.rv5-tiny) .rv5-ovbar, .radiance-v5.rv5-narrow:not(.rv5-panel-off):not(.rv5-tiny) .rv5-badge { left: calc((100% - 320px) / 2); }
/* Bottom bar. */
.radiance-v5.radiance-mode-simple .radiance-simple-bar { height: 84px; flex: 0 0 84px; gap: 16px; padding: 0 16px; margin: 0;
  background: ${T.chrome}; border: 0; border-top: 1px solid var(--rv5-line); border-radius: 0; box-shadow: none; font: 12px ${T.font}; color: ${T.soft}; }
.radiance-v5 .radiance-simple-bar button { font: 600 12px ${T.font}; }
.radiance-v5 .rv5-transport { display: flex; align-items: center; gap: 4px; flex-shrink: 0; }
.radiance-v5 .rv5-transport button { width: 36px; height: 36px; padding: 0; border-radius: 10px; border: 0; background: transparent; color: ${T.muted};
  display: inline-flex; align-items: center; justify-content: center; cursor: pointer; }
.radiance-v5 .rv5-transport button:hover:not(:disabled) { color: ${T.text}; background: rgba(255,255,255,.07); }
.radiance-v5 .radiance-simple-bar .rsb-play, .radiance-v5 .radiance-simple-bar .rsb-play.is-active, .radiance-v5 .radiance-simple-bar .rsb-play:hover:not(:disabled) {
  width: 48px; height: 48px; border-radius: 24px; background: var(--rv5-accent); color: ${T.accentInk}; box-shadow: 0 6px 20px rgba(0,0,0,.35); }
.radiance-v5 .rv5-time { width: 112px; display: flex; flex-direction: column; gap: 2px; flex-shrink: 0; }
.radiance-v5 .rv5-tc { font: 500 14px ${T.mono}; color: ${T.text}; }
.radiance-v5 .radiance-simple-bar .rsb-frame { font: 11px ${T.font}; color: ${T.muted}; text-align: left; min-width: 0; }
.radiance-v5 .rv5-strip-wrap { flex: 1 1 auto; position: relative; height: 50px; min-width: 80px; }
.radiance-v5 .rv5-strip { position: absolute; inset: 0; border-radius: 10px; overflow: hidden; display: flex; gap: 2px; background: #000;
  box-shadow: inset 0 0 0 1px rgba(255,255,255,.08); pointer-events: none; }
.radiance-v5 .rv5-strip img { flex: 1 1 0; min-width: 0; height: 100%; object-fit: cover; opacity: .9; }
.radiance-v5 .rv5-strip.is-empty { background: rgba(255,255,255,.05); }
.radiance-v5 .rv5-strip-wrap .radiance-cache-marks { position: absolute; inset: 0; }
.radiance-v5 .rv5-strip-wrap .rsb-scrub { -webkit-appearance: none; appearance: none; width: 100%; height: 100%; margin: 0; background: transparent; cursor: pointer; }
.radiance-v5 .rv5-strip-wrap .rsb-scrub::-webkit-slider-runnable-track { height: 100%; background: transparent; }
.radiance-v5 .rv5-strip-wrap .rsb-scrub::-webkit-slider-thumb { -webkit-appearance: none; width: 4px; height: 66px; margin-top: -8px; border-radius: 2px;
  background: #fafafa; box-shadow: 0 0 8px rgba(0,0,0,.7); }
.radiance-v5 .rv5-strip-wrap .rsb-scrub::-moz-range-track { height: 100%; background: transparent; }
.radiance-v5 .rv5-strip-wrap .rsb-scrub::-moz-range-thumb { width: 4px; height: 66px; border: 0; border-radius: 2px; background: #fafafa; }
.radiance-v5 .rv5-bright { display: flex; align-items: center; gap: 10px; flex-shrink: 0; color: ${T.muted}; }
.radiance-v5 .rv5-bright input { width: 130px; accent-color: var(--rv5-accent); }
.radiance-v5 .rv5-bright output { width: 40px; font: 12px ${T.mono}; color: ${T.soft}; }
.radiance-v5 .rv5-bright.is-on output { color: #f7c873; }
.radiance-v5 .rv5-zoom { height: 36px; border-radius: 10px; border: 1px solid rgba(255,255,255,.10); background: transparent; color: ${T.text};
  font: 13px ${T.font}; padding: 0 8px; flex-shrink: 0; }
.radiance-v5 .rv5-zoom option, .radiance-v5 .rv5-panel select option { background: ${T.raised}; color: ${T.text}; }
.radiance-v5 .rv5-sr { position: absolute; width: 1px; height: 1px; overflow: hidden; clip-path: inset(50%); }
/* Input settings, in the panel. */
.radiance-v5 .rv5-field { display: flex; flex-direction: column; gap: 6px; }
.radiance-v5 .rv5-field label, .radiance-v5 .rv5-field .rv5-flabel { font: 600 12px ${T.font}; color: ${T.soft}; }
.radiance-v5 .rv5-field select, .radiance-v5 .rv5-field input[type="number"] { height: 36px; border-radius: 10px; border: 1px solid rgba(255,255,255,.10);
  background: rgba(255,255,255,.05); color: ${T.text}; font: 13px ${T.font}; padding: 0 10px; box-sizing: border-box; width: 100%; }
.radiance-v5 .rv5-field input[type="number"] { font-family: ${T.mono}; }
.radiance-v5 .rv5-seg { display: inline-flex; gap: 2px; padding: 3px; border-radius: 10px; background: rgba(255,255,255,.05); align-self: flex-start; }
.radiance-v5 .rv5-seg button { height: 30px; padding: 0 12px; border: 0; border-radius: 7px; background: transparent; color: ${T.muted}; font: 600 12px ${T.font}; cursor: pointer; }
.radiance-v5 .rv5-seg button[aria-pressed="true"] { background: ${T.text}; color: ${T.canvas}; }
.radiance-v5 .rv5-switchrow { display: flex; align-items: center; justify-content: space-between; gap: 12px; }
.radiance-v5 .rv5-switch { width: 44px; height: 26px; flex-shrink: 0; border-radius: 13px; border: 0; padding: 3px; background: #3f3f46;
  display: flex; justify-content: flex-start; cursor: pointer; }
.radiance-v5 .rv5-switch[aria-checked="true"] { background: var(--rv5-accent); justify-content: flex-end; }
.radiance-v5 .rv5-switch::after { content: ''; width: 20px; height: 20px; border-radius: 10px; background: ${T.text}; display: block; }
.radiance-v5 .rv5-field [disabled], .radiance-v5 .rv5-switch[disabled] { opacity: .5; cursor: default; }
/* Smaller nodes: the picture and transport stay; words go first. */
.radiance-v5.rv5-narrow .rv5-shot, .radiance-v5.rv5-narrow .rv5-look-key, .radiance-v5.rv5-narrow .rv5-save-word,
.radiance-v5.rv5-narrow .rv5-hints, .radiance-v5.rv5-narrow .rv5-bright input, .radiance-v5.rv5-narrow .rv5-help { display: none !important; }
.radiance-v5.rv5-narrow .rv5-divider { display: none; }
.radiance-v5.rv5-narrow .radiance-pro-brand .rv5-brand-word { display: none; }
.radiance-v5.rv5-narrow .rv5-panel { width: 300px; top: 10px; right: 10px; bottom: 10px; }
.radiance-v5.rv5-tiny .rv5-tasks button { padding: 0 10px; }
.radiance-v5.rv5-tiny .rv5-task-word, .radiance-v5.rv5-tiny .rv5-look, .radiance-v5.rv5-tiny .rv5-time, .radiance-v5.rv5-tiny .rv5-zoom,
.radiance-v5.rv5-tiny .rv5-pro-btn, .radiance-v5.rv5-tiny .rv5-bright { display: none !important; }
.radiance-v5.rv5-tiny .rv5-panel { left: 10px; width: auto; }
.radiance-v5.rv5-tiny.radiance-mode-simple .radiance-simple-bar { height: 64px; flex-basis: 64px; gap: 10px; padding: 0 10px; }
.radiance-v5.rv5-tiny .radiance-simple-bar .rsb-play { width: 40px; height: 40px; }
.radiance-v5.rv5-tiny .rv5-strip-wrap { height: 40px; }
`;
}

function el(tag, cls, html) {
    const e = document.createElement(tag);
    if (cls) e.className = cls;
    if (html !== undefined) e.innerHTML = html;
    return e;
}

function button(cls, label, title, onClick) {
    const b = el('button', cls);
    b.type = 'button';
    if (label) b.textContent = label;
    if (title) { b.title = title; b.setAttribute('aria-label', title); }
    if (onClick) b.onclick = onClick;
    return b;
}

function iconButton(icon, title, onClick, cls = 'rv5-ib') {
    const b = button(cls, '', title, onClick);
    b.innerHTML = svgIcon(icon, 18);
    return b;
}

/** Close a popover on a click outside it or Escape. */
function popover(wrap, btn, pop, signal, onOpen) {
    const set = (on) => {
        pop.classList.toggle('is-open', on);
        btn.setAttribute('aria-expanded', String(on));
        if (on) onOpen?.();
    };
    btn.addEventListener('click', (e) => { e.stopPropagation(); set(!pop.classList.contains('is-open')); }, { signal });
    document.addEventListener('pointerdown', (e) => { if (!wrap.contains(e.target)) set(false); }, { capture: true, signal });
    pop.addEventListener('keydown', (e) => { if (e.key === 'Escape') { set(false); btn.focus(); } }, { signal });
    return set;
}

/**
 * Apply the shell to a viewer whose UI and Simple bar exist. Returns the
 * shell ({ sync, setTask, destroy, … }). A second call returns the first shell.
 */
export function installShell(v) {
    if (v._shell) return v._shell;
    if (typeof document !== 'undefined' && !document.getElementById('radiance-v5-shell')) {
        const st = document.createElement('style');
        st.id = 'radiance-v5-shell';
        st.textContent = shellCSS();
        document.head.appendChild(st);
    }
    const root = v.container;
    root.classList.add('radiance-v5');
    const ctl = new AbortController();
    const signal = ctl.signal;
    v._listenerSignal?.addEventListener?.('abort', () => ctl.abort(), { signal });
    const bar = v.proMenuBar;
    const props = () => (v.node ? (v.node.properties ||= {}) : {});
    let task = 'view';
    let lastCompare = 'wipe';
    let lastOverlay = 'exposure';

    // ── Header: brand.
    const brand = bar?.querySelector('.radiance-pro-brand');
    if (brand) {
        const mark = brand.querySelector('.radiance-pro-mark');
        if (mark) { mark.classList.add('rv5-icon'); mark.innerHTML = RADIANCE_ICON_SVG; }
        const word = [...brand.children].find((c) => c.tagName === 'SPAN' && !c.className);
        if (word) { word.className = 'rv5-brand-word'; word.textContent = 'Radiance'; }
    }
    const divider = el('div', 'rv5-divider');
    const shot = el('div', 'rv5-shot');
    const shotName = el('span', 'rv5-shot-name');
    const shotMeta = el('span', 'rv5-chip');
    const hdrBadge = el('span', 'rv5-hdr', 'HDR');
    hdrBadge.title = 'The frame holds values above SDR white';
    shot.append(shotName, shotMeta, hdrBadge);

    // ── Tasks.
    const tasksNav = el('nav', 'rv5-tasks');
    tasksNav.setAttribute('aria-label', 'Viewer task');
    const taskBtns = new Map();
    const count = el('span', 'rv5-count');
    for (const t of TASKS) {
        const b = button('', '', t.tip, () => shell.setTask(t.id));
        b.innerHTML = `${svgIcon(t.id, 16)}<span class="rv5-task-word">${t.label}</span>`;
        b.setAttribute('aria-label', t.label);
        if (t.id === 'check') b.append(count);
        taskBtns.set(t.id, b);
        tasksNav.append(b);
    }

    // ── Right: Look, Save frame, full screen, panel, Pro tools, help.
    const right = el('div', 'rv5-right');
    const look = el('div', 'rv5-look');
    const lookBtn = button('rv5-btn', '', 'Look: how the picture is shown on this screen');
    lookBtn.setAttribute('aria-haspopup', 'menu');
    lookBtn.setAttribute('aria-expanded', 'false');
    const lookMenu = el('div', 'rv5-menu rv5-glass');
    lookMenu.setAttribute('role', 'menu');
    lookMenu.setAttribute('aria-label', 'Look');
    look.append(lookBtn, lookMenu);
    const buildLookMenu = () => {
        const src = v._viewSelect;
        const opts = src ? [...src.options].map((o) => [o.value, o.textContent]) : (v.constructor?.VIEW_MODES || []);
        const head = el('div', 'rv5-sec', 'How the picture looks');
        const items = opts.map(([id, label]) => {
            const l = lookFor(id, label);
            const b = el('button', 'rv5-mi');
            b.type = 'button';
            b.setAttribute('role', 'menuitemradio');
            b.setAttribute('aria-checked', String((v.viewMode || 'auto') === id));
            b.dataset.look = id;
            b.innerHTML = `<i class="rv5-mi-dot"></i><span class="rv5-mi-txt"><b></b><span></span></span><span class="rv5-mi-tick">${svgIcon('tick', 16)}</span>`;
            b.querySelector('b').textContent = l.name;
            b.querySelector('.rv5-mi-txt span').textContent = l.desc;
            b.onclick = () => { v.setViewMode?.(id); setLook(false); shell.syncLook(); };
            return b;
        });
        const note = el('div', 'rv5-menu-note', 'Only changes what you see here. Your saved file is not affected.');
        lookMenu.replaceChildren(head, ...items, note);
    };
    const setLook = popover(look, lookBtn, lookMenu, signal, buildLookMenu);

    const save = el('div', 'rv5-split');
    const saveBtn = button('rv5-btn', '', 'Save this frame as a PNG', () => v.exportSnapshot?.('png'));
    saveBtn.innerHTML = `${svgIcon('camera', 16)}<span class="rv5-save-word">Save frame</span>`;
    const saveMore = button('rv5-btn', '', 'More ways to export: EXR, grade, Deliver');
    saveMore.innerHTML = svgIcon('chevron', 14);
    saveMore.onclick = (e) => { e.stopPropagation(); v.showExportMenu?.({ target: saveMore, currentTarget: saveMore }); };
    save.append(saveBtn, saveMore);
    const fullBtn = iconButton('fullscreen', 'Full screen', () => v.toggleFullscreen?.());
    const panelBtn = iconButton('panel', 'Details panel', () => shell.setPanel(!panelOpen()));
    panelBtn.classList.add('rv5-simple-only');
    const proBtn = button('rv5-btn rv5-pro-btn', '', 'Pro tools: grade, scopes, channels and the full QC report', () => {
        if (v.uiMode === 'advanced') shell.setTask(task);
        else v.setWorkspace?.(fullWorkspaceFor(v._lastFullWorkspace), { grow: true });
    });
    proBtn.innerHTML = `${svgIcon('pro', 16)}<span class="rv5-save-word">Pro tools</span>`;
    const helpBtn = iconButton('help', 'Help and keyboard shortcuts', () => v.toggleHelp?.());
    helpBtn.classList.add('rv5-help');
    right.append(look, save, fullBtn, panelBtn, proBtn, helpBtn);
    divider.classList.add('rv5-simple-only');
    shot.classList.add('rv5-simple-only');
    look.classList.add('rv5-simple-only');
    save.classList.add('rv5-simple-only');

    // Pro tools keep their workspace tabs (Color, Inspect, QC) and menus.
    const sw = bar?.querySelector('.radiance-mode-switch');
    if (sw) {
        sw.setAttribute('aria-label', 'Pro tools workspace');
        v._modeBtns?.forEach((b, id) => { if (id === 'review') b.setAttribute('data-rv5-hidden', ''); });
    }
    const menuItems = bar?.querySelector('.radiance-pro-menu-items');
    const spacer = el('div', 'rv5-spacer');
    const spacer0 = el('div', 'rv5-spacer');
    if (bar) {
        const after = brand || null;
        const seq = [divider, shot, spacer0, tasksNav, sw, spacer, menuItems, right].filter(Boolean);
        if (after) after.after(...seq); else bar.prepend(...seq);
    }

    // ── The stage: the panel and what floats over the picture.
    const stage = v.canvasWrapper || root;
    const panel = el('aside', 'rv5-panel rv5-glass rv5-stage-ui');
    const panelHead = el('div', 'rv5-panel-head');
    const panelTitle = el('b');
    panelHead.append(panelTitle, iconButton('close', 'Close panel', () => shell.setPanel(false)));
    const bodies = { view: el('div', 'rv5-body'), compare: el('div', 'rv5-body'), check: el('div', 'rv5-body') };
    panel.append(panelHead, bodies.view, bodies.compare, bodies.check);

    // View: what the frame is.
    const tiles = el('div', 'rv5-tiles');
    const tile = (label) => {
        const t = el('div', 'rv5-tile');
        const s = el('small'); s.textContent = label;
        const b = el('b'); const d = el('span');
        t.append(s, b, d);
        tiles.append(t);
        return { b, d };
    };
    const tSize = tile('Size'), tColour = tile('Colour'), tQuality = tile('Quality'), tLength = tile('Length');
    const brightGroup = el('div', 'rv5-group');
    const brightHead = el('div', 'rv5-sec', 'Brightness');
    const stats = el('div', 'rv5-stats');
    const stat = (label) => {
        const s = el('div', 'rv5-stat');
        const l = el('small'); l.textContent = label;
        const b = el('b'); b.textContent = '—';
        s.append(l, b); stats.append(s);
        return b;
    };
    const sPeak = stat('Brightest'), sRange = stat('Range'), sAbove = stat('Above white');
    const brightNote = el('div', 'rv5-note', 'Brightness details appear for float frames (EXR, HDR, linear).');
    const callout = el('div', 'rv5-callout');
    const calloutText = el('span');
    const calloutDo = button('rv5-soft-btn', 'Show me where', 'Show where on the picture', () => {
        shell.setTask('check'); shell.setOverlay('clip');
    });
    const calloutCol = el('div');
    calloutCol.append(calloutText, calloutDo);
    callout.innerHTML = svgIcon('warn', 18);
    callout.append(calloutCol);
    brightGroup.append(brightHead, stats, brightNote);

    const inputFold = el('details', 'rv5-fold rv5-group');
    const inputSummary = el('summary');
    inputSummary.innerHTML = `<span class="rv5-sec">Input settings</span>${svgIcon('chevron', 14)}`;
    const inputBody = el('div', 'rv5-fold-body');
    inputFold.append(inputSummary, inputBody);
    inputFold.addEventListener('toggle', () => { if (inputFold.open) buildInputSettings(); }, { signal });

    const reviewGroup = el('div', 'rv5-group');
    reviewGroup.append(el('div', 'rv5-sec', 'Notes and marks'));
    const moreGroup = el('div', 'rv5-group');
    moreGroup.style.gap = '2px';
    const moreHead = el('div', 'rv5-sec', 'More tools');
    moreHead.style.marginBottom = '6px';
    moreGroup.append(moreHead);
    for (const m of MORE_TOOLS) {
        const r = button('rv5-row', '', `${m.title}: opens Pro tools`, () => v.setWorkspace?.(m.workspace, { grow: true, tab: m.tab }));
        r.innerHTML = `<span><b></b><small></small></span>${svgIcon('right', 14)}`;
        r.querySelector('b').textContent = m.title;
        r.querySelector('small').textContent = m.text;
        r.dataset.tool = m.id;
        moreGroup.append(r);
    }
    bodies.view.append(tiles, brightGroup, callout, inputFold, reviewGroup, moreGroup);

    const uid = `rv5-${Math.random().toString(36).slice(2, 8)}`;
    function buildInputSettings() {
        const parts = [];
        for (const st of inputSettingsState(v.node)) {
            if (!st.present) continue;
            const id = `${uid}-${st.name}`;
            const note = st.linked ? 'Set by a link into the node.' : st.hint;
            const hint = el('div', 'rv5-note'); hint.textContent = note;
            if (st.kind === 'switch') {
                const row = el('div', 'rv5-switchrow');
                const txt = el('div', 'rv5-field');
                const b = el('div', 'rv5-flabel'); b.textContent = st.label;
                txt.append(b, hint);
                const s = button('rv5-switch', '', st.label);
                s.setAttribute('role', 'switch');
                s.setAttribute('aria-checked', String(!!st.value));
                s.disabled = st.linked;
                s.onclick = () => { if (setInputSetting(v.node, st.name, !st.value)) buildInputSettings(); };
                row.append(txt, s);
                parts.push(row);
                continue;
            }
            const field = el('div', 'rv5-field');
            if (st.kind === 'segmented' && st.options) {
                const lab = el('div', 'rv5-flabel'); lab.textContent = st.label;
                const seg = el('div', 'rv5-seg');
                seg.setAttribute('role', 'group');
                seg.setAttribute('aria-label', st.label);
                for (const opt of st.options) {
                    const o = button('', opt, '', () => { if (setInputSetting(v.node, st.name, opt)) buildInputSettings(); });
                    o.removeAttribute('aria-label');
                    o.setAttribute('aria-pressed', String(st.value === opt));
                    o.disabled = st.linked;
                    seg.append(o);
                }
                field.append(lab, seg);
            } else if (st.kind === 'combo' && st.options) {
                const lab = el('label'); lab.textContent = st.label; lab.htmlFor = id;
                const sel = document.createElement('select');
                sel.id = id;
                for (const opt of st.options) { const o = document.createElement('option'); o.value = o.textContent = opt; sel.append(o); }
                sel.value = st.value;
                sel.disabled = st.linked;
                sel.onchange = () => { setInputSetting(v.node, st.name, sel.value); shell.syncDetails(); };
                field.append(lab, sel);
            } else {
                const lab = el('label'); lab.textContent = st.label; lab.htmlFor = id;
                const num = document.createElement('input');
                num.type = 'number'; num.id = id; num.step = '0.001'; num.min = '0';
                num.value = Number(st.value ?? 0).toFixed(3);
                num.disabled = st.linked;
                num.onchange = () => { setInputSetting(v.node, st.name, num.value); };
                field.append(lab, num);
            }
            field.append(hint);
            parts.push(field);
        }
        if (!parts.length) parts.push(Object.assign(el('div', 'rv5-note'), { textContent: 'This node has no input settings.' }));
        else parts.push(Object.assign(el('div', 'rv5-note'), { textContent: 'Saved with the node. Changes apply on the next run.' }));
        inputBody.replaceChildren(...parts);
    }

    // Compare: B and how to show it.
    const cmpB = el('div', 'rv5-group');
    cmpB.append(el('div', 'rv5-sec', 'Compare against (B)'));
    const cmpEmpty = el('div', 'rv5-note',
        'Nothing to compare yet. Connect the compare_image input, or grab this frame as a still and change something.');
    const cmpShow = el('div', 'rv5-group');
    cmpShow.append(el('div', 'rv5-sec', 'Show as'));
    const cmpGrid = el('div', 'rv5-cmp');
    cmpGrid.setAttribute('role', 'group');
    cmpGrid.setAttribute('aria-label', 'Compare style');
    const cmpDiffOpts = el('div', 'rsb-group');
    cmpDiffOpts.setAttribute('aria-label', 'Difference options');
    cmpShow.append(cmpGrid, cmpDiffOpts);
    const cmpTip = el('div', 'rv5-note', 'Drag the line on the picture to move the split.');
    bodies.compare.append(cmpB, cmpShow, cmpTip);

    // Check: run the checks, read them in sentences, show their overlays.
    const checkSummary = el('div', 'rv5-note');
    const runBtn = button('rv5-primary', 'Run checks', 'Measure every frame: broken pixels, range, alpha, flicker', () => v.runQC?.());
    const runRow = el('div');
    runRow.style.cssText = 'display:flex;gap:8px;align-items:center';
    const fullQC = button('rv5-soft-btn', 'Full report', 'Every check and flagged frame, in Pro tools', () => v.setWorkspace?.('qc', { grow: true }));
    fullQC.style.alignSelf = 'center';
    runRow.append(runBtn, fullQC);
    const checkList = el('div', 'rv5-checks');
    bodies.check.append(checkSummary, runRow, checkList);

    // Over the picture.
    const hints = el('div', 'rv5-float rv5-hints rv5-glass rv5-stage-ui');
    hints.innerHTML = '<span><kbd>Space</kbd>Play</span><span><kbd>← →</kbd>Step</span><span><kbd>?</kbd>All shortcuts</span>';
    const ovbar = el('div', 'rv5-float rv5-ovbar rv5-glass rv5-stage-ui');
    ovbar.setAttribute('role', 'group');
    ovbar.setAttribute('aria-label', 'Overlay');
    const ovBtns = new Map();
    for (const o of OVERLAYS) {
        const b = button('', o.label, '', () => shell.setOverlay(o.id));
        b.removeAttribute('aria-label');
        ovBtns.set(o.id, b);
        ovbar.append(b);
    }
    const legend = el('div', 'rv5-float rv5-legend rv5-glass rv5-stage-ui');
    const legendRow = el('div', 'rv5-legend-row');
    for (const g of EXPOSURE_LEGEND) {
        const d = el('div');
        const i = el('i'); i.style.background = g.c;
        d.append(i, g.label);
        legendRow.append(d);
    }
    legend.append(el('div', 'rv5-sec', 'Exposure map'), legendRow);
    const aiLegend = el('div', 'rv5-float rv5-legend rv5-glass rv5-stage-ui');
    const aiRow = el('div', 'rv5-legend-row');
    for (const g of REGION_LEGEND) {
        const d = el('div');
        d.style.width = '96px';
        const i = el('i'); i.style.background = g.c;
        d.append(i, g.label);
        aiRow.append(d);
    }
    aiLegend.append(el('div', 'rv5-sec', 'Rebuilt by AI'), aiRow);
    const badge = el('div', 'rv5-float rv5-badge rv5-stage-ui');
    badge.setAttribute('role', 'status');
    const badgeText = el('span');
    badge.innerHTML = svgIcon('sun', 15);
    badge.append(badgeText, button('', 'Reset', 'Reset preview brightness', () => v.setViewExposure?.(0)));
    stage.append(panel, hints, ovbar, legend, aiLegend, badge);

    // ── Move the viewer's own Simple-bar controls into place.
    const sb = v.simpleBar;
    let tc = null, frameText = null, strip = null, bright = null, brightInput = null, brightOut = null, zoom = null;
    if (sb) {
        const groups = [...sb.querySelectorAll('.rsb-group')];
        const cmp = groups.find((g) => g.getAttribute('aria-label') === 'Compare');
        const review = groups.find((g) => g.getAttribute('aria-label') === 'Review');
        const transport = v._sbPlay?.parentElement;
        const scrubWrap = v._sbScrub?.closest('.radiance-cache-marks') || v._sbScrub;

        if (transport) {
            transport.classList.remove('rsb-group');
            transport.classList.add('rv5-transport');
            if (v._sbPrev) v._sbPrev.innerHTML = svgIcon('prev', 18);
            if (v._sbNext) v._sbNext.innerHTML = svgIcon('next', 18);
        }
        const time = el('div', 'rv5-time');
        tc = el('div', 'rv5-tc');
        tc.setAttribute('aria-label', 'Timecode');
        time.append(tc);
        if (v._sbFrame) time.append(v._sbFrame);
        frameText = v._sbFrame;

        const stripWrap = el('div', 'rv5-strip-wrap');
        strip = el('div', 'rv5-strip is-empty');
        strip.setAttribute('aria-hidden', 'true');
        stripWrap.append(strip);
        if (scrubWrap) stripWrap.append(scrubWrap);

        bright = el('div', 'rv5-bright');
        bright.innerHTML = svgIcon('sun', 18);
        brightInput = document.createElement('input');
        brightInput.type = 'range'; brightInput.min = '-6'; brightInput.max = '6'; brightInput.step = '0.1'; brightInput.value = '0';
        brightInput.setAttribute('aria-label', 'Preview brightness, in stops. Not saved.');
        brightInput.title = 'Preview brightness: only what you see here, never the saved file. Double-click to reset.';
        brightInput.addEventListener('input', () => {
            v.viewExposure = Number(brightInput.value) || 0;
            v.render?.();
            shell.syncExposure(true);
        }, { signal });
        brightInput.addEventListener('change', () => v.setViewExposure?.(Number(brightInput.value) || 0), { signal });
        brightInput.addEventListener('dblclick', () => v.setViewExposure?.(0), { signal });
        brightOut = el('output');
        bright.append(brightInput, brightOut);

        const zl = el('label', 'rv5-sr', 'Zoom');
        zl.htmlFor = `${uid}-zoom`;
        zoom = document.createElement('select');
        zoom.className = 'rv5-zoom';
        zoom.id = `${uid}-zoom`;
        for (const [val, label] of [['fit', 'Fit'], ['0.5', '50%'], ['1', '100%'], ['2', '200%']]) {
            const o = document.createElement('option'); o.value = val; o.textContent = label; zoom.append(o);
        }
        zoom.onchange = () => { if (zoom.value === 'fit') v.fitToView?.(); else v.setZoom?.(Number(zoom.value)); };

        // Compare panel: B, Grab still, B fit, the compare buttons and Difference options.
        if (v._sbBLabel) cmpB.append(v._sbBLabel);
        const cmpRow = el('div');
        cmpRow.style.cssText = 'display:flex;gap:6px;flex-wrap:wrap';
        for (const b of [v._sbPin, v._sbBFit]) if (b) { b.classList.add('rv5-moved'); cmpRow.append(b); }
        cmpB.append(cmpRow, cmpEmpty);
        const words = { none: 'A only', b: 'B only', wipe: 'Split', difference: 'Difference', blink: 'Flip' };
        // Side by side is drawn by the 2D fallback only, so it is not offered here.
        const order = ['wipe', 'difference', 'blink', 'b', 'none'];
        v._sbCompare?.forEach((b, m) => { b.textContent = words[m] || b.textContent; });
        for (const m of order) {
            const b = v._sbCompare?.get(m);
            if (b) cmpGrid.append(b);
        }
        for (const b of [v._sbDiffGain, v._sbDiffSigned, v._sbDiffDomain]) if (b) cmpDiffOpts.append(b);
        cmp?.remove();

        // View panel: review marks and notes, and onion skin.
        if (review) reviewGroup.append(review);
        if (v._sbOnion) { v._sbOnion.classList.add('rv5-moved'); reviewGroup.append(v._sbOnion); }
        const fit = [...sb.children].find((c) => c.tagName === 'BUTTON' && c.textContent === 'Fit');
        fit?.remove();

        sb.replaceChildren(...[transport, time, stripWrap, bright, zl, zoom].filter(Boolean));
    }

    // ── Sizes.
    const sizeClasses = () => {
        const w = root.clientWidth || 0;
        const narrow = w > 0 && w < 1000, tiny = w > 0 && w < 700;
        const was = root.classList.contains('rv5-narrow');
        root.classList.toggle('rv5-narrow', narrow);
        root.classList.toggle('rv5-tiny', tiny);
        // A small node opens with the panel closed, so the picture is all you see.
        if (narrow && !was && props().radiance_viewer_panel === undefined) root.classList.add('rv5-panel-off');
    };
    let ro = null;
    if (typeof ResizeObserver !== 'undefined') { ro = new ResizeObserver(sizeClasses); ro.observe(root); }

    // ── Follow the viewer's fields.
    const watched = [v._proFileName, v._proResolution, v._proDepth, v._proFps, v._sbQC].filter(Boolean);
    let mo = null;
    if (typeof MutationObserver !== 'undefined' && watched.length) {
        mo = new MutationObserver(() => { shell.syncShot(); shell.syncDetails(); shell.syncChecks(); });
        watched.forEach((n) => mo.observe(n, { childList: true, characterData: true, subtree: true }));
    }

    const panelOpen = () => !root.classList.contains('rv5-panel-off');
    let stripKey = '';
    let inSetTask = false;
    let stripRetry = 0, stripTries = 0;

    const shell = {
        get task() { return task; },
        /** Go to View, Compare or Check (in the Simple layout). */
        setTask(id) {
            if (inSetTask) return;
            inSetTask = true;
            try { shell._setTask(id); } finally { inSetTask = false; }
            shell.sync();
        },
        _setTask(id) {
            id = normalizeTask(id);
            const from = task;
            if (v.uiMode === 'advanced') v.setWorkspace?.('review');
            task = id;
            if (from === 'compare' && id !== 'compare') {
                if (v.compareMode && v.compareMode !== 'none') lastCompare = v.compareMode;
                v.setCompareMode?.('none');
            }
            if (from === 'check' && id !== 'check') {
                const cur = currentOverlay(v);
                if (cur !== 'off') lastOverlay = cur;
                setOverlay(v, 'off');
            }
            if (id === 'compare' && from !== 'compare' && (!v.compareMode || v.compareMode === 'none')
                && (v.image || v.renderer?.textures?.image)) v.setCompareMode?.(lastCompare);
            if (id === 'check' && from !== 'check') setOverlay(v, lastOverlay === 'ai' && !v.hasRegions?.() ? 'exposure' : lastOverlay);
            if (from !== id && !panelOpen() && !root.classList.contains('rv5-narrow')) shell.setPanel(true, { persist: false });
        },
        setOverlay(id) {
            setOverlay(v, id);
            if (id !== 'off') lastOverlay = id;
            shell.syncOverlay();
            shell.syncChecks();
        },
        setPanel(on, { persist = true } = {}) {
            root.classList.toggle('rv5-panel-off', !on);
            if (persist) props().radiance_viewer_panel = !!on;
            panelBtn.setAttribute('aria-pressed', String(!!on));
        },
        syncShot() {
            const name = (v._proFileName?.textContent || '').trim();
            shotName.textContent = name && name !== 'No shot loaded' ? name : (v.node?.title || 'Radiance Viewer');
            const res = (v._proResolution?.textContent || '').trim();
            const fps = (v._proFps?.textContent || '').trim();
            shotMeta.textContent = [res && res !== '— x —' ? res.replace(/\s*x\s*/i, '×') : '', fps ? fps.replace(/\.00 FPS$/, ' fps').replace(/ FPS$/, ' fps') : '']
                .filter(Boolean).join(' · ');
            shotMeta.style.display = shotMeta.textContent ? '' : 'none';
            const st = v._hdrZoneStats;
            hdrBadge.classList.toggle('is-on', !!(v.hdrData && st && st.nitPeak > 203));
        },
        syncLook() {
            const id = v.viewMode || 'auto';
            const label = v._viewSelect ? [...v._viewSelect.options].find((o) => o.value === id)?.textContent : '';
            lookBtn.innerHTML = `<i class="rv5-look-dot"></i><span class="rv5-look-key">Look</span> `;
            lookBtn.append(lookFor(id, label).name);
            lookBtn.insertAdjacentHTML('beforeend', svgIcon('chevron', 12));
            if (lookMenu.classList.contains('is-open')) buildLookMenu();
        },
        syncDetails() {
            const w = v.imageWidth || 0, h = v.imageHeight || 0;
            tSize.b.textContent = sizeName(w, h);
            tSize.d.textContent = w && h ? `${w} × ${h}` : 'Nothing loaded';
            const space = inputSettingsState(v.node).find((x) => x.name === 'input_space');
            const isHDR = !!(v.hdrData && v._hdrZoneStats?.nitPeak > 203);
            tColour.b.textContent = v.hdrData ? (isHDR ? 'HDR' : 'Float') : 'Standard';
            tColour.d.textContent = shortSpace(space?.value ?? 'Auto');
            const depth = (v._proDepth?.textContent || '').trim();
            tQuality.b.textContent = v.hdrData ? 'Full float' : '8-bit';
            tQuality.d.textContent = depth || (v.hdrData ? 'Float frames' : 'PNG preview');
            const fps = v.playbackFps || v.frameRate || 24;
            tLength.b.textContent = lengthName(v.totalFrames, fps);
            tLength.d.textContent = (v.totalFrames || 0) > 1 ? `${v.totalFrames} frames · ${Math.round(fps * 100) / 100} fps` : '1 frame';

            const st = v._hdrZoneStats;
            const has = !!(st && v.hdrData && Number.isFinite(st.nitPeak));
            stats.style.display = has ? '' : 'none';
            brightNote.style.display = has ? 'none' : '';
            let warn = false;
            if (has) {
                sPeak.textContent = `${Math.round(st.nitPeak).toLocaleString('en-US')} nits`;
                sRange.textContent = `${Number(st.evRange || 0).toFixed(1)} stops`;
                const peak = v._displayPeakNits?.() || 100;
                const above = Number(peak >= 1000 ? st.above1000Pct : st.clippedPct) || 0;
                sAbove.textContent = `${above.toFixed(1)}%`;
                warn = above >= 0.1;
                calloutText.textContent = peak >= 1000
                    ? `${above.toFixed(1)}% of the frame is brighter than a ${peak}-nit HDR screen shows.`
                    : `${above.toFixed(1)}% of the frame is brighter than white on this screen. The Look rolls it off; the file keeps it.`;
            }
            callout.classList.toggle('is-on', warn);
            if (inputFold.open) buildInputSettings();
        },
        syncChecks() {
            const regions = v.hasRegions?.() ? (v.regionShares?.() || 'loading') : null;
            const items = checkItems(v._qcReport, v.hdrData ? v._hdrZoneStats : null, v._displayPeakNits?.() || 100, regions);
            const n = issueCount(items);
            count.textContent = String(n);
            count.classList.toggle('is-on', n > 0 && task !== 'check');
            const running = !!v._qcRunning;
            runBtn.textContent = running ? `Stop (${(v._sbQC?.textContent || '').replace(/^QC\s*/, '') || '…'})`
                : v._qcReport ? 'Run again' : 'Run checks';
            fullQC.disabled = !v._qcReport;
            checkSummary.innerHTML = '';
            const lead = el('b');
            lead.style.color = TOKENS.text;
            if (!items.length) {
                checkSummary.textContent = 'Load a frame, then run the checks. They read the float frames, before the Look and the grade.';
            } else {
                lead.textContent = n ? `${n} thing${n === 1 ? '' : 's'} to look at.` : 'Nothing to worry about.';
                checkSummary.append(lead, v._qcReport ? ' ' : ' Run the checks to measure every frame.');
            }
            const cur = currentOverlay(v);
            checkList.replaceChildren(...items.map((it) => {
                const row = el('div', 'rv5-check');
                row.dataset.tone = it.tone;
                row.dataset.check = it.id;
                const i = el('i');
                i.textContent = it.tone === 'ok' ? '✓' : it.tone === 'skip' ? '–' : '!';
                const col = el('div');
                const b = el('b'); b.textContent = it.title;
                const s = el('span'); s.textContent = it.text;
                col.append(b, s);
                row.append(i, col);
                if (it.overlay) {
                    const on = cur === it.overlay;
                    const sb2 = button('rv5-soft-btn', on ? 'Showing' : 'Show', '', () => shell.setOverlay(on ? 'off' : it.overlay));
                    sb2.removeAttribute('aria-label');
                    sb2.setAttribute('aria-pressed', String(on));
                    row.append(sb2);
                } else if (it.frames?.length) {
                    const f = it.frames[0];
                    const go = button('rv5-soft-btn', `Frame ${f + 1}`, `Go to frame ${f + 1}`, () => v.setFrame?.(f));
                    row.append(go);
                }
                return row;
            }));
        },
        syncOverlay() {
            const cur = currentOverlay(v);
            const hasRegions = !!v.hasRegions?.();
            ovBtns.forEach((b, id) => {
                b.setAttribute('aria-pressed', String(id === cur));
                if (id === 'ai') b.classList.toggle('rv5-hide', !hasRegions);
            });
            const inCheck = task === 'check' && v.uiMode !== 'advanced';
            ovbar.classList.toggle('rv5-hide', !inCheck);
            legend.classList.toggle('rv5-hide', !(inCheck && cur === 'exposure'));
            aiLegend.classList.toggle('rv5-hide', !(inCheck && cur === 'ai'));
        },
        syncCompare() {
            const cm = v.compareMode || 'none';
            // Compare turned on some other way (X, a menu): the Compare task follows.
            if (cm !== 'none' && task !== 'compare' && v.uiMode !== 'advanced' && !inSetTask) { shell.setTask('compare'); return; }
            const hasB = !!(v._hasCompareB?.() || v.frameCompareImages?.length || v._stills?.length);
            cmpEmpty.style.display = hasB ? 'none' : '';
            cmpTip.style.display = cm === 'wipe' ? '' : 'none';
        },
        syncTransport() {
            if (tc) tc.textContent = smpteTimecode(v.currentFrame || 0, v.playbackFps || 24);
            if (v._sbPlay) {
                v._sbPlay.innerHTML = svgIcon(v.isPlaying ? 'pause' : 'play', 18);
                v._sbPlay.setAttribute('aria-label', v.isPlaying ? 'Pause' : 'Play');
            }
            if (frameText) {
                const n = v.totalFrames || 0;
                frameText.textContent = n ? `Frame ${(v.currentFrame || 0) + 1} of ${n}` : 'No frames';
            }
            shell.syncStrip();
        },
        /** Up to 12 preview frames under the scrubber. */
        syncStrip() {
            if (!strip) return;
            const imgs = v.frameImages || [];
            const n = Math.max(v.totalFrames || 0, imgs.length);
            const slots = Math.min(12, n);
            const picks = [];
            for (let i = 0; i < slots; i++) {
                const f = slots === 1 ? 0 : Math.round(i * (n - 1) / (slots - 1));
                const src = imgs[f]?.src || (n <= 1 ? v.image?.src : '');
                picks.push(src || '');
            }
            const real = picks.filter(Boolean);
            // Previews arrive after the frames; look again until every slot has one.
            clearTimeout(stripRetry);
            if (real.length < slots && (stripTries += 1) < 40) stripRetry = setTimeout(() => shell.syncStrip(), 500);
            else if (real.length === slots) stripTries = 0;
            const key = picks.join('|');
            if (key === stripKey) return;
            stripKey = key;
            strip.classList.toggle('is-empty', !real.length);
            strip.replaceChildren(...(real.length ? picks : []).map((src) => {
                const im = document.createElement('img');
                im.alt = '';
                im.decoding = 'async';
                if (src) im.src = src; else im.style.visibility = 'hidden';
                return im;
            }));
        },
        /** live: the slider is moving, so leave its value alone. */
        syncExposure(live = false) {
            const ev = v.viewExposure || 0;
            if (brightInput && !live) brightInput.value = String(Math.max(-6, Math.min(6, ev)));
            if (brightOut) brightOut.textContent = formatEV(ev);
            bright?.classList.toggle('is-on', Math.round(ev * 10) !== 0);
            const text = brightnessBadge(ev);
            badgeText.textContent = text || '';
            badge.classList.toggle('rv5-hide', !text);
            hints.classList.toggle('is-off', !!text || task !== 'view');
        },
        sync() {
            const full = v.uiMode === 'advanced';
            if (full && v.workspace) v._lastFullWorkspace = v.workspace;
            taskBtns.forEach((b, id) => {
                const on = !full && id === task;
                b.classList.toggle('is-active', on);
                b.setAttribute('aria-pressed', String(on));
            });
            proBtn.setAttribute('aria-pressed', String(full));
            const t = TASKS.find((x) => x.id === task);
            panelTitle.textContent = task === 'view' ? 'Details' : task === 'compare' ? 'Compare versions' : 'Check';
            panel.setAttribute('aria-label', t?.label || 'Details');
            Object.entries(bodies).forEach(([id, b]) => b.classList.toggle('is-on', id === task));
            panelBtn.setAttribute('aria-pressed', String(panelOpen()));
            shell.syncShot(); shell.syncLook(); shell.syncDetails(); shell.syncCompare(); shell.syncChecks();
            shell.syncOverlay(); shell.syncTransport(); shell.syncExposure();
            sizeClasses();
        },
        /** Older call sites (4.1 shell names). */
        syncView() { shell.syncLook(); },
        syncInput() { shell.syncDetails(); },
        destroy() {
            ctl.abort();
            clearTimeout(stripRetry);
            ro?.disconnect();
            mo?.disconnect();
            [panel, hints, ovbar, legend, aiLegend, badge].forEach((n) => n.remove());
        },
    };
    v._shell = shell;
    if (props().radiance_viewer_panel === false) root.classList.add('rv5-panel-off');
    shell.sync();
    return shell;
}
