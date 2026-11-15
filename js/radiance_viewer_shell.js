/**
 * Radiance Viewer 4.1 shell: the Radiance gray look, the Simple / Full switch
 * and the Input menu (the node's own settings, so nothing sits outside the viewer).
 *
 * Implements the approved "Radiance Viewer V4 — Simple & Full" mockup on top
 * of the existing viewer. It restyles and rearranges the DOM the viewer
 * already builds; every button keeps calling the viewer method it called
 * before, so behaviour (compare, B, onion, review, QC, report) is unchanged.
 *
 *  - Header: Radiance icon + RADIANCE VIEWER wordmark, a Simple | Full
 *    switch, the Full workspaces as tabs (Color, Inspect, QC), menus on the
 *    right. In Simple: the shot, the view transform and Export.
 *  - Simple: the bar under the picture becomes a floating dock with two rows
 *    (timecode + scrubber; transport, compare, preview exposure, zoom). Less
 *    used controls (B 1:1, grab still, onion, review) move into "More".
 *  - Preview exposure is labelled on the picture as not in delivery.
 *
 * The pure parts (tokens, formatting, dock layout) are exported for Node
 * tests; installShell is the only function that touches the DOM.
 */
import { smpteTimecode } from "./radiance_timecode.js";
import { fullWorkspaces, fullWorkspaceFor } from "./radiance_viewer_modes.js";

/**
 * Radiance space-gray: the colours of the Radiance nodes and Rudra (node body
 * #141419, widgets #121217, outlines #2c2c35), with one silver accent for
 * state. Kept in step with RadianceViewer.THEME.
 */
export const TOKENS = Object.freeze({
    canvas: '#111114',
    chrome: '#1a1a1f',
    raised: '#1e1e24',
    card: '#1e1e24',
    control: '#26262d',
    line: '#2c2c35',
    text: '#f2f2f5',
    soft: '#b4b4bc',
    muted: '#9a9aa2',
    accent: '#d8d8de',
    accentInk: '#111114',
    reference: '#ff9f0a',
    warn: '#ffd60a',
    fail: '#ff6961',
    pass: '#30d158',
    font: "'Manrope', 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif",
    mono: "'DM Mono', 'JetBrains Mono', 'SF Mono', 'Cascadia Code', Consolas, 'Liberation Mono', monospace",
});

/** Bundled fonts (SIL OFL, js/fonts). Served by ComfyUI from the extension folder; no network font request. */
export function fontFaceCSS(base = import.meta.url) {
    const u = (f) => { try { return new URL(`./fonts/${f}`, base).href; } catch { return `./fonts/${f}`; } };
    return `
@font-face { font-family: 'Manrope'; font-style: normal; font-weight: 200 800; font-display: swap;
  src: url('${u('manrope-latin-wght.woff2')}') format('woff2'); }
@font-face { font-family: 'DM Mono'; font-style: normal; font-weight: 400; font-display: swap;
  src: url('${u('dm-mono-latin-400.woff2')}') format('woff2'); }
@font-face { font-family: 'DM Mono'; font-style: normal; font-weight: 500; font-display: swap;
  src: url('${u('dm-mono-latin-500.woff2')}') format('woff2'); }`;
}

/** Viewer f-stops as the dock shows them: ±0.0, +0.5, −1.5 (a real minus sign). */
export function formatEV(stops) {
    const v = Math.round((Number(stops) || 0) * 10) / 10;
    if (v === 0) return '±0.0';
    return (v > 0 ? '+' : '−') + Math.abs(v).toFixed(1);
}

/** The label on the picture while the viewer f-stop is not zero, else null. */
export function exposureBadge(stops) {
    const v = Math.round((Number(stops) || 0) * 10) / 10;
    return v === 0 ? null : `Preview exposure ${formatEV(v)} EV · not in delivery`;
}

/** Simple / Full, from the workspace's layout. */
export function shellMode(uiMode) {
    return uiMode === 'advanced' ? 'full' : 'simple';
}

/**
 * Where the dock puts each existing Simple-bar control. `main` stays on the
 * dock; `more` goes into the More popover. Keys are the viewer fields.
 */
export const DOCK_LAYOUT = Object.freeze({
    main: ['_sbPrev', '_sbPlay', '_sbNext', '_sbBLabel'],
    more: ['_sbBFit', '_sbPin', '_sbOnion'],
});

/** Inline icons (stroke, currentColor), for buttons the dock owns. */
export const ICONS = Object.freeze({
    prev: '<path d="M6 5v14M18 6l-9 6 9 6z"/>',
    next: '<path d="M18 5v14M6 6l9 6-9 6z"/>',
    play: '<path d="M8 5v14l11-7z" fill="currentColor" stroke="none"/>',
    pause: '<rect x="6" y="5" width="4" height="14" rx="1" fill="currentColor" stroke="none"/><rect x="14" y="5" width="4" height="14" rx="1" fill="currentColor" stroke="none"/>',
    minus: '<path d="M5 12h14"/>',
    plus: '<path d="M5 12h14M12 5v14"/>',
    more: '<circle cx="5" cy="12" r="1.4" fill="currentColor"/><circle cx="12" cy="12" r="1.4" fill="currentColor"/><circle cx="19" cy="12" r="1.4" fill="currentColor"/>',
});

export function svgIcon(name, size = 16) {
    return `<svg viewBox="0 0 24 24" width="${size}" height="${size}" fill="none" stroke="currentColor" `
        + `stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${ICONS[name] || ''}</svg>`;
}

/** The Radiance app icon, as on the website (#1C1C1E tile, white ring, blue square). */
export const RADIANCE_ICON_SVG = '<svg viewBox="0 0 100 100" width="22" height="22" aria-hidden="true">'
    + '<rect width="100" height="100" rx="22.5" fill="#1C1C1E"/>'
    + '<path d="M52.2 28.4A24 24 0 1 0 71.6 47.8" fill="none" stroke="#F5F5F7" stroke-width="14"/>'
    + '<rect x="58" y="28" width="14" height="14" rx="3" fill="#0A84FF"/></svg>';

/**
 * The node's own settings, shown in the header's Input menu instead of as
 * ComfyUI widget rows above the viewer. Names are the INPUT_TYPES keys in
 * nodes/monitor/viewer.py; the values stay in the node's widgets, so saved
 * workflows, the API format and older Radiance versions are unchanged.
 */
export const INPUT_SETTINGS = Object.freeze([
    { name: 'input_space', label: 'Input space', kind: 'combo',
      hint: 'Auto reads the pixels as linear when any value leaves 0\u20131, otherwise sRGB.' },
    { name: 'float_precision', label: 'Float precision', kind: 'segmented',
      hint: 'Half is what OpenEXR, RV and Nuke use for display; Full keeps every fp32 bit.' },
    { name: 'fps', label: 'Playback rate', kind: 'number', hint: '0 = the source rate (24 for an image batch).' },
    { name: 'exposure_bracketing', label: 'Exposure brackets', kind: 'switch',
      hint: 'Writes \u22122 / +2 EV thumbnails for the analysis shelf. Costs disk on long shots.' },
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

/** The input space as the header button names it: "sRGB (ComfyUI IMAGE)" reads "sRGB". */
export function shortSpace(v) {
    return String(v ?? 'Auto').replace(/\s*\(.*\)\s*$/, '') || 'Auto';
}

/** Shell styles. Scoped to .radiance-v4 so nothing leaks into ComfyUI. */
export function shellCSS() {
    const T = TOKENS;
    return `
.radiance-v4 { --rv4-accent: ${T.accent}; --rv4-line: ${T.line}; }
.radiance-v4 .radiance-pro-menu { flex: 0 0 52px; gap: 14px; padding: 0 12px 0 14px; background: ${T.chrome};
  border-bottom: 1px solid var(--rv4-line); overflow: visible; }
.radiance-v4 .radiance-pro-brand { min-width: 0; gap: 9px; font: 700 12px ${T.font}; letter-spacing: .22em; }
.radiance-v4 .radiance-pro-brand .rv4-accent { color: var(--rv4-accent); }
.radiance-v4 .radiance-pro-mark.rv4-icon { width: 22px; height: 22px; display: block; gap: 0; }
.radiance-v4 .radiance-pro-mark.rv4-icon svg { display: block; }
.radiance-v4 .radiance-pro-version { letter-spacing: .04em; }
.radiance-v4 .rv4-seg { display: inline-flex; gap: 2px; padding: 3px; border-radius: 10px;
  background: rgba(255,255,255,.05); border: 1px solid rgba(255,255,255,.08); flex-shrink: 0; }
.radiance-v4 .rv4-seg button { height: 28px; padding: 0 12px; border: 0; border-radius: 7px; background: transparent;
  color: ${T.soft}; font: 600 12px ${T.font}; cursor: pointer; white-space: nowrap; }
.radiance-v4 .rv4-seg button:hover { color: ${T.text}; }
.radiance-v4 .rv4-seg button.is-active { background: ${T.text}; color: ${T.canvas}; }
/* Workspace tabs (the existing switch, restyled): Full only. */
.radiance-v4 .radiance-mode-switch { align-self: stretch; gap: 2px; padding: 0; border: 0; border-radius: 0; background: transparent; }
.radiance-v4 .radiance-mode-switch button { height: 100%; padding: 0 12px; border-radius: 0; font: 700 13px ${T.font};
  color: ${T.muted}; background: transparent; box-shadow: inset 0 -2px 0 transparent; }
.radiance-v4 .radiance-mode-switch button:hover { color: ${T.text}; }
.radiance-v4 .radiance-mode-switch button.is-active { background: transparent; color: ${T.text}; box-shadow: inset 0 -2px 0 var(--rv4-accent); }
.radiance-v4 .radiance-mode-switch button[data-rv4-hidden] { display: none; }
.radiance-v4.radiance-mode-simple .radiance-mode-switch { display: none; }
.radiance-v4 .rv4-spacer { flex: 1 1 auto; min-width: 8px; }
.radiance-v4 .radiance-pro-menu-items { gap: 2px; color: ${T.muted}; font: 500 12px ${T.font}; }
.radiance-v4 .radiance-pro-menu-items button { padding: 5px 8px; }
/* Simple header: shot, view, export. */
.radiance-v4 .rv4-simple-only { display: none; }
.radiance-v4.radiance-mode-simple .rv4-simple-only { display: flex; }
.radiance-v4 .rv4-shot { align-items: baseline; gap: 10px; min-width: 0; overflow: hidden; }
.radiance-v4 .rv4-shot-name { font: 700 13px ${T.font}; letter-spacing: 0; color: ${T.text}; white-space: nowrap;
  overflow: hidden; text-overflow: ellipsis; max-width: 260px; }
.radiance-v4 .rv4-shot-meta { font: 400 11px ${T.mono}; color: ${T.muted}; white-space: nowrap; }
.radiance-v4 .rv4-actions { align-items: center; gap: 8px; flex-shrink: 0; }
.radiance-v4 .rv4-view { height: 32px; max-width: 240px; padding: 0 28px 0 10px; border-radius: 9px; border: 1px solid rgba(255,255,255,.12);
  background: transparent; color: ${T.text}; font: 600 12px ${T.font}; cursor: pointer; }
.radiance-v4 .rv4-view option { background: ${T.raised}; color: ${T.text}; }
.radiance-v4 .rv4-primary { height: 32px; padding: 0 14px; border: 0; border-radius: 9px; background: ${T.text};
  color: ${T.accentInk}; font: 700 12px ${T.font}; cursor: pointer; }
.radiance-v4 .rv4-primary:hover { background: #ffffff; }
/* Simple: the dock. */
.radiance-v4.radiance-mode-simple .radiance-simple-bar { flex-direction: column; align-items: stretch; gap: 8px;
  margin: 0 12px 12px; padding: 10px 12px; border: 1px solid var(--rv4-line); border-radius: 14px;
  background: rgba(16,16,20,.96); box-shadow: 0 14px 40px rgba(0,0,0,.5); font: 12px ${T.font}; color: ${T.soft}; }
.radiance-v4 .rv4-row { display: flex; align-items: center; gap: 10px; min-width: 0; }
.radiance-v4 .radiance-simple-bar button { height: 30px; border-radius: 7px; font: 600 12px ${T.font}; color: ${T.soft}; }
.radiance-v4 .radiance-simple-bar button:hover:not(:disabled) { color: ${T.text}; background: rgba(255,255,255,.06); }
.radiance-v4 .radiance-simple-bar button.is-active { background: ${T.text}; color: ${T.canvas}; }
.radiance-v4 .radiance-simple-bar .rsb-group { border-radius: 10px; padding: 3px; gap: 2px; }
.radiance-v4 .rv4-transport { display: flex; align-items: center; gap: 2px; flex-shrink: 0; }
.radiance-v4 .rv4-transport button { width: 34px; padding: 0; display: inline-flex; align-items: center; justify-content: center; }
.radiance-v4 .radiance-simple-bar .rsb-play, .radiance-v4 .radiance-simple-bar .rsb-play.is-active {
  width: 38px; height: 38px; border-radius: 19px; background: ${T.text}; color: ${T.canvas}; }
.radiance-v4 .radiance-simple-bar .rsb-play:hover:not(:disabled) { background: #ffffff; color: ${T.canvas}; }
.radiance-v4 .rv4-tc { font: 500 13px ${T.mono}; color: ${T.text}; min-width: 96px; }
.radiance-v4 .radiance-simple-bar .rsb-frame { font: 400 12px ${T.mono}; color: ${T.muted}; text-align: right; min-width: 80px; }
.radiance-v4 .radiance-simple-bar .radiance-cache-marks { flex: 1 1 auto; }
.radiance-v4 .radiance-simple-bar .rsb-scrub { width: 100%; accent-color: var(--rv4-accent); }
.radiance-v4 .rv4-sep { width: 1px; height: 22px; background: var(--rv4-line); flex-shrink: 0; }
.radiance-v4 .rv4-label { font: 700 10px ${T.font}; letter-spacing: .14em; color: ${T.muted}; text-transform: uppercase; flex-shrink: 0; }
.radiance-v4 .radiance-simple-bar .rsb-b { max-width: 180px; padding: 0 10px; border: 1px solid rgba(255,255,255,.12); }
.radiance-v4 .rv4-ev { display: flex; align-items: center; gap: 2px; flex-shrink: 0; }
.radiance-v4 .rv4-ev button { width: 30px; padding: 0; display: inline-flex; align-items: center; justify-content: center; }
.radiance-v4 .rv4-ev-value { font: 400 12px ${T.mono}; color: ${T.text}; min-width: 64px; text-align: center; }
.radiance-v4 .rv4-ev-value span { color: ${T.muted}; }
.radiance-v4 .rv4-ev.is-on .rv4-ev-value { color: ${T.warn}; }
.radiance-v4 .rv4-more { position: relative; flex-shrink: 0; }
.radiance-v4 .rv4-more > button { width: 34px; padding: 0; display: inline-flex; align-items: center; justify-content: center; }
.radiance-v4 .rv4-more-pop { display: none; position: absolute; right: 0; bottom: calc(100% + 10px); z-index: 50;
  flex-direction: column; align-items: stretch; gap: 8px; min-width: 240px; padding: 10px; border-radius: 12px;
  background: ${T.raised}; border: 1px solid rgba(255,255,255,.12); box-shadow: 0 16px 40px rgba(0,0,0,.55); }
.radiance-v4 .rv4-more.is-open .rv4-more-pop { display: flex; }
.radiance-v4 .rv4-more-pop .rsb-group { flex-wrap: wrap; }
.radiance-v4 .rv4-more-pop > button { justify-content: flex-start; padding: 0 10px; text-align: left; }
/* The label on the picture while the viewer f-stop is not zero. */
.radiance-v4 .rv4-ev-badge { position: absolute; left: 14px; bottom: 14px; z-index: 30; display: none; align-items: center; gap: 6px;
  height: 26px; padding: 0 10px; border-radius: 7px; background: rgba(8,7,11,.82); border: 1px solid rgba(255,214,10,.5);
  font: 600 11px ${T.font}; color: ${T.warn}; pointer-events: none; }
.radiance-v4 .rv4-ev-badge.is-on { display: inline-flex; }
/* Input menu: the node's settings, in the header. */
.radiance-v4 .rv4-input { position: relative; flex-shrink: 0; }
.radiance-v4 .rv4-input > button { height: 32px; padding: 0 10px; border-radius: 9px; border: 1px solid rgba(255,255,255,.12);
  background: transparent; color: ${T.text}; font: 600 12px ${T.font}; display: inline-flex; align-items: center; gap: 7px; cursor: pointer; white-space: nowrap; }
.radiance-v4 .rv4-input > button:hover, .radiance-v4 .rv4-input.is-open > button { background: rgba(255,255,255,.08); border-color: #4a4a55; }
.radiance-v4 .rv4-input > button span { color: ${T.soft}; }
.radiance-v4 .rv4-input-pop { display: none; position: absolute; top: 40px; right: 0; z-index: 60; width: 340px; box-sizing: border-box;
  padding: 14px; border-radius: 14px; background: ${T.raised}; border: 1px solid ${T.line}; box-shadow: 0 18px 50px rgba(0,0,0,.6);
  flex-direction: column; gap: 14px; font: 12px ${T.font}; color: ${T.text}; white-space: normal; }
.radiance-v4 .rv4-input.is-open .rv4-input-pop { display: flex; }
.radiance-v4 .rv4-ip-head { display: flex; justify-content: space-between; align-items: baseline; }
.radiance-v4 .rv4-ip-head b { font-size: 13px; font-weight: 800; }
.radiance-v4 .rv4-ip-head span, .radiance-v4 .rv4-ip-hint, .radiance-v4 .rv4-ip-foot { font-size: 11px; color: ${T.muted}; line-height: 1.4; }
.radiance-v4 .rv4-ip-field { display: flex; flex-direction: column; gap: 6px; }
.radiance-v4 .rv4-ip-label { font-size: 11px; font-weight: 700; letter-spacing: .1em; text-transform: uppercase; color: ${T.muted}; }
.radiance-v4 .rv4-ip-field select, .radiance-v4 .rv4-ip-field input[type="number"] { height: 34px; border-radius: 8px; border: 1px solid ${T.line};
  background: ${T.control}; color: ${T.text}; font: 12px ${T.font}; padding: 0 10px; box-sizing: border-box; width: 100%; }
.radiance-v4 .rv4-ip-field input[type="number"] { font-family: ${T.mono}; }
.radiance-v4 .rv4-ip-row { display: flex; align-items: center; justify-content: space-between; gap: 12px; }
.radiance-v4 .rv4-switch { width: 44px; height: 26px; flex-shrink: 0; border-radius: 13px; border: 0; padding: 3px; background: #3a3a42;
  display: flex; justify-content: flex-start; cursor: pointer; }
.radiance-v4 .rv4-switch[aria-checked="true"] { background: ${T.accent}; justify-content: flex-end; }
.radiance-v4 .rv4-switch::after { content: ''; width: 20px; height: 20px; border-radius: 10px; background: ${T.text}; display: block; }
.radiance-v4 .rv4-switch[aria-checked="true"]::after { background: ${T.accentInk}; }
.radiance-v4 .rv4-ip-foot { border-top: 1px solid ${T.line}; padding-top: 10px; }
.radiance-v4 .rv4-ip-field [disabled], .radiance-v4 .rv4-switch[disabled] { opacity: .5; cursor: default; }
/* Narrow nodes: keep the picture and transport; drop the words first. */
.radiance-v4.rv4-narrow .rv4-shot-meta, .radiance-v4.rv4-narrow .rv4-label,
.radiance-v4.rv4-narrow .radiance-pro-version, .radiance-v4.rv4-narrow .rv4-brand-word { display: none; }
.radiance-v4.rv4-narrow .rv4-view { max-width: 130px; }
.radiance-v4.radiance-pro-container.radiance-v4 .radiance-pro-brand { min-width: 0; }
.radiance-v4.rv4-narrow .radiance-simple-bar .rv4-sep, .radiance-v4.rv4-narrow .radiance-simple-bar .radiance-mute,
.radiance-v4.rv4-narrow .rv4-zoom button:not(:first-child) { display: none; }
.radiance-v4.rv4-narrow .radiance-simple-bar [aria-label="Compare"] button { padding: 0 8px; }
.radiance-v4.rv4-narrow .radiance-simple-bar .rsb-b { max-width: 96px; }
.radiance-v4.rv4-narrow .rv4-ev-value { min-width: 52px; }
.radiance-v4.rv4-tiny .rv4-ev, .radiance-v4.rv4-tiny .rsb-b, .radiance-v4.rv4-tiny .rv4-shot { display: none; }
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
    b.textContent = label;
    if (title) { b.title = title; b.setAttribute('aria-label', title); }
    if (onClick) b.onclick = onClick;
    return b;
}

/**
 * Apply the shell to a viewer whose UI and Simple bar exist. Returns
 * { sync, destroy }. Safe to call once; a second call returns the first shell.
 */
export function installShell(v) {
    if (v._shell) return v._shell;
    if (!document.getElementById('radiance-v4-shell')) {
        const st = document.createElement('style');
        st.id = 'radiance-v4-shell';
        st.textContent = shellCSS();
        document.head.appendChild(st);
    }
    const root = v.container;
    root.classList.add('radiance-v4');
    const ctl = new AbortController();
    v._listenerSignal?.addEventListener?.('abort', () => ctl.abort(), { signal: ctl.signal });
    const bar = v.proMenuBar;

    // ── Brand: the Radiance icon and wordmark.
    const brand = bar?.querySelector('.radiance-pro-brand');
    if (brand) {
        const mark = brand.querySelector('.radiance-pro-mark');
        if (mark) { mark.classList.add('rv4-icon'); mark.innerHTML = RADIANCE_ICON_SVG; }
        const word = [...brand.children].find((c) => c.tagName === 'SPAN' && !c.className);
        if (word) { word.className = 'rv4-brand-word'; word.innerHTML = 'RADIANCE <span class="rv4-accent">VIEWER</span>'; }
    }

    // ── Simple | Full.
    const mode = el('div', 'rv4-seg rv4-mode');
    mode.setAttribute('role', 'group');
    mode.setAttribute('aria-label', 'Viewer mode');
    const simpleBtn = button('', 'Simple', 'Simple: picture, compare and playback', () => v.setWorkspace('review'));
    const fullBtn = button('', 'Full', 'Full: Color, Inspect and QC workspaces with every panel',
        () => v.setWorkspace(fullWorkspaceFor(v._lastFullWorkspace), { grow: true }));
    mode.append(simpleBtn, fullBtn);
    if (brand) brand.after(mode); else bar?.prepend(mode);

    // ── Workspace tabs: the existing switch, minus Review (that is Simple).
    const sw = bar?.querySelector('.radiance-mode-switch');
    if (sw) {
        sw.setAttribute('aria-label', 'Workspace');
        const fullIds = new Set(fullWorkspaces().map((w) => w.id));
        v._modeBtns?.forEach((b, id) => { if (!fullIds.has(id)) b.setAttribute('data-rv4-hidden', ''); });
        mode.after(sw);
    }

    // ── Simple header: shot, view transform, export.
    const shot = el('div', 'rv4-simple-only rv4-shot');
    const shotName = el('span', 'rv4-shot-name');
    const shotMeta = el('span', 'rv4-shot-meta');
    shot.append(shotName, shotMeta);
    const spacer = el('div', 'rv4-spacer');
    const actions = el('div', 'rv4-simple-only rv4-actions');
    const view = document.createElement('select');
    view.className = 'rv4-view';
    view.title = 'View transform, display only';
    view.setAttribute('aria-label', 'View transform');
    view.onchange = () => v.setViewMode?.(view.value);
    const exportBtn = button('rv4-primary', 'Export', 'Export: snapshot, EXR, grade and Deliver',
        () => v.showExportMenu?.({ target: exportBtn }));
    actions.append(view, exportBtn);
    // ── Input: the node's settings (they used to be widget rows above the viewer).
    const input = el('div', 'rv4-input');
    const inputBtn = button('', '', 'Input settings: input space, float precision, playback rate, exposure brackets');
    inputBtn.setAttribute('aria-haspopup', 'dialog');
    inputBtn.setAttribute('aria-expanded', 'false');
    const inputPop = el('div', 'rv4-input-pop');
    inputPop.setAttribute('role', 'dialog');
    inputPop.setAttribute('aria-label', 'Input settings');
    input.append(inputBtn, inputPop);
    const uid = `rv4-${Math.random().toString(36).slice(2, 8)}`;
    const buildInputPop = () => {
        const head = el('div', 'rv4-ip-head', '<b>Input</b><span>Saved with this node</span>');
        const parts = [head];
        for (const st of inputSettingsState(v.node)) {
            if (!st.present) continue;
            const id = `${uid}-${st.name}`;
            const note = st.linked ? 'Set by a link into the node.' : st.hint;
            if (st.kind === 'switch') {
                const row = el('div', 'rv4-ip-row');
                const txt = el('div');
                const b = el('div'); b.style.cssText = 'font-weight:700'; b.textContent = st.label;
                const h = el('div', 'rv4-ip-hint'); h.textContent = note;
                txt.append(b, h);
                const sw = button('rv4-switch', '', st.label);
                sw.setAttribute('role', 'switch');
                sw.setAttribute('aria-checked', String(!!st.value));
                sw.disabled = st.linked;
                sw.onclick = () => { if (setInputSetting(v.node, st.name, !st.value)) shell.syncInput(true); };
                row.append(txt, sw);
                parts.push(row);
                continue;
            }
            const field = el('div', 'rv4-ip-field');
            const label = el('label', 'rv4-ip-label'); label.textContent = st.label; label.htmlFor = id;
            field.append(label);
            if (st.kind === 'segmented' && st.options) {
                label.removeAttribute('for');
                const seg = el('div', 'rv4-seg');
                seg.setAttribute('role', 'group');
                seg.setAttribute('aria-label', st.label);
                seg.style.alignSelf = 'flex-start';
                for (const opt of st.options) {
                    const o = button(st.value === opt ? 'is-active' : '', opt, '', () => {
                        if (setInputSetting(v.node, st.name, opt)) shell.syncInput(true);
                    });
                    o.removeAttribute('aria-label');
                    o.setAttribute('aria-pressed', String(st.value === opt));
                    o.disabled = st.linked;
                    seg.append(o);
                }
                field.append(seg);
            } else if (st.kind === 'combo' && st.options) {
                const sel = document.createElement('select');
                sel.id = id;
                for (const opt of st.options) { const o = document.createElement('option'); o.value = o.textContent = opt; sel.append(o); }
                sel.value = st.value;
                sel.disabled = st.linked;
                sel.onchange = () => { if (setInputSetting(v.node, st.name, sel.value)) shell.syncInput(false); };
                field.append(sel);
            } else {
                const num = document.createElement('input');
                num.type = 'number'; num.id = id; num.step = '0.001'; num.min = '0';
                num.value = Number(st.value ?? 0).toFixed(3);
                num.disabled = st.linked;
                num.onchange = () => { if (setInputSetting(v.node, st.name, num.value)) shell.syncInput(false); };
                field.append(num);
            }
            const h = el('div', 'rv4-ip-hint'); h.textContent = note;
            field.append(h);
            parts.push(field);
        }
        parts.push(el('div', 'rv4-ip-foot', 'Changes apply on the next run.'));
        inputPop.replaceChildren(...parts);
    };
    const setInputOpen = (on) => {
        input.classList.toggle('is-open', on);
        inputBtn.setAttribute('aria-expanded', String(on));
        if (on) buildInputPop();
    };
    inputBtn.onclick = (e) => { e.stopPropagation(); setInputOpen(!input.classList.contains('is-open')); };
    document.addEventListener('pointerdown', (e) => { if (!input.contains(e.target)) setInputOpen(false); },
        { capture: true, signal: ctl.signal });
    inputPop.addEventListener('keydown', (e) => { if (e.key === 'Escape') { setInputOpen(false); inputBtn.focus(); } }, { signal: ctl.signal });

    const menuItems = bar?.querySelector('.radiance-pro-menu-items');
    (sw || mode).after(shot);
    shot.after(spacer);
    if (menuItems) spacer.after(menuItems);
    (menuItems || spacer).after(input);
    input.after(actions);

    // ── Simple dock: two rows, More for the rest.
    const sb = v.simpleBar;
    let tc = null, evWrap = null, evValue = null, more = null;
    if (sb) {
        const scrubWrap = v._sbScrub?.closest('.radiance-cache-marks') || v._sbScrub;
        const transport = v._sbPlay?.parentElement;
        const cmp = [...sb.querySelectorAll('.rsb-group')].find((g) => g.getAttribute('aria-label') === 'Compare');
        const review = [...sb.querySelectorAll('.rsb-group')].find((g) => g.getAttribute('aria-label') === 'Review');
        const fit = [...sb.children].find((c) => c.tagName === 'BUTTON' && c.textContent === 'Fit');

        tc = el('span', 'rv4-tc');
        tc.setAttribute('aria-label', 'Timecode');
        const row1 = el('div', 'rv4-row');
        row1.append(tc);
        if (scrubWrap) row1.append(scrubWrap);
        if (v._sbFrame) row1.append(v._sbFrame);

        if (transport) {
            transport.classList.remove('rsb-group');
            transport.classList.add('rv4-transport');
            if (v._sbPrev) v._sbPrev.innerHTML = svgIcon('prev', 15);
            if (v._sbNext) v._sbNext.innerHTML = svgIcon('next', 15);
        }

        evWrap = el('div', 'rv4-ev');
        evWrap.setAttribute('role', 'group');
        evWrap.setAttribute('aria-label', 'Viewer f-stop, preview only');
        const evDown = button('', '', 'Viewer f-stop down half a stop, preview only',
            () => v.setViewExposure?.((v.viewExposure || 0) - 0.5));
        evDown.innerHTML = svgIcon('minus', 14);
        evValue = el('span', 'rv4-ev-value');
        const evUp = button('', '', 'Viewer f-stop up half a stop, preview only',
            () => v.setViewExposure?.((v.viewExposure || 0) + 0.5));
        evUp.innerHTML = svgIcon('plus', 14);
        evValue.title = 'Viewer f-stop: display only, never baked into Deliver. Double-click to reset.';
        evValue.ondblclick = () => v.setViewExposure?.(0);
        evWrap.append(evDown, evValue, evUp);

        more = el('div', 'rv4-more');
        const moreBtn = button('', '', 'More: B fit, grab still, onion skin, bookmarks, notes, QC and report');
        moreBtn.innerHTML = svgIcon('more', 16);
        moreBtn.setAttribute('aria-haspopup', 'true');
        moreBtn.setAttribute('aria-expanded', 'false');
        const pop = el('div', 'rv4-more-pop');
        pop.setAttribute('role', 'group');
        pop.setAttribute('aria-label', 'More controls');
        DOCK_LAYOUT.more.forEach((k) => { if (v[k]) pop.appendChild(v[k]); });
        if (review) pop.appendChild(review);
        const setOpen = (on) => {
            more.classList.toggle('is-open', on);
            moreBtn.setAttribute('aria-expanded', String(on));
        };
        moreBtn.onclick = (e) => { e.stopPropagation(); setOpen(!more.classList.contains('is-open')); };
        document.addEventListener('pointerdown', (e) => { if (!more.contains(e.target)) setOpen(false); },
            { capture: true, signal: ctl.signal });
        more.append(moreBtn, pop);

        const zoom = el('div', 'rsb-group rv4-zoom');
        zoom.setAttribute('role', 'group');
        zoom.setAttribute('aria-label', 'Zoom');
        if (fit) zoom.appendChild(fit);
        zoom.appendChild(button('', '1:1', 'Actual pixels', () => v.setZoom?.(1)));

        const row2 = el('div', 'rv4-row');
        const cmpLabel = el('span', 'rv4-label', 'Compare');
        row2.append(...[transport, el('span', 'rv4-sep'), cmpLabel, cmp, v._sbBLabel,
            el('span', 'rv4-sep'), evWrap, el('div', 'rv4-spacer'), more, zoom].filter(Boolean));
        sb.replaceChildren(row1, row2);
    }

    // ── The label on the picture while the viewer f-stop is not zero.
    const badge = el('div', 'rv4-ev-badge');
    badge.setAttribute('role', 'status');
    (v.canvasWrapper || root).appendChild(badge);

    // ── Narrow nodes.
    const sizeClasses = () => {
        const w = root.clientWidth || 0;
        root.classList.toggle('rv4-narrow', w > 0 && w < 900);
        root.classList.toggle('rv4-tiny', w > 0 && w < 620);
    };
    let ro = null;
    if (typeof ResizeObserver !== 'undefined') { ro = new ResizeObserver(sizeClasses); ro.observe(root); }

    // ── Mirror the shot line from the Full toolbar's fields.
    const shotFields = [v._proFileName, v._proResolution, v._proDepth, v._proFps].filter(Boolean);
    let mo = null;
    if (typeof MutationObserver !== 'undefined' && shotFields.length) {
        mo = new MutationObserver(() => shell.syncShot());
        shotFields.forEach((n) => mo.observe(n, { childList: true, characterData: true, subtree: true }));
    }

    const shell = {
        syncShot() {
            shotName.textContent = v._proFileName?.textContent || 'No shot loaded';
            const parts = [v._proResolution, v._proDepth, v._proFps]
                .map((n) => (n?.textContent || '').trim()).filter((t) => t && t !== '— x —' && t !== '-');
            shotMeta.textContent = parts.join(' · ');
        },
        syncView() {
            const src = v._viewSelect;
            const opts = src ? [...src.options].map((o) => [o.value, o.textContent])
                : (v.constructor?.VIEW_MODES || []);
            view.replaceChildren(...opts.map(([value, label]) => {
                const o = document.createElement('option');
                o.value = value; o.textContent = label;
                return o;
            }));
            view.value = v.viewMode || 'auto';
        },
        syncTransport() {
            if (tc) tc.textContent = smpteTimecode(v.currentFrame || 0, v.playbackFps || 24);
            if (v._sbPlay) {
                v._sbPlay.innerHTML = svgIcon(v.isPlaying ? 'pause' : 'play', 15);
                v._sbPlay.setAttribute('aria-label', v.isPlaying ? 'Pause' : 'Play');
            }
        },
        syncInput(rebuild) {
            const st = inputSettingsState(v.node).find((x) => x.name === 'input_space');
            inputBtn.innerHTML = '<svg viewBox="0 0 24 24" width="15" height="15" fill="none" stroke="currentColor" stroke-width="1.8" '
                + 'stroke-linecap="round" aria-hidden="true"><path d="M4 7h10M18 7h2M4 17h4M12 17h8"/><circle cx="16" cy="7" r="2"/>'
                + '<circle cx="10" cy="17" r="2"/></svg>';
            inputBtn.append('Input ');
            const val = el('span'); val.textContent = `\u00B7 ${shortSpace(st?.value)}`;
            inputBtn.append(val);
            input.style.display = inputSettingsState(v.node).some((x) => x.present) ? '' : 'none';
            if (rebuild && input.classList.contains('is-open')) buildInputPop();
        },
        syncExposure() {
            const ev = v.viewExposure || 0;
            if (evValue) evValue.innerHTML = `${formatEV(ev)} <span>EV</span>`;
            evWrap?.classList.toggle('is-on', Math.round(ev * 10) !== 0);
            const text = exposureBadge(ev);
            badge.textContent = text || '';
            badge.classList.toggle('is-on', !!text);
        },
        sync() {
            const m = shellMode(v.uiMode);
            simpleBtn.classList.toggle('is-active', m === 'simple');
            fullBtn.classList.toggle('is-active', m === 'full');
            simpleBtn.setAttribute('aria-pressed', String(m === 'simple'));
            fullBtn.setAttribute('aria-pressed', String(m === 'full'));
            if (m === 'full' && v.workspace) v._lastFullWorkspace = v.workspace;
            shell.syncShot(); shell.syncView(); shell.syncTransport(); shell.syncExposure(); shell.syncInput(false);
            sizeClasses();
        },
        destroy() {
            ctl.abort();
            ro?.disconnect();
            mo?.disconnect();
            badge.remove();
        },
    };
    v._shell = shell;
    shell.sync();
    return shell;
}
