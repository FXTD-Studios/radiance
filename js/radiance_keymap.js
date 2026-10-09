/**
 * The Viewer's keyboard map: one table for the key handler, the help overlay,
 * the menu bar and the tooltips.
 *
 * The keys used to be written down four times (handleKey, the help overlay,
 * the menus and the timeline tooltips) and the copies had drifted: help said
 * H was Histogram while H toggled help, the menus showed 0, S and A/B against
 * actions those keys did not run, 200% (2) and Full Screen (F11) were not
 * bound at all, and Shift+V switched both the timeline track and the
 * vectorscope. Every label is now made from the binding it describes, so a
 * label without a binding cannot be written.
 *
 * Keys match on KeyboardEvent.code, the physical key, so a shortcut does not
 * change with the characters a modifier produces: on macOS Option+X types
 * "≈" and the old e.key test never matched it. "ctrl" means Ctrl or Cmd.
 *
 * No ComfyUI imports, so this loads in Node for the tests.
 */

/**
 * One row per action. Fields:
 *   id       the action the Viewer runs (RadianceViewer._runKeyAction)
 *   keys     bindings: { code, shift?, alt?, ctrl?, arg? }; shift: 'any'
 *            matches with or without Shift. An action with no keys is a
 *            menu item only.
 *   label    the name in the menus and the help
 *   help     the help overlay's wording, when it differs from label
 *   section  the help overlay heading
 *   menu     the top-level menu it appears in, if any; group splits a menu
 *            with a separator where the group changes
 *   keyText  how the help writes a run of keys (the numpad presets)
 */
export const KEYMAP = Object.freeze([
    // ── File ──
    { id: 'file.export', keys: [], label: 'Snapshot / Export…', menu: 'File', group: 'out' },
    { id: 'file.savePNG', keys: [], label: 'Save PNG (Result)', menu: 'File', group: 'out' },
    { id: 'file.exportCDL', keys: [], label: 'Export Grade as .CDL', menu: 'File', group: 'grade' },
    { id: 'file.exportCUBE', keys: [], label: 'Export Grade as .CUBE', menu: 'File', group: 'grade' },
    { id: 'file.pinB', keys: [], label: 'Pin Current Frame as B', menu: 'File', group: 'pin' },

    // ── Edit ──
    { id: 'edit.undo', keys: [{ code: 'KeyZ', ctrl: true }], label: 'Undo', section: 'General', menu: 'Edit', group: 'undo' },
    { id: 'edit.redo', keys: [{ code: 'KeyY', ctrl: true }, { code: 'KeyZ', ctrl: true, shift: true }],
        label: 'Redo', section: 'General', menu: 'Edit', group: 'undo' },
    { id: 'edit.resetGrade', keys: [], label: 'Reset Grade Tab', menu: 'Edit', group: 'reset' },
    { id: 'edit.resetAll', keys: [], label: 'Reset Entire Grade', menu: 'Edit', group: 'reset' },

    // ── View ──
    { id: 'view.fit', keys: [{ code: 'KeyF' }], label: 'Fit', help: 'Fit to view', section: 'View', menu: 'View', group: 'zoom' },
    { id: 'view.zoom100', keys: [{ code: 'Digit1' }], label: '100%', help: 'Zoom 100% (1:1 pixels)', section: 'View', menu: 'View', group: 'zoom' },
    { id: 'view.zoom200', keys: [{ code: 'Digit2' }], label: '200%', help: 'Zoom 200%', section: 'View', menu: 'View', group: 'zoom' },
    {
        id: 'view.zoomPreset', section: 'View', label: 'Zoom preset',
        help: 'Zoom: 0 fits, 1 to 8 are 100% to 800%', keyText: 'Numpad 0-8',
        keys: [0, 1, 2, 3, 4, 5, 6, 7, 8].map((n) => ({ code: `Numpad${n}`, arg: n })),
    },
    { id: 'view.safeAreas', keys: [{ code: 'KeyS' }], label: 'Safe Areas', help: 'Cycle safe areas', section: 'View', menu: 'View', group: 'guides' },
    { id: 'view.grid', keys: [{ code: 'KeyG', shift: true }], label: 'Grid', help: 'Cycle grid modes', section: 'View', menu: 'View', group: 'guides' },
    { id: 'view.pixelFilter', keys: [{ code: 'KeyN' }], label: 'Nearest / Linear Filter', help: 'Nearest or linear magnification', section: 'View', menu: 'View', group: 'guides' },
    { id: 'view.loupe', keys: [{ code: 'KeyQ' }], label: 'Pixel Loupe', section: 'View', menu: 'View', group: 'guides' },
    { id: 'view.exposureUp', keys: [{ code: 'Equal', shift: 'any' }], label: 'Viewer f-stop +½', help: 'Viewer f-stop up half a stop (display only)', section: 'View', menu: 'View', group: 'viewer' },
    { id: 'view.exposureDown', keys: [{ code: 'Minus', shift: 'any' }], label: 'Viewer f-stop −½', help: 'Viewer f-stop down half a stop (display only)', section: 'View', menu: 'View', group: 'viewer' },
    { id: 'view.resetViewer', keys: [{ code: 'Digit0' }], label: 'Reset Viewer f-stop and γ', section: 'View', menu: 'View', group: 'viewer' },
    { id: 'view.compare', keys: [{ code: 'KeyX' }], label: 'Compare (Wipe A/B)', help: 'Compare on or off (wipe)', section: 'View', menu: 'View', group: 'compare' },

    // ── Channels (help only; the rail has the buttons) ──
    { id: 'channel.r', keys: [{ code: 'KeyR' }], label: 'Red channel', section: 'Channels' },
    { id: 'channel.g', keys: [{ code: 'KeyG' }], label: 'Green channel', section: 'Channels' },
    { id: 'channel.b', keys: [{ code: 'KeyB' }], label: 'Blue channel', section: 'Channels' },
    { id: 'channel.luma', keys: [{ code: 'KeyY' }], label: 'Luma', section: 'Channels' },
    { id: 'channel.rgb', keys: [{ code: 'KeyC' }], label: 'RGB (colour)', section: 'Channels' },
    { id: 'channel.alpha', keys: [{ code: 'KeyA' }], label: 'Alpha on or off', section: 'Channels' },

    // ── Color ──
    { id: 'panel.grade', keys: [{ code: 'Digit2', alt: true }], label: 'Grade', help: 'Panel: Grade', section: 'Panel', menu: 'Color', group: 'grade' },
    { id: 'panel.curves', keys: [], label: 'Curves', menu: 'Color', group: 'grade' },
    { id: 'panel.masks', keys: [{ code: 'Digit3', alt: true }], label: 'Masks & Qualifiers', help: 'Panel: Masks & Qualifiers', section: 'Panel', menu: 'Color', group: 'grade' },
    { id: 'panel.presets', keys: [], label: 'Grade Presets', menu: 'Color', group: 'grade' },
    { id: 'panel.ocio', keys: [], label: 'OCIO Config…', menu: 'Color', group: 'grade' },
    { id: 'grade.exposureUp', keys: [{ code: 'NumpadAdd' }], label: 'Grade exposure +¼', section: 'Grade' },
    { id: 'grade.exposureDown', keys: [{ code: 'NumpadSubtract' }], label: 'Grade exposure −¼', section: 'Grade' },
    { id: 'view.falseColor', keys: [{ code: 'KeyE' }], label: 'False Color', help: 'False colour (exposure)', section: 'Analysis', menu: 'Color', group: 'overlay' },
    { id: 'view.zebra', keys: [], label: 'Zebra', menu: 'Color', group: 'overlay' },
    { id: 'view.heatmap', keys: [], label: 'HDR Heatmap', menu: 'Color', group: 'overlay' },
    { id: 'view.focusPeaking', keys: [{ code: 'KeyK', shift: true }], label: 'Focus Peaking', section: 'Analysis', menu: 'Color', group: 'overlay' },

    // ── Analysis ──
    { id: 'panel.scopes', keys: [{ code: 'Digit5', alt: true }], label: 'Scopes', help: 'Panel: Scopes', section: 'Panel', menu: 'Tools', group: 'panels' },
    { id: 'panel.analysis', keys: [{ code: 'Digit6', alt: true }], label: 'Analysis', help: 'Panel: Analysis', section: 'Panel', menu: 'Tools', group: 'panels' },
    { id: 'panel.effects', keys: [{ code: 'Digit4', alt: true }], label: 'Effects + Depth', help: 'Panel: Effects', section: 'Panel', menu: 'Tools', group: 'panels' },
    { id: 'panel.inspector', keys: [{ code: 'Digit1', alt: true }], label: 'Inspector', help: 'Panel: Inspector', section: 'Panel', menu: 'Tools', group: 'panels' },
    { id: 'scope.waveform', keys: [{ code: 'KeyW' }], label: 'Waveform (overlay)', help: 'Waveform over the picture', section: 'Analysis', menu: 'Tools', group: 'scopes' },
    { id: 'scope.parade', keys: [{ code: 'KeyM' }], label: 'Waveform Parade', help: 'Waveform as RGB parade', section: 'Analysis', menu: 'Tools', group: 'scopes' },
    { id: 'scope.vectorscope', keys: [{ code: 'KeyV' }], label: 'Vectorscope (overlay)', help: 'Vectorscope over the picture', section: 'Analysis', menu: 'Tools', group: 'scopes' },
    { id: 'view.metadata', keys: [], label: 'Metadata Overlay', menu: 'Tools', group: 'overlays' },
    { id: 'view.depth', keys: [{ code: 'KeyZ' }], label: 'Depth Overlay', help: 'Z-depth on or off', section: 'Analysis', menu: 'Tools', group: 'overlays' },
    { id: 'tools.precision', keys: [{ code: 'KeyB', alt: true }], label: 'Pipeline Bit Depth', help: 'Cycle pipeline bit depth (8 / 16 / 32)', section: 'General', menu: 'Tools', group: 'system' },
    { id: 'tools.terminal', keys: [{ code: 'Backquote', shift: 'any' }], label: 'Terminal', help: 'Terminal on or off', section: 'General', menu: 'Tools', group: 'system' },

    // ── Playback ──
    { id: 'play.toggle', keys: [{ code: 'Space' }], label: 'Play / pause', section: 'Playback' },
    { id: 'play.prev', keys: [{ code: 'ArrowLeft' }], label: 'Previous frame', section: 'Playback' },
    { id: 'play.next', keys: [{ code: 'ArrowRight' }], label: 'Next frame', section: 'Playback' },
    { id: 'play.reverse', keys: [{ code: 'KeyJ' }], label: 'Play reverse (again: faster)', section: 'Playback' },
    { id: 'play.stop', keys: [{ code: 'KeyK' }], label: 'Stop', section: 'Playback' },
    { id: 'play.forward', keys: [{ code: 'KeyL' }], label: 'Play forward (again: faster)', section: 'Playback' },
    { id: 'play.in', keys: [{ code: 'KeyI' }, { code: 'BracketLeft' }], label: 'Set in point', section: 'Playback' },
    { id: 'play.out', keys: [{ code: 'KeyO' }, { code: 'BracketRight' }], label: 'Set out point', section: 'Playback' },
    { id: 'play.clearInOut', keys: [{ code: 'KeyX', alt: true }], label: 'Clear in and out', section: 'Playback' },
    { id: 'play.first', keys: [{ code: 'Home' }], label: 'First frame (in point)', section: 'Playback' },
    { id: 'play.last', keys: [{ code: 'End' }], label: 'Last frame (out point)', section: 'Playback' },

    // ── Timeline tools (Shift, so they never collide with the single keys) ──
    { id: 'tool.select', keys: [{ code: 'KeyA', shift: true }], label: 'Select tool', help: 'Timeline: Select (click to scrub)', section: 'Timeline' },
    { id: 'tool.slip', keys: [{ code: 'KeyS', shift: true }], label: 'Slip tool', help: 'Timeline: Slip a Ref Wipe block (B runs ahead or behind)', section: 'Timeline' },
    { id: 'tool.reference', keys: [{ code: 'KeyR', shift: true }], label: 'Ref Wipe tool', help: 'Timeline: place a Ref Wipe block on V2', section: 'Timeline' },

    // ── Window / General ──
    { id: 'window.panel', keys: [], label: 'Right Panel', menu: 'Window', group: 'layout' },
    { id: 'window.compact', keys: [], label: 'Compact Layout', menu: 'Window', group: 'layout' },
    { id: 'window.fullscreen', keys: [{ code: 'KeyF', shift: true }], label: 'Full Screen', section: 'General', menu: 'Window', group: 'screen' },
    { id: 'general.run', keys: [{ code: 'Enter', shift: true }], label: 'Run workflow', section: 'General' },
    { id: 'general.escape', keys: [{ code: 'Escape' }], label: 'Close help, leave full screen', section: 'General' },
    { id: 'help.toggle', keys: [{ code: 'KeyH' }, { code: 'Slash', shift: true }], label: 'Keyboard Shortcuts', help: 'This help', section: 'General', menu: 'Help', group: 'help' },
]);

/** The menu bar, left to right. */
export const MENU_ORDER = Object.freeze(['File', 'Edit', 'View', 'Color', 'Tools', 'Window', 'Help']);

/** The help overlay's sections, in order. */
export const HELP_SECTIONS = Object.freeze(['View', 'Playback', 'Channels', 'Analysis', 'Grade', 'Panel', 'Timeline', 'General']);

/** Pointer gestures, for the help overlay. They are not keys, so they sit apart. */
export const POINTER_HELP = Object.freeze([
    ['Wheel', 'Zoom about the pointer'],
    ['Ctrl+Wheel / pinch', 'Zoom (trackpad pinch, touch pinch)'],
    ['Two-finger scroll', 'Pan (trackpad)'],
    ['Shift+drag / middle drag', 'Pan (mouse, pen)'],
    ['One-finger drag', 'Pan (touch)'],
    ['Alt+drag', 'Draw an annotation'],
    ['Right-click', 'Store a probe point'],
]);

const BY_ID = new Map(KEYMAP.map((e) => [e.id, e]));

/** The keymap row for an action id. */
export function keyEntry(id) {
    return BY_ID.get(id) || null;
}

const CODE_TEXT = {
    Space: 'Space', ArrowLeft: '←', ArrowRight: '→', ArrowUp: '↑', ArrowDown: '↓',
    Equal: '+', Minus: '−', BracketLeft: '[', BracketRight: ']', Slash: '/', Backquote: '`',
    Escape: 'Esc', Enter: 'Enter', Home: 'Home', End: 'End',
    NumpadAdd: 'Numpad +', NumpadSubtract: 'Numpad −',
};

/** One binding as text: "Shift+K", "Alt+2", "Ctrl+Z", "?" for Shift+/. */
export function keyLabel(key) {
    if (!key) return '';
    const { code } = key;
    if (code === 'Slash' && key.shift === true) return '?';
    let name = CODE_TEXT[code];
    if (!name) {
        if (/^Key[A-Z]$/.test(code)) name = code.slice(3);
        else if (/^Digit\d$/.test(code)) name = code.slice(5);
        else if (/^Numpad\d$/.test(code)) name = `Numpad ${code.slice(6)}`;
        else name = code;
    }
    const mods = [];
    if (key.ctrl) mods.push('Ctrl');
    if (key.alt) mods.push('Alt');
    if (key.shift === true) mods.push('Shift');
    return [...mods, name].join('+');
}

/** Every binding of an action as text, "I or [" style; '' when it has none. */
export function shortcutLabel(id, { all = false } = {}) {
    const e = BY_ID.get(id);
    if (!e || !e.keys.length) return '';
    if (e.keyText) return e.keyText;
    return all ? e.keys.map(keyLabel).join(' or ') : keyLabel(e.keys[0]);
}

/**
 * The physical key for an event. A synthetic KeyboardEvent built with only
 * `key` (as some tests and other extensions dispatch them) has an empty
 * code; for those the US-layout key that types that character stands in.
 */
export function eventCode(e) {
    if (e.code) return e.code;
    const k = e.key || '';
    if (/^[a-z]$/i.test(k)) return `Key${k.toUpperCase()}`;
    if (/^[0-9]$/.test(k)) return `Digit${k}`;
    return ({
        ' ': 'Space', '=': 'Equal', '+': 'Equal', '-': 'Minus', '_': 'Minus',
        '[': 'BracketLeft', ']': 'BracketRight', '/': 'Slash', '?': 'Slash',
        '`': 'Backquote', '~': 'Backquote', Esc: 'Escape',
    })[k] || k;
}

function bindingMatches(key, code, e) {
    if (key.code !== code) return false;
    if (key.shift !== 'any' && !!key.shift !== !!e.shiftKey) return false;
    if (!!key.alt !== !!e.altKey) return false;
    return !!key.ctrl === !!(e.ctrlKey || e.metaKey);
}

/** The action a key event asks for: { entry, key } or null. */
export function matchKey(e) {
    const code = eventCode(e);
    if (!code) return null;
    for (const entry of KEYMAP) {
        for (const key of entry.keys) {
            if (bindingMatches(key, code, e)) return { entry, key };
        }
    }
    return null;
}

/** Input types that take typed text. Range, checkbox, colour and the buttons do not. */
const NON_TEXT_INPUTS = new Set(['range', 'checkbox', 'radio', 'button', 'submit', 'reset', 'color', 'file', 'image', 'hidden']);

/**
 * True when keys belong to the focused element because it takes typed text:
 * a text-type input (number included), a textarea, or contenteditable.
 * A range slider or a select does not count: the viewer's keys still work
 * after the scrubber or a dropdown has been clicked.
 */
export function isTextEntry(target) {
    if (!target) return false;
    if (target.isContentEditable) return true;
    const tag = target.tagName;
    if (tag === 'TEXTAREA') return true;
    if (tag === 'INPUT') return !NON_TEXT_INPUTS.has(String(target.type || 'text').toLowerCase());
    return target.getAttribute?.('role') === 'textbox';
}

/** Keys a focused range slider uses itself (its own step), so they stay with it. */
export const RANGE_KEYS = Object.freeze(new Set(['ArrowLeft', 'ArrowRight', 'ArrowUp', 'ArrowDown', 'Home', 'End', 'PageUp', 'PageDown']));

/** Rows that appear in a menu, in table order. */
export function menuEntries(menu) {
    return KEYMAP.filter((e) => e.menu === menu);
}

/** The help overlay content: [{ section, items: [[keys, text]] }]. */
export function helpSections() {
    return HELP_SECTIONS.map((section) => {
        const items = KEYMAP.filter((e) => e.section === section && e.keys.length)
            .map((e) => [shortcutLabel(e.id, { all: true }), e.help || e.label]);
        // The panel tabs read best in Alt+1 to Alt+6 order.
        if (section === 'Panel') items.sort((a, b) => a[0].localeCompare(b[0]));
        return { section, items };
    }).filter((s) => s.items.length);
}

export default { KEYMAP, MENU_ORDER, matchKey, keyLabel, shortcutLabel, isTextEntry, helpSections, menuEntries };
