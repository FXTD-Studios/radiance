/**
 * Viewer workspaces: Review, Color, Inspect, Compare (QC when it exists).
 *
 * Pure data and functions, no DOM, so the rules are testable in Node.
 * radiance_viewer.js holds the DOM and calls in here.
 *
 * Compatibility contract:
 *  - `radiance_viewer_mode` on the node stays 'simple' | 'advanced', exactly as
 *    before, so saved graphs load and older Radiance versions still read it.
 *  - `radiance_viewer_workspace` (new) records which workspace the person was in.
 *    It is only read when it agrees with the layout; a missing or unknown value
 *    falls back to the layout's default workspace.
 *  - Review is the old Simple layout. Color, Inspect and Compare all use the old
 *    Advanced layout (every panel) and differ only in what they focus first.
 *    Hiding panels per workspace is a later step.
 */

export const LAYOUT_SIMPLE = 'simple';
export const LAYOUT_ADVANCED = 'advanced';

/**
 * `available: false` workspaces are defined but never offered: QC has no
 * implementation yet, and a button that opens nothing would be a phantom feature.
 */
export const WORKSPACES = Object.freeze([
    { id: 'review', label: 'Review', layout: LAYOUT_SIMPLE, available: true,
      tip: 'Picture, compare and playback only' },
    { id: 'color', label: 'Color', layout: LAYOUT_ADVANCED, available: true, tab: 'primaries',
      tip: 'Grade, curves, effects and scopes' },
    { id: 'inspect', label: 'Inspect', layout: LAYOUT_ADVANCED, available: true, tab: 'probe',
      tip: 'Pixel probe, channels and values' },
    { id: 'compare', label: 'Compare', layout: LAYOUT_ADVANCED, available: true,
      tip: 'A/B, wipe, difference and blink against a reference' },
    { id: 'qc', label: 'QC', layout: LAYOUT_ADVANCED, available: false,
      tip: 'Measured checks (not implemented yet)' },
]);

const BY_ID = new Map(WORKSPACES.map((w) => [w.id, w]));

export function workspaceById(id) {
    return BY_ID.get(id) || null;
}

/** The workspaces the switcher shows. */
export function availableWorkspaces() {
    return WORKSPACES.filter((w) => w.available);
}

/** Old two-value layout name from anything a saved graph might hold. */
export function normalizeLayout(value) {
    return value === LAYOUT_ADVANCED ? LAYOUT_ADVANCED : LAYOUT_SIMPLE;
}

/** The workspace a layout opens in when none was recorded. */
export function defaultWorkspaceFor(layout) {
    return normalizeLayout(layout) === LAYOUT_ADVANCED ? 'color' : 'review';
}

/**
 * Resolve what a node opens in from its saved properties.
 * Returns { layout, workspace }. A recorded workspace is honoured only if it is
 * available and belongs to the saved layout; otherwise the layout's default is
 * used, so a hand-edited or stale value can never open a mismatched UI.
 */
export function resolveSaved(props) {
    const p = props || {};
    const layout = normalizeLayout(p.radiance_viewer_mode);
    const ws = workspaceById(p.radiance_viewer_workspace);
    if (ws && ws.available && ws.layout === layout) return { layout, workspace: ws.id };
    return { layout, workspace: defaultWorkspaceFor(layout) };
}

/** Properties to store for a workspace. The layout key keeps its old meaning. */
export function propsFor(workspaceId) {
    const ws = workspaceById(workspaceId);
    if (!ws || !ws.available) return null;
    return { radiance_viewer_mode: ws.layout, radiance_viewer_workspace: ws.id };
}

/**
 * A suggestion for what the person may want, from facts about the loaded media.
 * Never applied automatically: the viewer shows it as a hint. Returns null when
 * there is nothing worth saying.
 *   facts: { frames, hasB, isHDR }
 */
export function suggestWorkspace(facts, current) {
    const f = facts || {};
    if (f.hasB && current !== 'compare') {
        return { workspace: 'compare', reason: 'A reference (B) is connected.' };
    }
    return null;
}
