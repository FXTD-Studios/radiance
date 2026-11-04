/**
 * The Radiance welcome: what it shows, and when. Pure data and functions, no
 * DOM, so the rules are testable in Node (radiance_welcome.js draws it).
 *
 *  - First install (no version recorded): the tour, from its first page.
 *  - A new Radiance version: one page, What's new, with the tour a click away.
 *  - The same version, or "Don't show at startup" ticked: nothing.
 *  Help > Radiance Tour opens the tour whatever these say.
 */

export const TOUR_PAGES = Object.freeze(['welcome', 'workflows', 'viewer', 'backdrops', 'gizmos', 'learn']);

export const LINKS = Object.freeze({
    radiance: 'https://www.fxtdstudios.com/radiance-comfyui.html',
    issues: 'https://github.com/FXTD-Studios/radiance/issues',
    studio: 'https://www.fxtdstudios.com',
});

/**
 * What changed in each release, newest first, for the update page. Add the
 * release's entry when its version is bumped: an update to a version with no
 * entry shows the version change and the release-notes link only.
 */
export const WHATS_NEW = Object.freeze([
    {
        version: '4.0.0',
        items: [
            'RUDRA ACES 2: SDR to HDR with nothing to set up, and several frames at once on long clips.',
            'A new Viewer: Review, Color and Inspect, with a Project colour sheet and scopes in nits.',
            '39 workflows, 25 of them API, each in the Templates menu and with a card in the Workflow Library.',
            'FLUX 3 Klein, and one Prompt node that holds several prompts.',
        ],
    },
]);

/** -1, 0 or 1, comparing dotted versions ("4.0.10" > "4.0.9"); text after a "-" is ignored. */
export function compareVersions(a, b) {
    const parts = (v) => String(v || '').split('-')[0].split('.').map((x) => parseInt(x, 10) || 0);
    const pa = parts(a), pb = parts(b);
    for (let i = 0; i < Math.max(pa.length, pb.length); i++) {
        const d = (pa[i] || 0) - (pb[i] || 0);
        if (d) return d > 0 ? 1 : -1;
    }
    return 0;
}

/**
 * What to open at startup: 'tour', 'update' or null.
 *   lastVersion: the version the welcome last closed on ('' on a first install)
 *   current:     RADIANCE_VERSION
 *   showAtStartup: the "Show the Radiance tour at startup" setting
 */
export function startupView({ lastVersion, current, showAtStartup = true }) {
    if (!showAtStartup) return null;
    if (!lastVersion) return 'tour';
    return compareVersions(current, lastVersion) > 0 ? 'update' : null;
}

/** The update page's notes: the current version's entry, else none. */
export function notesFor(version) {
    const hit = WHATS_NEW.find((e) => compareVersions(e.version, version) === 0);
    return hit ? hit.items.slice() : [];
}

/** The page after/before `page` in the tour, clamped to its ends. */
export function stepPage(page, delta) {
    const i = TOUR_PAGES.indexOf(page);
    const j = Math.max(0, Math.min(TOUR_PAGES.length - 1, (i < 0 ? 0 : i) + delta));
    return TOUR_PAGES[j];
}
