/**
 * The Radiance welcome: what it shows, and when. Pure data and functions, no
 * DOM, so the rules are testable in Node (radiance_welcome.js draws it).
 *
 *  - First install (no version recorded): the tour, from its first page.
 *  - A new Radiance version: one page, What's new, with the tour a click away.
 *  - The same version, or "Don't show at startup" ticked: nothing.
 *  Help > Radiance Tour opens the tour whatever these say.
 */

export const TOUR_PAGES = Object.freeze(['welcome', 'setup', 'workflows', 'viewer', 'backdrops', 'gizmos', 'learn']);

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
            '45 workflows, 25 of them API, each in the Templates menu and with a card in Radiance › Workflows.',
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

// ── The "Set up" page: what /radiance/setup/scan found, in a few lines ────────

/** Bytes in decimal GB / MB, as GPU memory and downloads are quoted. */
export function sizeText(n) {
    const v = Number(n) || 0;
    if (v >= 1e9) return `${(v / 1e9).toFixed(v >= 1e10 ? 0 : 1)} GB`;
    if (v >= 1e6) return `${Math.round(v / 1e6)} MB`;
    return v > 0 ? `${Math.max(1, Math.round(v / 1e3))} KB` : '0 MB';
}

/** GPU and RAM sizes in binary GB, as cards are sold (a 24 GB card reads 24 GB). */
export function memoryText(n) {
    const gb = (Number(n) || 0) / 1024 ** 3;
    return gb >= 10 ? `${Math.round(gb)} GB` : gb > 0 ? `${gb.toFixed(1)} GB` : '';
}

/**
 * The tour's summary of a setup scan:
 *   gpu, memory, disk: one line each for this machine
 *   tier: { tone, text } what the GPU can run
 *   rudra: { installed, path } RUDRA ACES 2, the model SDR -> HDR needs
 *   ready: "N of M local workflows are ready to run"
 */
export function setupSummary(scan) {
    const hw = scan?.hardware || {};
    const gpus = hw.gpus || [];
    const g = gpus.find((x) => x.vram) || gpus[0];
    const vram = Math.max(0, ...gpus.map((x) => Number(x.vram) || 0));
    const gb = vram / 1024 ** 3;
    const tier = !vram ? { tone: 'warn', text: 'No CUDA GPU found: SDR to HDR and grading work; generation is very slow.' }
        : gb >= 22 ? { tone: 'ok', text: 'This GPU runs every local workflow, the 22B video models included.' }
            : gb >= 15 ? { tone: 'ok', text: 'This GPU runs the image workflows and the AI fill; big video models offload to RAM.' }
                : gb >= 7.5 ? { tone: 'warn', text: 'This GPU runs SDR to HDR and the Starter; larger models offload to RAM (slow).' }
                    : { tone: 'warn', text: 'This GPU is fine for SDR to HDR and grading; generation will be slow.' };
    const rudra = (scan?.core || [])[0] || scan?.models?.['radiance/RUDRA_ACES_2'] || null;
    const s = scan?.summary || { workflows: 0, ready: 0 };
    return {
        gpu: g ? (g.vram ? `${g.name} · ${memoryText(g.vram)}` : g.name) : 'No GPU found',
        memory: hw.ram ? `${memoryText(hw.ram)} RAM` : '',
        disk: hw.disk_free != null ? `${sizeText(hw.disk_free)} free for models` : '',
        tier,
        rudra: { installed: !!rudra?.installed, path: scan?.paths?.radiance || '' },
        ready: `${s.ready} of ${s.workflows} local workflows are ready to run`,
        downloadsOff: scan ? !scan.downloads_allowed : false,
    };
}
