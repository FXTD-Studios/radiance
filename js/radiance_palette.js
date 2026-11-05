/**
 * One colour code for Radiance, by discipline. Pure data, no DOM.
 *
 *  - `group`: ComfyUI's own group palette (LGraphCanvas.node_colors
 *    groupcolor), so a workflow's groups look the same with or without
 *    Radiance: blue generation, yellow HDR / RUDRA, purple optional
 *    (bypassed) parts, black review.
 *  - `header`: the darker node-title colour of the same family, given to
 *    Radiance nodes only. Other packs' and ComfyUI's own nodes keep theirs.
 *
 * Read by radiance_style.js (node headers), radiance_backdrop.js (Smart
 * Backdrop) and radiance_studio.js (backdrop presets), and by
 * tests/test_palette.py for the shipped workflows' groups.
 */

export const BODY = "#141419";

export const DISCIPLINES = Object.freeze({
    generate: { label: "Generation", group: "#3f789e", header: "#1c3647", title: "◎ AI Generation" },
    hdr: { label: "HDR / RUDRA", group: "#b58b2a", header: "#4a3912", title: "◎ HDR" },
    sdr: { label: "SDR output", group: "#8A8", header: "#2f4a2f", title: "◎ SDR Output" },
    color: { label: "Colour", group: "#8AA", header: "#2b4545", title: "◎ Color Grading" },
    vfx: { label: "VFX", group: "#b06634", header: "#4a2a15", title: "◎ VFX" },
    upscale: { label: "Upscale / image", group: "#88A", header: "#2c2c48", title: "◎ Upscale" },
    io: { label: "Read / write", group: "#444", header: "#2a2a2e", title: "◎ Inputs / Outputs" },
    review: { label: "Review / QC", group: "#444", header: "#26262a", title: "◎ Review" },
    pipeline: { label: "Pipeline", group: "#444", header: "#26262a", title: "◎ Pipeline" },
    film: { label: "Film / looks", group: "#A88", header: "#4a2f2f", title: "◎ Film Look" },
    optional: { label: "Optional (bypassed)", group: "#a1309b", header: "#461544", title: "◎ Optional" },
});

/** The colour of a group that is not one of the above. */
export const DEFAULT_GROUP = "#444";

/**
 * Discipline of a Radiance node from its category
 * ("FXTD STUDIOS/Radiance/◎ HDR/…"), or null for any other node.
 */
export function disciplineForCategory(category) {
    const m = /^FXTD STUDIOS\/Radiance\/(?:◎\s*)?([^/]+)/.exec(String(category || ""));
    if (!m) return null;
    const top = m[1].trim().toLowerCase();
    if (top === "hdr") return "hdr";
    if (top === "color" || top === "colour") return "color";
    if (top === "vfx") return "vfx";
    if (top === "generate" || top === "video") return "generate";
    if (top === "upscale") return "upscale";
    if (top === "load & save") return "io";
    if (top === "review") return "review";
    if (top === "film") return "film";
    return "pipeline";   // Pipeline, Core, Gizmos and anything new
}

/** Every node-header colour Radiance has ever saved into a workflow, old themes included. */
export const OLD_THEME_HEADERS = Object.freeze([
    "#0e3452", "#3c2020", "#2e3a3a", "#221834", "#122c1e", "#182238", "#3e3e4a",
    "#282830", "#3a1515", "#1c223c", "#2a153c", "#202026",
]);
export const OLD_THEME_BODIES = Object.freeze(["#141419"]);

/**
 * Discipline of any node from its category, ComfyUI's own and other packs'
 * included (for Smart Backdrop, which never recolours those nodes).
 */
export function disciplineForAnyCategory(category) {
    const own = disciplineForCategory(category);
    if (own) return own;
    const c = String(category || "").toLowerCase();
    if (!c) return null;
    if (/(^|\/)(sampling|loaders?|conditioning|latent|model_patches|advanced\/model)/.test(c)) return "generate";
    if (/upscal|image\/(transform|upscaling)/.test(c)) return "upscale";
    if (/mask|compositing/.test(c)) return "vfx";
    if (/(^|\/)image(\/|$)|save|load/.test(c)) return "io";
    return null;
}

/** The discipline a set of nodes mostly belongs to (Smart Backdrop), or null. */
export function dominantDiscipline(categories) {
    const counts = new Map();
    for (const c of categories) {
        const d = disciplineForAnyCategory(c);
        if (d) counts.set(d, (counts.get(d) || 0) + 1);
    }
    let best = null, n = 0;
    for (const [d, k] of counts) if (k > n) { best = d; n = k; }
    return best;
}
