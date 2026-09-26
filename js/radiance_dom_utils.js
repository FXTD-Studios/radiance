// ◎ Radiance — shared DOM helpers.
//
// escapeHtml existed in five places (radiance_gizmo.js, radiance_workspace.js,
// radiance_viewer.js, assets_dashboard.mjs, project_manager_dashboard.mjs) with
// two behaviours: three coerced null/undefined to "" and two rendered them as
// the literal text "null"/"undefined" in the UI. Five copies of a security
// primitive is four too many — the day one of them needs a fix, four keep the
// bug. This is the one implementation, with the safer coercion.

/**
 * Escape a value for interpolation into an HTML template literal.
 *
 * Covers the five characters that matter inside element content and inside
 * both quoted attribute forms. Not a substitute for proper DOM construction
 * when the value lands somewhere exotic (a URL, an inline event handler, a
 * <script> body) — escape there is not enough, and you want textContent or a
 * real URL parser instead.
 *
 * @param {*} value  Anything; null and undefined become an empty string.
 * @returns {string}
 */
export function escapeHtml(value) {
    return String(value ?? "").replace(/[&<>"']/g, (char) => ({
        "&": "&amp;",
        "<": "&lt;",
        ">": "&gt;",
        "\"": "&quot;",
        "'": "&#39;",
    })[char]);
}

export default { escapeHtml };
