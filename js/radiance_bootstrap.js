/**
 * radiance_bootstrap.js
 * ═══════════════════════════════════════════════════════════════════════════
 * ◎ Radiance — Early console noise filter & bootstrap
 *
 * Installs a console.warn / console.log proxy at module-load time to
 * suppress high-frequency 3rd-party warnings that pollute the dev console.
 *
 * Filtered sources (2026-05 audit):
 *   · [Context Menu Compat]  — ComfyUI compat layer, fires every right-click
 *   · ImpactSchedulerAdapter — Impact Pack defaultInput deprecation spam
 *   · getCanvasMenuOptions   — rgthree monkey-patch deprecation, every click
 *   · KJNodes.browserstatus  — disabled extension notice (unactionable)
 *   · materialdesignicons    — Chrome slow-network font fallback (env noise)
 * ═══════════════════════════════════════════════════════════════════════════
 */

import { app } from "../../scripts/app.js";

// ─────────────────────────────────────────────────────────────────────────────
// Fast substring deny-list — checked before the regex list for performance.
// Any warn/log whose first string argument contains one of these substrings
// is silently dropped. Substrings are intentionally specific to avoid
// accidentally silencing legitimate messages.
// ─────────────────────────────────────────────────────────────────────────────
const DENY_SUBSTRINGS = [
    "[Context Menu Compat] Monkey patch",
    "ImpactSchedulerAdapter:scheduler is deprecated",
    "Monkey-patching getCanvasMenuOptions is deprecated",
    "KJNodes.browserstatus is disabled",
    "Slow network is detected",
    "materialdesignicons-webfont",
    // ComfyUI prompt_service: fires when viewer/API queues before UI tracking is ready
    "'execution_start' fired before prompt was made",
    // Legacy bootstrap message leaking from a cached/third-party source
    "Early console noise filter active",
];

/**
 * Returns true if the console call's arguments match a noise pattern.
 * Checks every argument, not just the first, to handle multi-arg warn calls.
 */
function _isNoise(args) {
    for (const a of args) {
        const s = typeof a === "string" ? a : (a instanceof Error ? a.message : null);
        if (!s) continue;
        for (const sub of DENY_SUBSTRINGS) {
            if (s.includes(sub)) return true;
        }
    }
    return false;
}

const ENABLE_CONSOLE_FILTER = window.localStorage?.getItem("radiance.consoleFilter") === "1";

// Capture originals before optional patching — used internally to bypass the proxy.
const _origLog  = console.log;   // line 50
const _origWarn = console.warn;  // line 51
const _origInfo = console.info;  // line 52
// console.error intentionally left unpatched — errors must always surface.

// Expose the original console.log globally so downstream modules
// (e.g. RadianceViewer._termLog) can log without triggering any proxy.
window.__radianceOrigLog = _origLog;

if (ENABLE_CONSOLE_FILTER) {
    // ─────────────────────────────────────────────────────────────────────────
    // Optional proxy — disabled by default so Radiance never hides diagnostics
    // from ComfyUI or third-party extensions unless the artist explicitly opts in.
    // ─────────────────────────────────────────────────────────────────────────
    console.log = function radianceBootstrapLogProxy(...args) {
        if (_isNoise(args)) return;
        _origLog.apply(console, args);
    };

    console.warn = function radianceBootstrapWarnProxy(...args) {
        if (_isNoise(args)) return;
        _origWarn.apply(console, args);
    };

    console.info = function radianceBootstrapInfoProxy(...args) {
        if (_isNoise(args)) return;
        _origInfo.apply(console, args);
    };

    _origLog("[Radiance Bootstrap] Console noise filter active. Filtered", DENY_SUBSTRINGS.length, "patterns.");
}

// ─────────────────────────────────────────────────────────────────────────────
// ComfyUI extension registration
// ─────────────────────────────────────────────────────────────────────────────
app.registerExtension({
    name: "FXTD.Radiance.Bootstrap",
    async init() {
        // Proxy installed at module-load time above — nothing more to do here.
    },
});
