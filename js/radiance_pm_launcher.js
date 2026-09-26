// radiance_pm_launcher.js
// ─────────────────────────────────────────────────────────────────────────────
// Optional grouped launcher for the Radiance Project Manager node.
// Renders a compact tile layout (Open row + gold Save + footer links) as a DOM
// widget. SELF-CONTAINED and SAFE: it lives in its own file, so the core
// radiance_workspace.js is never modified. If the DOM widget renders, it removes
// the native launcher buttons; if it does NOT render (older ComfyUI), it removes
// itself and leaves the native buttons untouched — the node can never end empty.
// To revert entirely: delete this file.
// ─────────────────────────────────────────────────────────────────────────────
import { app } from "../../scripts/app.js";

// ALBABIT-FIX: resolve extension base at runtime so the path works regardless of the install folder name (e.g. "radiance" vs "radiance-beta")
const _EXT_BASE = import.meta.url.replace(/\/[^/]+$/, '');

const NATIVE_BTN_NAMES = new Set([
    "launch_project_manager", "launch_dashboard", "launch_assets",
    "quick_save", "inc_save", "docs_link", "site_link",
]);

const openDash = (file, title) => {
    if (window.showRadianceDashboard) {
        window.showRadianceDashboard(`${_EXT_BASE}/${file}`, title);
    } else {
        window.open(`${_EXT_BASE}/${file}`, "_blank");
    }
};

console.log("[Radiance PM Launcher] module loaded");

function buildLauncher(node) {
    const mk = (tag, css, txt) => { const e = document.createElement(tag); e.style.cssText = css; if (txt != null) e.textContent = txt; return e; };
    const secLabel = (s) => mk("div", "font-size:9px;letter-spacing:.14em;text-transform:uppercase;color:#52555c;margin:4px 0 1px;", s);
    const tile = (icon, label, accent, on) => {
        const el = mk("div", `height:46px;background:#26262c;border:1px solid ${accent ? "rgba(200,169,110,0.35)" : "rgba(255,255,255,0.08)"};border-radius:6px;display:flex;flex-direction:column;align-items:center;justify-content:center;gap:3px;font-size:9.5px;color:#cdd0d6;cursor:pointer;`);
        el.innerHTML = `<span style="font-size:14px;color:${accent ? "#c8a96e" : "#8a8f98"};">${icon}</span>${label}`;
        el.onmouseenter = () => { el.style.background = "#2f2f37"; };
        el.onmouseleave = () => { el.style.background = "#26262c"; };
        el.onclick = on;
        return el;
    };
    const primary = (label, on) => { const b = mk("div", "height:30px;background:#c8a96e;color:#241c08;border-radius:6px;display:flex;align-items:center;justify-content:center;font-size:11px;font-weight:600;cursor:pointer;", label); b.onmouseenter = () => { b.style.background = "#d8bd87"; }; b.onmouseleave = () => { b.style.background = "#c8a96e"; }; b.onclick = on; return b; };
    const ghost = (label, on) => { const b = mk("div", "height:30px;background:none;border:1px solid rgba(255,255,255,0.12);border-radius:6px;display:flex;align-items:center;justify-content:center;font-size:11px;color:#cdd0d6;cursor:pointer;", label); b.onmouseenter = () => { b.style.background = "rgba(255,255,255,0.05)"; }; b.onmouseleave = () => { b.style.background = "none"; }; b.onclick = on; return b; };
    const flink = (label, url) => { const s = mk("span", "cursor:pointer;", label); s.onmouseenter = () => { s.style.color = "#cdd0d6"; }; s.onmouseleave = () => { s.style.color = "#6c707a"; }; s.onclick = () => window.open(url, "_blank"); return s; };

    const wrap = mk("div", "display:flex;flex-direction:column;gap:6px;padding:4px 2px 6px;width:100%;height:100%;box-sizing:border-box;font-family:'Inter',system-ui,sans-serif;color:#e8e8ec;");
    wrap.appendChild(secLabel("Open"));
    const openRow = mk("div", "display:grid;grid-template-columns:1fr 1fr 1fr;gap:6px;");
    openRow.appendChild(tile("◎", "Manager", true, () => openDash("project_manager_dashboard.html", "Radiance Project Manager")));
    openRow.appendChild(tile("▤", "Library", false, () => openDash("workspace_dashboard.html", "Radiance Workflow Library")));
    openRow.appendChild(tile("▦", "Assets", false, () => openDash("assets_dashboard.html", "Radiance Assets")));
    wrap.appendChild(openRow);
    wrap.appendChild(secLabel("Save"));
    const saveRow = mk("div", "display:grid;grid-template-columns:1fr 1fr;gap:6px;");
    saveRow.appendChild(primary("Quick save", () => node.saveToLibrary && node.saveToLibrary()));
    saveRow.appendChild(ghost("Increment", () => node.incrementalSave && node.incrementalSave()));
    wrap.appendChild(saveRow);
    const footer = mk("div", "display:flex;justify-content:center;gap:10px;margin-top:4px;padding-top:7px;border-top:1px solid rgba(255,255,255,0.06);font-size:10.5px;color:#6c707a;");
    footer.appendChild(flink("Docs", "https://www.fxtdstudios.com"));
    footer.appendChild(mk("span", "color:#3a3d42;", "·"));
    footer.appendChild(flink("FXTD Studios", "https://www.fxtdstudios.com"));
    wrap.appendChild(footer);
    return wrap;
}

app.registerExtension({
    name: "Radiance.ProjectManager.Launcher",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (!["RadianceWorkspace", "RadianceProjectManager"].includes(nodeData.name)) return;

        const orig = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            orig?.apply(this, arguments);
            const node = this;
            if (typeof node.addDOMWidget !== "function") return; // keep native buttons

            try {
                if ((node.size?.[1] || 0) < 360) node.size = [Math.max(node.size?.[0] || 264, 264), 360];
                const wrap = buildLauncher(node);
                const w = node.addDOMWidget("radiance_launcher_grp", "radiance_launcher_grp", wrap, { serialize: false, hideOnZoom: false });
                if (!w) return; // could not add the widget — keep native buttons
                w.computeSize = () => [node.size[0] - 16, 196];
                // Remove the native launcher buttons. This must run AFTER they're
                // added: radiance_workspace.js loads after this file (alphabetical),
                // so its onNodeCreated (which adds the buttons) runs after ours.
                // Defer to the next tick so the buttons exist by the time we strip
                // them. We match by type ("button") since their .name is the label.
                const stripButtons = () => {
                    if (!node.widgets) return;
                    node.widgets = node.widgets.filter((wd) => wd === w || wd.type !== "button");
                    node.setDirtyCanvas?.(true, true);
                };
                setTimeout(stripButtons, 0);
                setTimeout(stripButtons, 250);
                console.log("[Radiance PM Launcher] grouped launcher active");
            } catch (err) {
                console.error("[Radiance PM Launcher] failed, native buttons kept:", err);
            }
        };
    },
});
