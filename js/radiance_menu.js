// radiance_menu.js
// ─────────────────────────────────────────────────────────────────────────────
// Launcher so the Radiance dashboards are reachable without adding the
// Project Manager node. Adds a ◎ button to ComfyUI's action bar that opens a
// small menu — Project Manager / Workflow Library / Assets — via the in-canvas
// overlay (window.showRadianceDashboard, defined by radiance_workspace.js).
// To revert: delete this file.
// ─────────────────────────────────────────────────────────────────────────────
import { app } from "../../scripts/app.js";

// ALBABIT-FIX: resolve extension base at runtime so the path works regardless of the install folder name (e.g. "radiance" vs "radiance-beta")
const _EXT_BASE = import.meta.url.replace(/\/[^/]+$/, '');

const GOLD = "#c8a96e";

function openDash(file, title) {
    if (window.showRadianceDashboard) window.showRadianceDashboard(`${_EXT_BASE}/${file}`, title);
    else window.open(`${_EXT_BASE}/${file}`, "_blank");
}

// ALBABIT-FIX: the button lives in ComfyUI's action bar. As a fixed bottom-right
// element it covered the canvas toolbar (select, fit view, zoom, minimap) that
// the frontend now draws in that corner.
const style = document.createElement("style");
style.textContent = `.radiance-launcher-icon::before { content: "◎"; color: ${GOLD}; font-style: normal; font-size: 16px; line-height: 1; }`;
document.head.appendChild(style);

let menu = null;

function buildMenu() {
    menu = document.createElement("div");
    menu.id = "radiance-launcher-menu";
    Object.assign(menu.style, {
        position: "fixed", zIndex: "9000", minWidth: "190px", fontFamily: "'Inter',system-ui,sans-serif",
        background: "#111114", border: "1px solid rgba(255,255,255,0.1)", borderRadius: "10px",
        padding: "6px", display: "none", boxShadow: "0 24px 60px -30px rgba(0,0,0,0.9)",
    });
    const item = (icon, label, file, title) => {
        const b = document.createElement("button");
        b.innerHTML = `<span style="color:${GOLD};width:16px;display:inline-block;">${icon}</span>${label}`;
        Object.assign(b.style, {
            display: "flex", alignItems: "center", gap: "9px", width: "100%", textAlign: "left",
            background: "none", border: "0", color: "#e8e8ec", fontFamily: "inherit", fontSize: "12.5px",
            padding: "8px 10px", borderRadius: "7px", cursor: "pointer",
        });
        b.onmouseenter = () => { b.style.background = "rgba(255,255,255,0.05)"; };
        b.onmouseleave = () => { b.style.background = "none"; };
        b.onclick = () => { menu.style.display = "none"; openDash(file, title); };
        return b;
    };
    menu.appendChild(item("◎", "Project Manager", "project_manager_dashboard.html", "Radiance Project Manager"));
    menu.appendChild(item("▤", "Workflow Library", "workspace_dashboard.html", "Radiance Workflow Library"));
    menu.appendChild(item("▦", "Assets", "assets_dashboard.html", "Radiance Assets"));

    document.addEventListener("click", () => { menu.style.display = "none"; });
    document.body.appendChild(menu);
}

function toggleMenu(e) {
    e.stopPropagation();
    if (menu.style.display !== "none") {
        menu.style.display = "none";
        return;
    }
    const r = e.target.closest("button").getBoundingClientRect();
    menu.style.top = `${r.bottom + 6}px`;
    menu.style.right = `${window.innerWidth - r.right}px`;
    menu.style.display = "block";
}

app.registerExtension({
    name: "Radiance.Launcher.FAB",
    actionBarButtons: [
        { icon: "radiance-launcher-icon", tooltip: "Radiance dashboards", onClick: toggleMenu },
    ],
    async setup() {
        buildMenu();
    },
});

console.log("[Radiance Launcher] action bar menu ready");
