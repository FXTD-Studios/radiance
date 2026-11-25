// radiance_menu.js
// ─────────────────────────────────────────────────────────────────────────────
// Launcher so the Radiance dashboards are reachable without adding the
// Project Manager node. Adds a ◎ button to ComfyUI's action bar that opens a
// small menu — Radiance (Home / Workflows / Gizmos / Assets / Shots / Models) / Radiance Tour — via the in-canvas
// overlay (window.showRadianceDashboard, defined by radiance_workspace.js).
// To revert: delete this file.
// ─────────────────────────────────────────────────────────────────────────────
import { app } from "../../scripts/app.js";

// ALBABIT-FIX: resolve extension base at runtime so the path works regardless of the install folder name (e.g. "radiance" vs "radiance-beta")
const _EXT_BASE = import.meta.url.replace(/\/[^/]+$/, '');

// FXTD blue (fxtdstudios.com, and the icon's square).
const BRAND = "#0a84ff";

function openDash(file, title) {
    if (window.showRadianceDashboard) window.showRadianceDashboard(`${_EXT_BASE}/${file}`, title);
    else window.open(`${_EXT_BASE}/${file}`, "_blank");
}

// The FXTD community Discord (also on the tour's last page, LINKS.discord).
const DISCORD_URL = "https://discord.gg/C7aTNFhJeP";
const DISCORD_ICON = '<svg width="15" height="12" viewBox="0 0 24 18" aria-hidden="true" style="display:block"><path fill="#5865F2" d="M20.3 1.5A19.8 19.8 0 0 0 15.4 0l-.6 1.3a18.4 18.4 0 0 0-5.5 0L8.6 0a19.7 19.7 0 0 0-4.9 1.5C.6 6.2-.3 10.7.1 15.2A19.9 19.9 0 0 0 6.2 18l1.3-2.1a12.9 12.9 0 0 1-2-1l.5-.4a14.2 14.2 0 0 0 12 0l.5.4c-.6.4-1.3.7-2 1L17.8 18a19.8 19.8 0 0 0 6-3c.5-5.2-.8-9.7-3.5-13.5ZM8 12.4c-1.2 0-2.2-1.1-2.2-2.4S6.8 7.6 8 7.6s2.2 1.1 2.2 2.4-1 2.4-2.2 2.4Zm8 0c-1.2 0-2.2-1.1-2.2-2.4s1-2.4 2.2-2.4 2.2 1.1 2.2 2.4-1 2.4-2.2 2.4Z"/></svg>';

// ALBABIT-FIX: the button lives in ComfyUI's action bar. As a fixed bottom-right
// element it covered the canvas toolbar (select, fit view, zoom, minimap) that
// the frontend now draws in that corner.
const style = document.createElement("style");
// The button shows the Radiance for ComfyUI icon, as on fxtdstudios.com.
style.textContent = `.radiance-launcher-icon::before { content: ""; display: inline-block; width: 16px; height: 16px;
    background: url("${_EXT_BASE}/r_icon.png") center / contain no-repeat; }`;
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
    const item = (icon, label, file, title, run = null) => {
        const b = document.createElement("button");
        b.innerHTML = `<span style="color:${BRAND};width:16px;display:inline-block;">${icon}</span>${label}`;
        Object.assign(b.style, {
            display: "flex", alignItems: "center", gap: "9px", width: "100%", textAlign: "left",
            background: "none", border: "0", color: "#e8e8ec", fontFamily: "inherit", fontSize: "12.5px",
            padding: "8px 10px", borderRadius: "7px", cursor: "pointer",
        });
        b.onmouseenter = () => { b.style.background = "rgba(255,255,255,0.05)"; };
        b.onmouseleave = () => { b.style.background = "none"; };
        b.type = "button";
        b.onclick = () => { menu.style.display = "none"; if (run) run(); else openDash(file, title); };
        return b;
    };
    menu.appendChild(item("◎", "Home", "studio.html#home", "Radiance"));
    menu.appendChild(item("▤", "Workflows", "studio.html#workflows", "Radiance"));
    menu.appendChild(item("◇", "Gizmos", "studio.html#gizmos", "Radiance"));
    menu.appendChild(item("▦", "Assets", "studio.html#assets", "Radiance"));
    menu.appendChild(item("▭", "Shots", "studio.html#shots", "Radiance"));
    menu.appendChild(item("⬇", "Models and setup", "studio.html#models", "Radiance"));
    // The welcome tour (radiance_welcome.js), also under Help > Radiance Tour.
    const sep = document.createElement("div");
    Object.assign(sep.style, { height: "1px", margin: "4px 6px", background: "rgba(255,255,255,0.08)" });
    menu.appendChild(sep);
    const tour = item("✦", "Radiance Tour", null, null, () => window.radianceWelcome?.open("tour"));
    tour.dataset.radianceMenu = "tour";
    menu.appendChild(tour);
    // Community: the FXTD Discord, in a new tab.
    const discord = item(DISCORD_ICON, `Join our Discord<span style="margin-left:auto;color:#8e8e93;">\u2197</span>`, null, null,
        () => window.open(DISCORD_URL, "_blank", "noopener,noreferrer"));
    discord.dataset.radianceMenu = "discord";
    menu.appendChild(discord);

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
