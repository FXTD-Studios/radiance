// radiance_workspace.js
// Handles the serialization of the entire ComfyUI app.graph into a secure .rad file

import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

import { escapeHtml } from "./radiance_dom_utils.js";

// ALBABIT-FIX: resolve extension base at runtime so the path works regardless of the install folder name (e.g. "radiance" vs "radiance-beta")
const _EXT_BASE = import.meta.url.replace(/\/[^/]+$/, '');

const HEADER = "RAD_WORKSPACE_V1::";
const SEARCH_DEBOUNCE_MS = 200;
const TOAST_DURATION_MS = 3000;

// Listen for postMessage updates from Studio Dashboard
window.addEventListener("message", (event) => {
    // Reject cross-origin senders. Without this, any page that can get a handle
    // to this window could load an arbitrary graph, or ask for the user's graph
    // back via the radiance_save_project_version reply below. The sibling
    // handler in project_manager_dashboard.mjs already does this check.
    if (event.origin !== window.location.origin) return;
    if (event.data && event.data.type === "radiance_load_workflow") {
        try {
            const graphData = typeof event.data.content === 'string' ? JSON.parse(event.data.content) : event.data.content;
            app.loadGraphData(graphData, false);
            showToast("Workflow loaded from dashboard!", "success");
        } catch(err) {
            showToast("Failed to parse loaded graph", "error");
        }
    } else if (event.data && event.data.type === "radiance_append_workflow") {
        try {
            const graphData = typeof event.data.content === 'string' ? JSON.parse(event.data.content) : event.data.content;
            app.loadGraphData(graphData, true);
            showToast("Workflow merged from dashboard!", "success");
        } catch(err) {
            showToast("Failed to parse merged graph", "error");
        }
    } else if (event.data && event.data.type === "radiance_save_project_version") {
        try {
            const projectId = event.data.projectId || "general";
            const graphData = app.graph.serialize();

            // Sync with the active canvas widgets if a RadianceWorkspace node is present
            const wsNode = app.graph._nodes.find(n => n.type === "RadianceWorkspace");
            let filename = undefined;
            let artist = undefined;
            if (wsNode) {
                const filenameWidget = wsNode.widgets.find(w => w.name === "filename");
                const artistWidget = wsNode.widgets.find(w => w.name === "artist");
                if (filenameWidget?.value) filename = filenameWidget.value;
                if (artistWidget?.value) artist = artistWidget.value;
            }

            const bodyPayload = {
                content: JSON.stringify(graphData),
                message: "Saved from Project Manager dashboard",
                author: artist || localStorage.getItem("radiance_artist") || "Radiance Artist",
            };
            if (filename) bodyPayload.filename = filename;

            const response = api.fetchApi(`/radiance/projects/${encodeURIComponent(projectId)}/save-version`, {
                method: "POST",
                body: JSON.stringify(bodyPayload),
                headers: { "Content-Type": "application/json" },
            });

            response
                .then((res) => res.json().then((data) => ({ ok: res.ok, data })))
                .then(({ ok, data }) => {
                    if (!ok || !data.success) throw new Error(data.error || "Save failed");
                    showToast("Project Manager saved the current canvas.", "success");
                    event.source?.postMessage({ type: "radiance_project_action_result", action: "save-version", success: true }, window.location.origin);
                })
                .catch((err) => {
                    showToast(`Project Manager save failed: ${err.message}`, "error");
                    event.source?.postMessage({ type: "radiance_project_action_result", action: "save-version", success: false, error: err.message }, window.location.origin);
                });
        } catch(err) {
            showToast(`Project Manager save failed: ${err.message}`, "error");
        }
    }
});

// ═══════════════════════════════════════════════════════════════════════════════
//        IN-CANVAS DASHBOARD OVERLAY  (renders inside ComfyUI, not a new tab)
// ═══════════════════════════════════════════════════════════════════════════════
//
// Opens a Radiance dashboard page (Project Manager / Workflow Library) as a modal
// overlay layered on top of the ComfyUI canvas in the SAME tab. The page runs in
// an <iframe>; it already posts load/save events to window.parent, which the
// listener above handles — so "load into canvas" keeps working. Closeable via the
// X button, the Esc key, or clicking the dimmed backdrop.
if (!window.showRadianceDashboard) {
    window.showRadianceDashboard = function (url, title = "Radiance") {
        // Cache-bust so dashboard edits always load fresh (ComfyUI caches static files).
        const bust = (u) => u + (u.includes("?") ? "&" : "?") + "t=" + Date.now();
        // If one is already open, just bring focus / replace its source.
        const existing = document.getElementById("radiance-dashboard-overlay");
        if (existing) {
            const fr = existing.querySelector("iframe");
            if (fr && url) fr.src = bust(url);
            return;
        }

        const overlay = document.createElement("div");
        overlay.id = "radiance-dashboard-overlay";
        Object.assign(overlay.style, {
            position: "fixed", inset: "0", zIndex: "10000",
            display: "flex", alignItems: "center", justifyContent: "center",
            background: "rgba(4,4,4,0.72)", backdropFilter: "blur(6px)",
            WebkitBackdropFilter: "blur(6px)",
        });

        const panel = document.createElement("div");
        Object.assign(panel.style, {
            position: "relative",
            width: "min(94vw, 1500px)", height: "min(92vh, 980px)",
            display: "flex", flexDirection: "column",
            background: "#0d0d0d",
            border: "1px solid rgba(255,255,255,0.10)", borderRadius: "14px",
            overflow: "hidden",
            boxShadow: "0 40px 120px -40px rgba(0,0,0,0.9)",
        });

        const bar = document.createElement("div");
        Object.assign(bar.style, {
            display: "flex", alignItems: "center", gap: "10px",
            padding: "12px 16px", flex: "0 0 auto",
            borderBottom: "1px solid rgba(255,255,255,0.08)",
            background: "linear-gradient(180deg,#121212,#0d0d0d)",
            fontFamily: "'Space Grotesk','Inter',sans-serif",
        });
        const dot = document.createElement("span");
        Object.assign(dot.style, {
            width: "9px", height: "9px", borderRadius: "50%",
            background: "#c8a96e", boxShadow: "0 0 10px rgba(200,169,110,0.8)",
        });
        const label = document.createElement("div");
        label.textContent = title;
        Object.assign(label.style, {
            color: "#fff", fontSize: "13px", fontWeight: "600",
            letterSpacing: "0.04em", flex: "1 1 auto",
        });
        const openTab = document.createElement("button");
        openTab.textContent = "Open in tab \u2197";
        Object.assign(openTab.style, {
            background: "transparent", color: "#888",
            border: "1px solid rgba(255,255,255,0.12)", borderRadius: "7px",
            padding: "5px 10px", cursor: "pointer", fontSize: "11px",
            fontFamily: "inherit", letterSpacing: "0.03em",
        });
        openTab.onmouseenter = () => { openTab.style.color = "#fff"; openTab.style.borderColor = "rgba(255,255,255,0.3)"; };
        openTab.onmouseleave = () => { openTab.style.color = "#888"; openTab.style.borderColor = "rgba(255,255,255,0.12)"; };
        openTab.onclick = () => window.open(url, "_blank");

        const closeBtn = document.createElement("button");
        closeBtn.textContent = "\u2715";
        Object.assign(closeBtn.style, {
            background: "transparent", color: "#bbb",
            border: "1px solid rgba(255,255,255,0.12)", borderRadius: "7px",
            width: "30px", height: "28px", cursor: "pointer",
            fontSize: "14px", lineHeight: "1", fontFamily: "inherit",
        });
        closeBtn.onmouseenter = () => { closeBtn.style.color = "#fff"; closeBtn.style.background = "rgba(255,80,80,0.18)"; closeBtn.style.borderColor = "rgba(255,80,80,0.4)"; };
        closeBtn.onmouseleave = () => { closeBtn.style.color = "#bbb"; closeBtn.style.background = "transparent"; closeBtn.style.borderColor = "rgba(255,255,255,0.12)"; };

        const iframe = document.createElement("iframe");
        iframe.src = bust(url);
        Object.assign(iframe.style, {
            flex: "1 1 auto", width: "100%", border: "0", background: "#0d0d0d",
        });

        const close = () => {
            window.removeEventListener("keydown", onKey);
            overlay.remove();
        };
        const onKey = (e) => { if (e.key === "Escape") close(); };
        closeBtn.onclick = close;
        overlay.addEventListener("mousedown", (e) => { if (e.target === overlay) close(); });
        window.addEventListener("keydown", onKey);

        bar.appendChild(dot);
        bar.appendChild(label);
        bar.appendChild(openTab);
        bar.appendChild(closeBtn);
        panel.appendChild(bar);
        panel.appendChild(iframe);
        overlay.appendChild(panel);
        document.body.appendChild(overlay);
    };
}

// ═══════════════════════════════════════════════════════════════════════════════
//                           ENCODING HELPERS
// ═══════════════════════════════════════════════════════════════════════════════

/**
 * UTF-8 safe Base64 encode using TextEncoder (no deprecated unescape/escape).
 */
function utf8ToBase64(str) {
    const bytes = new TextEncoder().encode(str);
    let binary = "";
    for (let i = 0; i < bytes.length; i++) {
        binary += String.fromCharCode(bytes[i]);
    }
    return btoa(binary);
}

/**
 * UTF-8 safe Base64 decode using TextDecoder.
 */
function base64ToUtf8(b64) {
    const binary = atob(b64);
    const bytes = new Uint8Array(binary.length);
    for (let i = 0; i < binary.length; i++) {
        bytes[i] = binary.charCodeAt(i);
    }
    return new TextDecoder().decode(bytes);
}

// ═══════════════════════════════════════════════════════════════════════════════
//                           TOAST NOTIFICATION
// ═══════════════════════════════════════════════════════════════════════════════

function showToast(message, type = "info") {
    const existing = document.querySelector(".radiance-toast");
    if (existing) existing.remove();

    const toast = document.createElement("div");
    toast.className = `radiance-toast radiance-toast-${type}`;
    toast.textContent = message;
    document.body.appendChild(toast);

    // Trigger reflow then animate in
    requestAnimationFrame(() => {
        toast.style.opacity = "1";
        toast.style.transform = "translateY(0)";
    });

    setTimeout(() => {
        toast.style.opacity = "0";
        toast.style.transform = "translateY(10px)";
        setTimeout(() => toast.remove(), 300);
    }, TOAST_DURATION_MS);
}

function confirmRadianceAction(message, confirmLabel = "Continue") {
    return new Promise((resolve) => {
        const overlay = document.createElement("div");
        overlay.className = "radiance-modal-overlay";

        const modal = document.createElement("div");
        modal.className = "radiance-modal";
        modal.style.maxWidth = "420px";

        const title = document.createElement("h2");
        title.textContent = "Confirm Action";

        const copy = document.createElement("p");
        copy.style.margin = "14px 0 22px";
        copy.style.color = "#b8c0cc";
        copy.style.lineHeight = "1.45";
        copy.textContent = message;

        const actions = document.createElement("div");
        actions.style.display = "flex";
        actions.style.justifyContent = "flex-end";
        actions.style.gap = "10px";

        const cancel = document.createElement("button");
        cancel.className = "radiance-btn";
        cancel.textContent = "Cancel";

        const confirm = document.createElement("button");
        confirm.className = "radiance-btn danger";
        confirm.textContent = confirmLabel;

        const close = (value) => {
            overlay.remove();
            resolve(value);
        };

        cancel.addEventListener("click", () => close(false));
        confirm.addEventListener("click", () => close(true));
        overlay.addEventListener("click", (event) => {
            if (event.target === overlay) close(false);
        });

        actions.append(cancel, confirm);
        modal.append(title, copy, actions);
        overlay.appendChild(modal);
        document.body.appendChild(overlay);
    });
}

function promptRadianceAction(titleText, message, defaultValue = "", confirmLabel = "Continue") {
    return new Promise((resolve) => {
        const overlay = document.createElement("div");
        overlay.className = "radiance-modal-overlay";

        const modal = document.createElement("div");
        modal.className = "radiance-modal";
        modal.style.maxWidth = "420px";

        const title = document.createElement("h2");
        title.textContent = titleText;

        const copy = document.createElement("p");
        copy.style.margin = "14px 0 12px";
        copy.style.color = "#b8c0cc";
        copy.style.lineHeight = "1.45";
        copy.textContent = message;

        const input = document.createElement("input");
        input.type = "text";
        input.value = defaultValue;
        input.style.cssText = `
            width: 100%; box-sizing: border-box; height: 36px; margin-bottom: 18px;
            border-radius: 8px; border: 1px solid rgba(255,255,255,0.12);
            background: rgba(255,255,255,0.06); color: #f5f5f7;
            padding: 0 10px; outline: none;
        `;

        const actions = document.createElement("div");
        actions.style.display = "flex";
        actions.style.justifyContent = "flex-end";
        actions.style.gap = "10px";

        const cancel = document.createElement("button");
        cancel.className = "radiance-btn";
        cancel.textContent = "Cancel";

        const confirm = document.createElement("button");
        confirm.className = "radiance-btn";
        confirm.textContent = confirmLabel;

        const close = (value) => {
            overlay.remove();
            resolve(value);
        };

        cancel.addEventListener("click", () => close(null));
        confirm.addEventListener("click", () => close(input.value.trim()));
        input.addEventListener("keydown", (event) => {
            if (event.key === "Enter") close(input.value.trim());
            if (event.key === "Escape") close(null);
        });
        overlay.addEventListener("click", (event) => {
            if (event.target === overlay) close(null);
        });

        actions.append(cancel, confirm);
        modal.append(title, copy, input, actions);
        overlay.appendChild(modal);
        document.body.appendChild(overlay);
        input.focus();
        input.select();
    });
}

function escapeAttr(value) {
    return escapeHtml(value).replace(/`/g, "&#96;");
}

// ═══════════════════════════════════════════════════════════════════════════════
//                           DEBOUNCE UTILITY
// ═══════════════════════════════════════════════════════════════════════════════

function debounce(fn, ms) {
    let timer;
    return function (...args) {
        clearTimeout(timer);
        timer = setTimeout(() => fn.apply(this, args), ms);
    };
}

// ═══════════════════════════════════════════════════════════════════════════════
//                           STYLES
// ═══════════════════════════════════════════════════════════════════════════════

const style = document.createElement("style");
style.textContent = `
    .radiance-toast {
        position: fixed; bottom: 24px; left: 50%; transform: translateX(-50%) translateY(10px);
        background: rgba(30, 32, 40, 0.95); color: #fff; padding: 12px 24px;
        border-radius: 8px; font-size: 0.9em; z-index: 10001;
        opacity: 0; transition: opacity 0.3s, transform 0.3s;
        box-shadow: 0 4px 20px rgba(0,0,0,0.4); border: 1px solid rgba(255,255,255,0.1);
        pointer-events: none; font-family: 'Inter', sans-serif;
    }
    .radiance-toast-success { border-left: 3px solid #34c759; }
    .radiance-toast-error   { border-left: 3px solid #ff3b30; }
    .radiance-toast-info    { border-left: 3px solid #4db3ff; }

    .radiance-modal-overlay {
        position: fixed; top: 0; left: 0; right: 0; bottom: 0;
        background: rgba(0,0,0,0.7); backdrop-filter: blur(8px); z-index: 10000;
        display: flex; justify-content: center; align-items: center;
        font-family: 'Inter', sans-serif;
    }
    .radiance-modal {
        background: rgba(30, 32, 40, 0.95); color: #ddd; padding: 25px; border-radius: 12px;
        width: 85%; max-width: 750px; max-height: 85%; overflow-y: auto;
        box-shadow: 0 10px 40px rgba(0,0,0,0.6); border: 1px solid rgba(255,255,255,0.1);
    }
    .radiance-modal-header {
        display: flex; justify-content: space-between; align-items: center;
        margin-bottom: 20px; border-bottom: 1px solid rgba(255,255,255,0.1); padding-bottom: 15px;
    }
    .radiance-modal h2 { margin: 0; color: #fff; font-size: 1.5em; letter-spacing: 0; }

    .radiance-search-container { margin-bottom: 20px; }
    .radiance-search-input {
        width: 100%; box-sizing: border-box;
        background: rgba(0,0,0,0.3); border: 1px solid rgba(255,255,255,0.1);
        padding: 10px 15px; border-radius: 6px; color: #fff; font-size: 1em; outline: none;
    }
    .radiance-search-input:focus { border-color: #4db3ff; box-shadow: 0 0 0 2px rgba(77, 179, 255, 0.2); }

    .radiance-workflow-list { list-style: none; padding: 0; margin: 0; }
    .radiance-workflow-item {
        display: flex; justify-content: space-between; align-items: center;
        padding: 15px; border-bottom: 1px solid rgba(255,255,255,0.05);
        background: rgba(255,255,255,0.02); margin-bottom: 8px; border-radius: 8px;
        transition: transform 0.2s, background 0.2s;
    }
    .radiance-workflow-item:hover { background: rgba(255,255,255,0.05); transform: translateX(4px); }
    .radiance-workflow-info { display: flex; flex-direction: column; flex: 1; margin-right: 15px; min-width: 0; }
    .radiance-workflow-name {
        font-weight: 600; color: #fff; font-size: 1.1em;
        white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
    }
    .radiance-workflow-desc { font-size: 0.85em; color: #888; margin-top: 4px; line-height: 1.4; }
    .radiance-workflow-meta { font-size: 0.75em; color: #555; margin-top: 6px; font-variant-numeric: tabular-nums; }

    .radiance-actions { display: flex; gap: 8px; flex-shrink: 0; }
    .radiance-btn {
        background: rgba(255,255,255,0.1); color: white; border: none; padding: 8px 14px;
        border-radius: 6px; cursor: pointer; font-size: 0.85em; font-weight: 500;
        transition: background 0.2s, color 0.2s;
    }
    .radiance-btn:hover { background: rgba(255,255,255,0.2); }
    .radiance-btn:focus-visible { outline: 2px solid #4db3ff; outline-offset: 2px; }
    .radiance-btn.primary { background: #005bb7; }
    .radiance-btn.primary:hover { background: #0076eb; }
    .radiance-btn.danger { background: rgba(255, 59, 48, 0.1); color: #ff3b30; }
    .radiance-btn.danger:hover { background: rgba(255, 59, 48, 0.2); }
    .radiance-btn.gold { background: rgba(212, 168, 83, 0.2); color: #d4a853; border: 1px solid rgba(212, 168, 83, 0.3); }
    .radiance-btn.gold:hover { background: rgba(212, 168, 83, 0.3); }

    .radiance-badge {
        background: #444; color: #aaa; padding: 2px 6px; border-radius: 4px;
        font-size: 0.7em; text-transform: uppercase; margin-left: 8px; vertical-align: middle;
    }
    .radiance-badge.system { background: #1a1c23; color: #4db3ff; border: 1px solid #005bb7; }

    .radiance-folder-group { margin-bottom: 25px; }
    .radiance-folder-header {
        font-size: 0.9em; font-weight: 700; color: #4db3ff; text-transform: uppercase;
        letter-spacing: 0.1em; margin-bottom: 12px; display: flex; align-items: center;
        opacity: 0.8;
    }
    .radiance-folder-header::before {
        content: ""; display: inline-block; width: 4px; height: 14px;
        background: #4db3ff; margin-right: 10px; border-radius: 2px;
    }

    .radiance-modal-close { margin-top: 25px; width: 100%; padding: 12px; font-weight: bold; background: #333; }

    .radiance-empty-state {
        text-align: center; padding: 40px 20px; opacity: 0.5;
    }
    .radiance-empty-state p { margin: 0; }

    /* --- Control Deck HUD Styles --- */
    .radiance-hud-container {
        font-family: 'Roboto Mono', monospace;
        color: #00f2ff;
        font-size: 10px;
        pointer-events: none;
    }
    .radiance-hud-label { color: #555; text-transform: uppercase; margin-right: 5px; }
    .radiance-hud-value { color: #aaa; }
    .radiance-hud-led {
        display: inline-block; width: 6px; height: 6px; border-radius: 50%;
        margin-right: 6px; box-shadow: 0 0 5px rgba(0, 242, 255, 0.5);
    }
    .radiance-hud-led.active { background: #00f2ff; animation: radiance-pulse 2s infinite; }
    .radiance-hud-led.idle { background: #333; box-shadow: none; }

    @keyframes radiance-pulse {
        0% { opacity: 1; box-shadow: 0 0 2px #00f2ff; }
        50% { opacity: 0.4; box-shadow: 0 0 8px #00f2ff; }
        100% { opacity: 1; box-shadow: 0 0 2px #00f2ff; }
    }
`;
document.head.appendChild(style);


// ═══════════════════════════════════════════════════════════════════════════════
//                       NODE EXTENSION REGISTRATION
// ═══════════════════════════════════════════════════════════════════════════════

app.registerExtension({
    name: "Radiance.ProjectManager",

    async beforeRegisterNodeDef(nodeType, nodeData, _app) {
        if (!["RadianceWorkspace", "RadianceProjectManager"].includes(nodeData.name)) return;

        const origOnNodeCreated = nodeType.prototype.onNodeCreated;

        nodeType.prototype.onNodeCreated = function () {
            origOnNodeCreated?.apply(this, arguments);

            // Keep the Project Manager dashboard separate from the workflow library.
            this.addWidget("button", "◎ PROJECT MANAGER", "launch_project_manager", () => {
                window.showRadianceDashboard("${_EXT_BASE}/project_manager_dashboard.html", "Radiance Project Manager");
            });
            this.addWidget("button", "WORKFLOW LIBRARY", "launch_dashboard", () => {
                window.showRadianceDashboard("${_EXT_BASE}/workspace_dashboard.html", "Radiance Workflow Library");
            });
            this.addWidget("button", "◎ ASSETS", "launch_assets", () => {
                window.showRadianceDashboard("${_EXT_BASE}/assets_dashboard.html", "Radiance Assets");
            });
            this.addWidget("button", "QUICK SAVE", "quick_save", () => this.saveToLibrary());
            this.addWidget("button", "INCREMENTAL SAVE", "inc_save", () => this.incrementalSave());
            this.addWidget("button", "DOCUMENTATION", "docs_link", () => window.open("https://www.fxtdstudios.com", "_blank"));
            this.addWidget("button", "FXTD STUDIOS", "site_link", () => window.open("https://www.fxtdstudios.com", "_blank"));

            this.color = "#111111";
            this.bgcolor = "#111111";
            this.size = [240, 300]; // Space for filename, artist, version widgets

            this.properties = this.properties || {};
            this.properties.last_commit = "No commits yet";
            this.properties.vram_peak = "0.0 GB";
            this.properties.version = "v0.0";
        };

        // --- Custom Drawing: Minimal Status HUD ---
        const origOnDrawForeground = nodeType.prototype.onDrawForeground;
        nodeType.prototype.onDrawForeground = function(ctx) {
            origOnDrawForeground?.apply(this, arguments);
            if (this.flags.collapsed) return;

            ctx.save();

            const y = this.size[1] - 25;

            // 1. Draw elegant horizontal divider
            ctx.strokeStyle = "rgba(0, 242, 255, 0.2)";
            ctx.lineWidth = 1;
            ctx.beginPath();
            ctx.moveTo(10, y - 5);
            ctx.lineTo(this.size[0] - 10, y - 5);
            ctx.stroke();

            // 2. Neon Border Glow (Slight active canvas highlight)
            ctx.strokeStyle = "#00f2ff";
            ctx.lineWidth = 1;
            ctx.globalAlpha = 0.15;
            ctx.strokeRect(0, 0, this.size[0], this.size[1]);
            ctx.globalAlpha = 1.0;

            // 3. Status Pulse LED indicator
            ctx.beginPath();
            ctx.arc(20, y + 4, 3, 0, Math.PI * 2);
            ctx.fillStyle = "#00f2ff";
            ctx.shadowBlur = 4;
            ctx.shadowColor = "#00f2ff";
            ctx.fill();
            ctx.shadowBlur = 0;

            // 4. Compact Status Monospace Text
            ctx.font = "bold 9px monospace";
            ctx.fillStyle = "#555";
            ctx.fillText("STUDIO LINK:", 32, y + 7);
            ctx.fillStyle = "#00f2ff";
            ctx.fillText("ACTIVE", 102, y + 7);

            ctx.restore();
        };

        // ─── Serialization ───────────────────────────────────────────

        nodeType.prototype.buildRadContent = async function (description = "") {
            const graphData = app.graph.serialize();
            const response = await api.fetchApi("/radiance/workflows/pack", {
                method: "POST",
                body: JSON.stringify({
                    content: JSON.stringify(graphData),
                    description: description,
                    author: localStorage.getItem("radiance_artist") || "Radiance Artist"
                }),
                headers: { "Content-Type": "application/json" },
            });
            if (!response.ok) throw new Error("Backend failed to pack workflow");
            return await response.blob();
        };

        nodeType.prototype.parseAndLoadRadContent = async function (blob, append = false) {
            try {
                const response = await api.fetchApi("/radiance/workflows/unpack", {
                    method: "POST",
                    body: blob,
                    headers: { "Content-Type": "application/octet-stream" },
                });

                const data = await response.json();
                if (!data.success) throw new Error(data.error);

                const graphData = typeof data.content === 'string' ? JSON.parse(data.content) : data.content;
                app.loadGraphData(graphData, append);

                if (data.secure) {
                    console.log("[Radiance] Verified secure binary .rad workspace.");
                } else {
                    console.warn("[Radiance] Loaded legacy plain-text .rad workspace.");
                }
                return true;
            } catch (err) {
                console.error("[Radiance] Failed to parse .rad content:", err);
                showToast(`Modernization Error: ${err.message}`, "error");
                return false;
            }
        };

        // ─── Export / Import (local file) ────────────────────────────

        nodeType.prototype.exportRadWorkspace = async function () {
            try {
                const defaultName = `shot_${Date.now()}`;
                let chosenName = await promptRadianceAction("Export Project", "Enter a name for the Project export.", defaultName, "Export");

                if (chosenName === null) return;
                chosenName = chosenName.trim() || defaultName;
                if (!chosenName.endsWith(".rad")) chosenName += ".rad";

                const description = await promptRadianceAction("Project Description", "Optional description for this .rad export.", "", "Continue") || "";
                showToast("Packing secure workflow...", "info");

                const blob = await this.buildRadContent(description);
                const url = URL.createObjectURL(blob);

                const a = document.createElement("a");
                a.href = url;
                a.download = chosenName;
                document.body.appendChild(a);
                a.click();

                setTimeout(() => {
                    document.body.removeChild(a);
                    URL.revokeObjectURL(url);
                }, 100);

                showToast(`Exported ${chosenName}`, "success");
            } catch (e) {
                console.error("[Radiance] Failed to export workspace:", e);
                showToast("Export failed — check console.", "error");
            }
        };

        nodeType.prototype.importRadWorkspace = function () {
            const input = document.createElement("input");
            input.type = "file";
            input.accept = ".rad";

            input.onchange = (e) => {
                const file = e.target.files?.[0];
                if (!file) return;

                const reader = new FileReader();
                reader.onload = async (event) => {
                    // Send raw bytes to unpacker
                    const blob = new Blob([event.target.result], { type: "application/octet-stream" });
                    const success = await this.parseAndLoadRadContent(blob);
                    if (success) {
                        showToast(`Imported ${file.name}`, "success");
                    }
                };
                reader.readAsArrayBuffer(file);
            };

            input.click();
        };

        nodeType.prototype.purgeVram = async function () {
            try {
                showToast("Purging VRAM & Unloading Models...", "info");
                const response = await api.fetchApi("/free", {
                    method: "POST",
                    body: JSON.stringify({ unload_models: true, free_memory: true }),
                    headers: { "Content-Type": "application/json" },
                });

                if (response.ok) {
                    showToast("GPU Cache Cleared. Radiance in Standby.", "success");
                    this.properties.vram_peak = "0.0 GB (Purged)";
                }
            } catch (e) {
                console.error("[Radiance] VRAM Purge failed:", e);
                showToast("Purge failed.", "error");
            }
        };

        // ─── Library: Save ───────────────────────────────────────────

        nodeType.prototype.saveToLibrary = async function (explicitName = null) {
            try {
                const filenameWidget = this.widgets.find(w => w.name === "filename");
                const versionWidget = this.widgets.find(w => w.name === "version");
                const artistWidget = this.widgets.find(w => w.name === "artist");

                const stem = filenameWidget?.value || "workflow";
                const ver = versionWidget?.value || 1;
                const verStr = `v${ver.toString().padStart(4, '0')}`;
                const artist = artistWidget?.value || localStorage.getItem('radiance_artist') || "unknown";
                if (artist && artist !== "unknown") {
                    localStorage.setItem('radiance_artist', artist);
                }

                let chosenBin = "GENERAL";

                if (explicitName) {
                    chosenBin = this.properties.production_bin || "GENERAL";
                } else {
                    chosenBin = await promptRadianceAction("Production Bin", "Choose a bin, for example TEMPLATES, SHOTS, or EXPERIMENTS.", this.properties.production_bin || "GENERAL", "Save");
                    if (chosenBin === null) return;
                    chosenBin = chosenBin.trim().toUpperCase() || "GENERAL";
                    this.properties.production_bin = chosenBin;
                }

                const chosenName = `${stem}_${artist}_${verStr}.rad`;

                const message = explicitName
                    ? `Incremental: ${chosenName}`
                    : (await promptRadianceAction("Commit Message", "Why did you save this workflow version?", "Updated workflow logic", "Save") || "Auto-commit");
                const description = explicitName
                    ? ""
                    : (await promptRadianceAction("Short Description", "Optional short description for the library entry.", "", "Continue") ?? "");
                const graphData = app.graph.serialize();

                // --- NEW: Generate Preview Image (Canvas Snapshot) ---
                let previewImage = "";
                try {
                    const canvas = app.canvas.canvas;
                    if (canvas) {
                        // Create a small thumbnail version
                        const thumbCanvas = document.createElement("canvas");
                        thumbCanvas.width = 1280;
                        thumbCanvas.height = 720;
                        const tCtx = thumbCanvas.getContext("2d");
                        tCtx.drawImage(canvas, 0, 0, thumbCanvas.width, thumbCanvas.height);
                        previewImage = thumbCanvas.toDataURL("image/png", 0.7);
                    }
                } catch (e) {
                    console.warn("[Radiance] Could not capture preview:", e);
                }

                // Get some basic stats for the "Rich" metadata
                const stats = {
                    node_count: app.graph._nodes.length,
                    saved_at: Date.now(),
                    os: navigator.platform
                };

                const response = await api.fetchApi("/radiance/workflows/save", {
                    method: "POST",
                    body: JSON.stringify({
                        filename: `${chosenBin}/${chosenName}`,
                        content: JSON.stringify(graphData),
                        description: description,
                        message: message,
                        author: artist,
                        stats: stats,
                        preview_image: previewImage
                    }),
                    headers: { "Content-Type": "application/json" },
                });

                if (response.ok) {
                    showToast(`Project Saved: "${chosenName}"`, "success");
                    this.properties.last_commit = message.length > 15 ? message.substring(0, 12) + "..." : message;
                    this.setDirtyCanvas(true);
                } else {
                    const err = await response.json().catch(() => ({}));
                    showToast(`Commit failed: ${err.error || response.statusText}`, "error");
                }
            } catch (e) {
                console.error("[Radiance] Failed to save to library:", e);
                showToast("Save failed — check console.", "error");
            }
        };

        nodeType.prototype.incrementalSave = async function () {
            const filenameWidget = this.widgets.find(w => w.name === "filename");
            const versionWidget = this.widgets.find(w => w.name === "version");

            if (!filenameWidget || !versionWidget) {
                showToast("Cannot perform incremental save: Missing filename/version widgets.", "error");
                return;
            }

            const stem = filenameWidget.value || "workflow";
            const ver = versionWidget.value;
            const verStr = `v${ver.toString().padStart(4, '0')}`;

            const incFilename = `${stem}_${verStr}`;

            // Execute the save
            await this.saveToLibrary(incFilename);

            // Increment for next time
            versionWidget.value = ver + 1;
            this.setDirtyCanvas(true);
        };

        // ─── Portal: Sync ────────────────────────────────────────────


        // ─── Library: Open ───────────────────────────────────────────

        nodeType.prototype.openLibrary = async function () {
            try {
                showToast("Scanning Project Library...", "info");

                // Fetch local workflows
                const localResp = await api.fetchApi("/radiance/workflows/list");
                const localData = await localResp.json();
                let allWorkflows = (localData.workflows || []).map(w => ({ ...w, origin: "local" }));

                this.showLibraryModal(allWorkflows);
            } catch (e) {
                console.error("[Radiance] Failed to load library:", e);
                showToast("Failed to open Project Library.", "error");
            }
        };

        // ─── Library: Load / Delete ──────────────────────────────────

        nodeType.prototype.loadFromLibrary = async function (wf, append = false) {
            try {
                let content = "";
                let metadata = {};

                const response = await api.fetchApi(
                    `/radiance/workflows/get?filename=${encodeURIComponent(wf.filename)}`
                );
                if (!response.ok) throw new Error(`HTTP ${response.status}`);
                const data = await response.json();
                if (!data.success) throw new Error(data.error);
                content = data.content;
                metadata = data.metadata;

                app.loadGraphData(typeof content === 'string' ? JSON.parse(content) : content, append);
                const success = true;
                if (success) {
                    const action = append ? "Appended" : "Loaded";
                    showToast(`${action} workflow: ${wf.name || wf.filename}`, "success");
                }
            } catch (e) {
                console.error("[Radiance] Failed to load workflow:", e);
                showToast("Failed to load workflow from library.", "error");
            }
        };

        // Helper to handle the library modal data binding
        nodeType.prototype.showLibraryModal = function (workflows) {
            this._showLibraryModal(workflows);
        };

        nodeType.prototype.deleteFromLibrary = async function (filename) {
            try {
                const response = await api.fetchApi("/radiance/workflows/delete", {
                    method: "POST",
                    body: JSON.stringify({ filename }),
                    headers: { "Content-Type": "application/json" },
                });
                if (!response.ok) throw new Error(`HTTP ${response.status}`);
                showToast("Workflow deleted.", "info");
            } catch (e) {
                console.error("[Radiance] Failed to delete workflow:", e);
                showToast("Delete failed.", "error");
            }
        };

        // ─── Library: Modal UI ───────────────────────────────────────

        nodeType.prototype._showLibraryModal = function (initialWorkflows) {
            // Mutable reference that the delete handler can update
            const state = { workflows: initialWorkflows };

            const overlay = document.createElement("div");
            overlay.className = "radiance-modal-overlay";

            const modal = document.createElement("div");
            modal.className = "radiance-modal";

            // Header
            const headerDiv = document.createElement("div");
            headerDiv.className = "radiance-modal-header";
            const title = document.createElement("h2");
            title.textContent = "◎ Radiance Project Library";
            headerDiv.appendChild(title);

            const refreshBtn = document.createElement("button");
            refreshBtn.className = "radiance-btn";
            refreshBtn.style.marginLeft = "auto";
            refreshBtn.textContent = "↻ Refresh";
            refreshBtn.onclick = async () => {
                showToast("Refreshing Library...", "info");
                const resp = await api.fetchApi("/radiance/workflows/list");
                const data = await resp.json();
                state.workflows = (data.workflows || []).map(w => ({ ...w, origin: "local" }));
                renderList(searchInput.value);
            };
            headerDiv.appendChild(refreshBtn);

            modal.appendChild(headerDiv);

            // Search
            const searchContainer = document.createElement("div");
            searchContainer.className = "radiance-search-container";
            const searchInput = document.createElement("input");
            searchInput.placeholder = "Search workflows by name or description…";
            searchInput.className = "radiance-search-input";
            searchContainer.appendChild(searchInput);
            modal.appendChild(searchContainer);

            // List container
            const list = document.createElement("ul");
            list.className = "radiance-workflow-list";
            modal.appendChild(list);

            // ── Cleanup helper ──
            const closeModal = () => {
                overlay.remove();
                document.removeEventListener("keydown", onKeyDown);
            };

            const onKeyDown = (e) => {
                if (e.key === "Escape") closeModal();
            };
            document.addEventListener("keydown", onKeyDown);

            // ── History View helper ──
            const showHistory = async (wf) => {
                showToast(`Fetching history for ${wf.filename}...`, "info");
                const resp = await api.fetchApi(`/radiance/workflows/history?filename=${encodeURIComponent(wf.filename)}`);
                const data = await resp.json();

                if (!data.success) {
                    showToast("Failed to load history.", "error");
                    return;
                }

                // Temporary override modal content with history list
                const originalContent = modal.innerHTML;
                modal.innerHTML = `
                    <div class="radiance-modal-header">
                        <h2>History: ${escapeHtml(wf.filename.split('/').pop())}</h2>
                        <button class="radiance-btn" id="history-back">← Back</button>
                    </div>
                    <div class="radiance-workflow-list" style="margin-top:20px;">
                        ${data.history.map(v => `
                            <div class="radiance-workflow-item">
                                <div class="radiance-workflow-info">
                                    <span class="radiance-workflow-name">${escapeHtml(v.message || "Manual Backup")}</span>
                                    <span class="radiance-workflow-desc">Author: ${escapeHtml(v.author || "Unknown")}</span>
                                    <span class="radiance-workflow-meta">${escapeHtml(new Date(v.timestamp * 1000).toLocaleString())} • ${escapeHtml(v.version_file)}</span>
                                </div>
                                <div class="radiance-actions">
                                    <button class="radiance-btn primary history-restore" data-file="${escapeAttr(v.version_file)}">Restore</button>
                                </div>
                            </div>
                        `).join('')}
                    </div>
                `;

                modal.querySelector("#history-back").onclick = () => {
                    modal.innerHTML = originalContent;
                    // Re-bind search and other logic
                    this._rebindLibraryLogic(modal, state, renderList, closeModal);
                };

                modal.querySelectorAll(".history-restore").forEach(btn => {
                    btn.onclick = async () => {
                        const vFile = btn.dataset.file;
                        if (await confirmRadianceAction("Restore workflow to this version? Current state will be backed up.", "Restore")) {
                            const restResp = await api.fetchApi("/radiance/workflows/restore", {
                                method: "POST",
                                body: JSON.stringify({ filename: wf.filename, version_file: vFile }),
                                headers: { "Content-Type": "application/json" }
                            });
                            const restData = await restResp.json();
                            if (restData.success) {
                                showToast("Restoration complete!", "success");
                                this.loadFromLibrary(wf, false);
                                closeModal();
                            }
                        }
                    };
                });
            };

            // ── Render list ──
            const renderList = (filter = "") => {
                list.innerHTML = "";
                const lowerFilter = filter.toLowerCase();
                const filtered = state.workflows.filter((wf) => {
                    const haystack = `${wf.filename} ${wf.name || ""} ${wf.metadata?.description || ""}`.toLowerCase();
                    return haystack.includes(lowerFilter);
                });

                if (filtered.length === 0) {
                    const empty = document.createElement("div");
                    empty.className = "radiance-empty-state";
                    const p = document.createElement("p");
                    p.textContent = filter ? "No workflows match your search." : "Library is empty — save a workflow to get started.";
                    empty.appendChild(p);
                    list.appendChild(empty);
                    return;
                }

                const groups = {};
                for (const wf of filtered) {
                    const parts = wf.filename.split("/");
                    const folder = parts.length > 1 ? parts.slice(0, -1).join("/") : "Root";
                    (groups[folder] ||= []).push(wf);
                }

                const sortedFolders = Object.keys(groups).sort((a, b) => {
                    if (a === "Root") return -1;
                    if (b === "Root") return 1;
                    return a.localeCompare(b);
                });

                for (const folder of sortedFolders) {
                    const groupDiv = document.createElement("div");
                    groupDiv.className = "radiance-folder-group";

                    const header = document.createElement("div");
                    header.className = "radiance-folder-header";
                    header.style.background = "rgba(0, 242, 255, 0.03)";
                    header.style.padding = "6px 12px";
                    header.style.borderRadius = "4px";
                    header.style.borderLeft = "3px solid #00f2ff";

                    let categoryLabel = folder === "Root" ? "GENERAL BIN" : `${folder.toUpperCase()} BIN`;
                    if (folder.toLowerCase().startsWith("official")) {
                        categoryLabel = "◎ OFFICIAL RADIANCE BUNDLE";
                    } else if (folder.toLowerCase().startsWith("system")) {
                        categoryLabel = "⚙ SYSTEM UTILITIES";
                    }

                    header.textContent = categoryLabel;
                    groupDiv.appendChild(header);

                    const groupList = document.createElement("ul");
                    groupList.className = "radiance-workflow-list";

                    for (const wf of groups[folder]) {
                        const li = document.createElement("li");
                        li.className = "radiance-workflow-item";
                        li.style.flexDirection = "column";
                        li.style.alignItems = "stretch";

                        const mainRow = document.createElement("div");
                        mainRow.style.display = "flex";
                        mainRow.style.justifyContent = "space-between";
                        mainRow.style.alignItems = "center";
                        li.appendChild(mainRow);

                        // Thumbnail Preview
                        if (wf.metadata?.has_preview) {
                            const thumbContainer = document.createElement("div");
                            thumbContainer.style.width = "100%";
                            thumbContainer.style.height = "120px";
                            thumbContainer.style.background = "#000";
                            thumbContainer.style.marginBottom = "10px";
                            thumbContainer.style.borderRadius = "4px";
                            thumbContainer.style.overflow = "hidden";
                            thumbContainer.style.position = "relative";

                            const img = document.createElement("img");
                            // Use correct path prefix for ComfyUI API
                            img.src = `./radiance/workflows/preview?filename=${encodeURIComponent(wf.filename)}`;
                            img.style.width = "100%";
                            img.style.height = "100%";
                            img.style.objectFit = "cover";
                            img.style.opacity = "0.7";
                            thumbContainer.appendChild(img);

                            // Stats Overlay
                            if (wf.metadata?.stats) {
                                const statsOverlay = document.createElement("div");
                                statsOverlay.style.position = "absolute";
                                statsOverlay.style.bottom = "5px";
                                statsOverlay.style.right = "5px";
                                statsOverlay.style.background = "rgba(0,0,0,0.7)";
                                statsOverlay.style.padding = "2px 6px";
                                statsOverlay.style.fontSize = "9px";
                                statsOverlay.style.color = "#00f2ff";
                                statsOverlay.style.fontFamily = "monospace";
                                statsOverlay.textContent = `${wf.metadata.stats.vram_peak || "N/A"} VRAM`;
                                thumbContainer.appendChild(statsOverlay);
                            }

                            li.insertBefore(thumbContainer, mainRow);
                        }

                        const info = document.createElement("div");
                        info.className = "radiance-workflow-info";

                        const nameSpan = document.createElement("span");
                        nameSpan.className = "radiance-workflow-name";
                        const displayName = wf.filename.split("/").pop().replace(".rad", "");
                        nameSpan.textContent = displayName;

                        if (wf.filename === "default.rad" || wf.filename.startsWith("System/")) {
                            const badge = document.createElement("span");
                            badge.className = "radiance-badge system";
                            badge.textContent = "System";
                            nameSpan.appendChild(badge);
                        } else if (wf.filename.toLowerCase().startsWith("official/")) {
                            const badge = document.createElement("span");
                            badge.className = "radiance-badge system";
                            badge.style.borderColor = "#d4a853";
                            badge.style.color = "#d4a853";
                            badge.textContent = "Official Workflow";
                            nameSpan.appendChild(badge);
                        } else {
                            const badge = document.createElement("span");
                            badge.className = "radiance-badge";
                            badge.textContent = "User Workflow";
                            nameSpan.appendChild(badge);
                        }

                        info.appendChild(nameSpan);

                        // --- NEW: Technical Pipeline Badges ---
                        const pipeData = wf.metadata?.pipeline;
                        if (pipeData) {
                            const badgeRow = document.createElement("div");
                            badgeRow.style.display = "flex";
                            badgeRow.style.gap = "4px";
                            badgeRow.style.marginTop = "4px";
                            badgeRow.style.flexWrap = "wrap";

                            const addBadge = (text, color = "#00f2ff") => {
                                const b = document.createElement("span");
                                b.style.background = "rgba(0, 242, 255, 0.05)";
                                b.style.color = color;
                                b.style.border = `1px solid ${color}44`;
                                b.style.padding = "1px 5px";
                                b.style.borderRadius = "3px";
                                b.style.fontSize = "9px";
                                b.style.fontWeight = "bold";
                                b.style.textTransform = "uppercase";
                                b.textContent = text;
                                badgeRow.appendChild(b);
                            };

                            if (pipeData.resolutions?.length) {
                                pipeData.resolutions.forEach(r => addBadge(`${r}P`));
                            }
                            if (pipeData.fps) {
                                addBadge(`${pipeData.fps} FPS`, "#34c759");
                            }
                            if (pipeData.color_spaces?.length) {
                                pipeData.color_spaces.forEach(cs => {
                                    const label = cs.length > 8 ? cs.substring(0, 6) + ".." : cs;
                                    addBadge(label, "#ff9500");
                                });
                            }
                            if (pipeData.is_hdr) {
                                addBadge("HDR", "#af52de");
                            }

                            info.appendChild(badgeRow);
                        }

                        if (wf.metadata?.description) {
                            const descDiv = document.createElement("div");
                            descDiv.className = "radiance-workflow-desc";
                            descDiv.textContent = wf.metadata.description;
                            info.appendChild(descDiv);
                        }

                        const metaSpan = document.createElement("span");
                        metaSpan.className = "radiance-workflow-meta";
                        const date = new Date(wf.mtime * 1000).toLocaleString();
                        const sizeKB = Math.round(wf.size / 1024);
                        metaSpan.textContent = `${date}  •  ${sizeKB} KB`;
                        info.appendChild(metaSpan);
                        mainRow.appendChild(info);

                        // Action buttons
                        const actions = document.createElement("div");
                        actions.className = "radiance-actions";

                        const historyBtn = document.createElement("button");
                        historyBtn.className = "radiance-btn";
                        historyBtn.innerHTML = "<span>🕗</span>";
                        historyBtn.title = "View Version History";
                        historyBtn.onclick = () => showHistory(wf);
                        actions.appendChild(historyBtn);

                        const appendBtn = document.createElement("button");
                        appendBtn.className = "radiance-btn";
                        appendBtn.textContent = "Append";
                        appendBtn.onclick = () => {
                            this.loadFromLibrary(wf, true);
                            closeModal();
                        };

                        const loadBtn = document.createElement("button");
                        loadBtn.className = "radiance-btn primary";
                        loadBtn.textContent = "Load";
                        loadBtn.onclick = async () => {
                            if (await confirmRadianceAction(`Replace current graph with "${displayName}"?`, "Replace")) {
                                this.loadFromLibrary(wf, false);
                                closeModal();
                            }
                        };

                        if (wf.filename === "default.rad" || wf.filename.startsWith("System/")) {
                            const updateBtn = document.createElement("button");
                            updateBtn.className = "radiance-btn";
                            updateBtn.style.background = "rgba(40, 167, 69, 0.2)";
                            updateBtn.style.color = "#28a745";
                            updateBtn.textContent = "Update";
                            updateBtn.onclick = async () => {
                                if (await confirmRadianceAction(`Overwrite System Workflow "${displayName}" with your current canvas?`, "Overwrite")) {
                                    const graphData = app.graph.serialize();
                                    const response = await api.fetchApi("/radiance/workflows/save", {
                                        method: "POST",
                                        body: JSON.stringify({
                                            filename: wf.filename,
                                            content: JSON.stringify(graphData),
                                            description: wf.metadata?.description || "Updated System Workflow"
                                        }),
                                        headers: { "Content-Type": "application/json" },
                                    });
                                    if (response.ok) {
                                        showToast(`System Workflow "${displayName}" updated!`, "success");
                                    }
                                }
                            };
                            actions.appendChild(updateBtn);
                        }

                        const delBtn = document.createElement("button");
                        delBtn.className = "radiance-btn danger";
                        delBtn.textContent = "Delete";
                        delBtn.onclick = async () => {
                            if (!await confirmRadianceAction(`Permanently delete "${wf.filename}"?`, "Delete")) return;
                            await this.deleteFromLibrary(wf.filename);

                            // Refresh list from server
                            try {
                                const response = await api.fetchApi("/radiance/workflows/list");
                                const data = await response.json();
                                state.workflows = data.workflows || [];
                            } catch {
                                // Remove locally as fallback
                                state.workflows = state.workflows.filter((w) => w.filename !== wf.filename);
                            }
                            renderList(searchInput.value);
                        };

                        actions.appendChild(appendBtn);
                        actions.appendChild(loadBtn);
                        actions.appendChild(delBtn);
                        mainRow.appendChild(actions);
                        groupList.appendChild(li);
                    }

                    groupDiv.appendChild(groupList);
                    list.appendChild(groupDiv);
                }
            };

            // Debounced search
            searchInput.oninput = debounce((e) => renderList(e.target.value), SEARCH_DEBOUNCE_MS);
            renderList();

            // Close button
            const closeBtn = document.createElement("button");
            closeBtn.className = "radiance-btn radiance-modal-close";
            closeBtn.textContent = "Close";
            closeBtn.onclick = closeModal;
            modal.appendChild(closeBtn);

            // Click-outside to dismiss
            overlay.onclick = (e) => {
                if (e.target === overlay) closeModal();
            };

            overlay.appendChild(modal);
            document.body.appendChild(overlay);

            // Focus search on open
            requestAnimationFrame(() => searchInput.focus());
        };

        // Internal helper to re-bind search logic after history-back
        nodeType.prototype._rebindLibraryLogic = function(modal, state, renderList, closeModal) {
            const searchInput = modal.querySelector(".radiance-search-input");
            const refreshBtn = modal.querySelector(".radiance-btn");
            const closeBtn = modal.querySelector(".radiance-modal-close");

            searchInput.oninput = debounce((e) => renderList(e.target.value), SEARCH_DEBOUNCE_MS);

            refreshBtn.onclick = async () => {
                showToast("Refreshing Library...", "info");
                const resp = await api.fetchApi("/radiance/workflows/list");
                const data = await resp.json();
                state.workflows = (data.workflows || []).map(w => ({ ...w, origin: "local" }));
                renderList(searchInput.value);
            };

            closeBtn.onclick = closeModal;
            requestAnimationFrame(() => searchInput.focus());
            renderList(searchInput.value);
        };
    },
});
