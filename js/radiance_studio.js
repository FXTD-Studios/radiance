import { app } from "../../scripts/app.js";
import { DISCIPLINES } from "./radiance_palette.js";

// ALBABIT-FIX: resolve extension base at runtime so the path works regardless of the install folder name (e.g. "radiance" vs "radiance-beta")
const _EXT_BASE = import.meta.url.replace(/\/[^/]+$/, '');

/**
 * ═══════════════════════════════════════════════════════════════════════════════
 *                         RADIANCE STUDIO
 *                  Professional Workflow Tools for ComfyUI
 * ═══════════════════════════════════════════════════════════════════════════════
 *
 * - Nuke-style colored backdrop creation with presets
 * - Node alignment tools
 */

// ─────────────────────────────────────────────────────────────────────────────
//  BACKDROP COLOR PALETTE (Nuke-style, desaturated for professional look)
// ─────────────────────────────────────────────────────────────────────────────

// ComfyUI's own group colours (LGraphCanvas.node_colors groupcolor), so a
// backdrop looks the same with or without Radiance.
const BACKDROP_COLORS = {
    "Red": "#A88",
    "Brown": "#b06634",
    "Green": "#8A8",
    "Blue": "#88A",
    "Pale blue": "#3f789e",
    "Cyan": "#8AA",
    "Purple": "#a1309b",
    "Yellow": "#b58b2a",
    "Black": "#444",
};

// Quick presets: the shared colour code (radiance_palette.js).
const BACKDROP_PRESETS = ["generate", "hdr", "color", "sdr", "vfx", "upscale", "io", "review", "optional"]
    .map((k) => ({ label: DISCIPLINES[k].label, colorValue: DISCIPLINES[k].group, icon: "◎" }));

function promptRadianceStudio(titleText, message, defaultValue = "", confirmLabel = "Create") {
    return new Promise((resolve) => {
        const overlay = document.createElement("div");
        overlay.style.cssText = "position:fixed;inset:0;z-index:10001;display:grid;place-items:center;background:rgba(0,0,0,0.55);backdrop-filter:blur(8px);";
        const dialog = document.createElement("div");
        dialog.style.cssText = "width:min(420px,calc(100vw - 32px));padding:18px;background:rgba(18,18,24,0.96);color:#f5f5f7;border:1px solid rgba(255,255,255,0.1);border-radius:8px;box-shadow:0 18px 60px rgba(0,0,0,0.65);font:13px/1.45 -apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;";
        const title = document.createElement("div");
        title.textContent = titleText;
        title.style.cssText = "font-size:15px;font-weight:700;margin-bottom:8px;";
        const copy = document.createElement("div");
        copy.textContent = message;
        copy.style.cssText = "color:#b8c0cc;margin-bottom:12px;";
        const input = document.createElement("input");
        input.type = "text";
        input.value = defaultValue;
        input.style.cssText = "width:100%;box-sizing:border-box;height:36px;margin-bottom:16px;border-radius:8px;border:1px solid rgba(255,255,255,0.14);background:rgba(255,255,255,0.06);color:#f5f5f7;padding:0 10px;outline:none;";
        const actions = document.createElement("div");
        actions.style.cssText = "display:flex;gap:10px;justify-content:flex-end;";
        const cancel = document.createElement("button");
        cancel.type = "button";
        cancel.textContent = "Cancel";
        cancel.style.cssText = "height:32px;padding:0 12px;border-radius:6px;border:1px solid rgba(255,255,255,0.1);background:rgba(255,255,255,0.05);color:#f5f5f7;cursor:pointer;";
        const confirm = document.createElement("button");
        confirm.type = "button";
        confirm.textContent = confirmLabel;
        confirm.style.cssText = "height:32px;padding:0 12px;border-radius:6px;border:1px solid rgba(0,168,255,0.45);background:rgba(0,168,255,0.16);color:#9fdcff;cursor:pointer;font-weight:700;";
        const close = (value) => {
            overlay.remove();
            resolve(value);
        };
        cancel.onclick = () => close(null);
        confirm.onclick = () => close(input.value.trim());
        input.onkeydown = (event) => {
            if (event.key === "Enter") close(input.value.trim());
            if (event.key === "Escape") close(null);
        };
        overlay.onclick = (event) => {
            if (event.target === overlay) close(null);
        };
        actions.append(cancel, confirm);
        dialog.append(title, copy, input, actions);
        overlay.appendChild(dialog);
        document.body.appendChild(overlay);
        input.focus();
        input.select();
    });
}

// ─────────────────────────────────────────────────────────────────────────────
//  EXTENSION REGISTRATION
// ─────────────────────────────────────────────────────────────────────────────

app.registerExtension({
    name: "FXTD.Radiance.Studio",

    getCanvasMenuItems() {
        const canvas = app.canvas;
        const graph = canvas.graph;
        const selectedNodes = Object.values(canvas.selected_nodes || {});
        const projectManagerOptions = [
            {
                content: "Open Project Manager",
                callback: () => {
                    if (window.showRadianceDashboard) {
                        window.showRadianceDashboard(`${_EXT_BASE}/project_manager_dashboard.html`, "Radiance Project Manager");
                    } else {
                        window.open(`${_EXT_BASE}/project_manager_dashboard.html`, "_blank");
                    }
                }
            },
            {
                content: "Open Workflow Library",
                callback: () => {
                    if (window.showRadianceDashboard) {
                        window.showRadianceDashboard(`${_EXT_BASE}/workspace_dashboard.html`, "Radiance Workflow Library");
                    } else {
                        window.open(`${_EXT_BASE}/workspace_dashboard.html`, "_blank");
                    }
                }
            },
            {
                content: "Radiance Tour",
                callback: () => window.radianceWelcome?.open("tour"),
            },
        ];

        if (selectedNodes.length === 0) {
            return [
                null,
                {
                    content: "◎ Radiance Studio",
                    submenu: {
                        options: projectManagerOptions,
                    },
                },
            ];
        }

        return [
            null, // Separator
            {
                content: "◎ Radiance Studio",
                submenu: {
                    options: [
                        ...projectManagerOptions,
                        null,
                        // ── Color Backdrop ────────────────────────
                        {
                            content: "Create Backdrop",
                            submenu: {
                                options: [
                                    // Solid colors
                                    ...Object.entries(BACKDROP_COLORS).map(([name, color]) => ({
                                        content: `■ ${name}`,
                                        callback: async () => {
                                            const title = await promptRadianceStudio("Backdrop Name", "Name this production backdrop.", name, "Create");
                                            if (title === null) return;
                                            createBackdrop(graph, selectedNodes, title, color);
                                        }
                                    })),
                                    null, // Separator
                                    // No prompt — instant gray
                                    {
                                        content: "Quick (no title)",
                                        callback: () => createBackdrop(graph, selectedNodes, "", BACKDROP_COLORS.Black),
                                    },
                                ]
                            }
                        },
                        // ── Quick Presets ─────────────────────────
                        {
                            content: "Quick Presets",
                            submenu: {
                                options: BACKDROP_PRESETS.map(preset => ({
                                    content: `${preset.icon} ${preset.label}`,
                                    callback: () => createBackdrop(graph, selectedNodes, preset.label, preset.colorValue),
                                }))
                            }
                        },
                        null, // Separator
                        // ── Alignment ────────────────────────────
                        {
                            content: "Align Horizontal",
                            callback: () => alignNodes(graph, selectedNodes, "horizontal"),
                        },
                        {
                            content: "Align Vertical",
                            callback: () => alignNodes(graph, selectedNodes, "vertical"),
                        },
                        {
                            content: "Distribute Evenly",
                            callback: () => distributeNodes(graph, selectedNodes),
                        },
                    ]
                }
            }
        ];
    }
});

// ─────────────────────────────────────────────────────────────────────────────
//  BACKDROP CREATION
// ─────────────────────────────────────────────────────────────────────────────

function createBackdrop(graph, nodes, title, color) {
    if (!nodes || nodes.length === 0) return;

    // Calculate bounding box of selected nodes
    let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;

    for (const node of nodes) {
        const x = node.pos[0];
        const y = node.pos[1];
        const w = node.size[0];
        const h = node.size[1];

        if (x < minX) minX = x;
        if (y < minY) minY = y;
        if (x + w > maxX) maxX = x + w;
        if (y + h > maxY) maxY = y + h;
    }

    // Padding — generous for professional look
    const PAD_X = 50;
    const PAD_Y = 50;
    const HEADER = 60;

    minX -= PAD_X;
    minY -= PAD_Y + HEADER;
    maxX += PAD_X;
    maxY += PAD_Y;

    const width = maxX - minX;
    const height = maxY - minY;

    // Create LiteGraph group
    const group = new LiteGraph.LGraphGroup();
    group.title = title || "";
    group.pos = [minX, minY];
    group.size = [width, height];
    group.color = color || BACKDROP_COLORS.Black;
    group.font_size = title ? 24 : 16;

    graph.add(group);
    graph.setDirtyCanvas(true, true);
}

// ─────────────────────────────────────────────────────────────────────────────
//  NODE ALIGNMENT
// ─────────────────────────────────────────────────────────────────────────────

function alignNodes(graph, nodes, direction) {
    if (nodes.length < 2) return;

    if (direction === "horizontal") {
        // Align Y to the first node (sort by X)
        nodes.sort((a, b) => a.pos[0] - b.pos[0]);
        const targetY = nodes[0].pos[1];
        for (const node of nodes) {
            node.pos[1] = targetY;
        }
    } else {
        // Align X to the first node (sort by Y)
        nodes.sort((a, b) => a.pos[1] - b.pos[1]);
        const targetX = nodes[0].pos[0];
        for (const node of nodes) {
            node.pos[0] = targetX;
        }
    }

    graph.setDirtyCanvas(true, true);
}

function distributeNodes(graph, nodes) {
    if (nodes.length < 3) return;

    // Sort by X position
    nodes.sort((a, b) => a.pos[0] - b.pos[0]);

    const firstX = nodes[0].pos[0];
    const lastX = nodes[nodes.length - 1].pos[0];
    const spacing = (lastX - firstX) / (nodes.length - 1);

    for (let i = 0; i < nodes.length; i++) {
        nodes[i].pos[0] = firstX + (spacing * i);
    }

    graph.setDirtyCanvas(true, true);
}
