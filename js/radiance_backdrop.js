/**
 * radiance_backdrop.js
 * ═══════════════════════════════════════════════════════════════════════════
 * ◎ RADIANCE SMART GLASSMORPHIC BACKDROPS  v1  —  PIPELINE WRAPPER
 * Upgrades standard LiteGraph groups into premium VFX backdrops.
 * ═══════════════════════════════════════════════════════════════════════════
 */

import { app } from "../../scripts/app.js";

// ─── Constants & Palette Tokens ──────────────────────────────────────────────
const ACCENT_BLUE = "#00a8ff";

const RADIANCE_PALETTE = {
    HDR:       "#b38600", // warm amber-gold
    Color:     "#008c9e", // vibrant teal-cyan
    Film:      "#8c30e8", // rich purple-violet
    VFX:       "#009f4d", // vibrant forest-green
    Generate:  "#0066cc", // deep navy-blue
    IO:        "#009999", // bright teal
    Pipeline:  "#5c6b73", // neutral slate
    Training:  "#cc0029", // crimson
    Image:     "#4361ee", // indigo
    Utilities: "#4a5568", // charcoal
    Default:   "#3a4b5c"  // classic Radiance slate
};

// Background glass fill
const GLASS_FILL = "rgba(15, 15, 20, 0.45)";

// Tinted glass: near-opaque dark base so the backdrop reads as a real panel
// on the dark canvas (the old 0.45 fill was effectively invisible).
const BASE_FILL = "rgba(18, 18, 24, 0.9)";
// Strength of the category-colour wash bleeding down from the top edge.
const WASH_ALPHA = 0.20;
const WASH_HEIGHT = 130;

// Convert "#rrggbb" -> "rgba(r,g,b,a)"; returns null for non-hex input.
function hexToRgba(hex, a) {
    if (typeof hex !== "string" || hex[0] !== "#" || hex.length < 7) return null;
    const r = parseInt(hex.slice(1, 3), 16);
    const g = parseInt(hex.slice(3, 5), 16);
    const b = parseInt(hex.slice(5, 7), 16);
    if ([r, g, b].some(Number.isNaN)) return null;
    return `rgba(${r}, ${g}, ${b}, ${a})`;
}


function promptBackdropTitle(defaultValue = "") {
    return new Promise((resolve) => {
        const overlay = document.createElement("div");
        overlay.style.cssText = "position:fixed;inset:0;z-index:10001;display:grid;place-items:center;background:rgba(0,0,0,0.55);backdrop-filter:blur(8px);";
        const dialog = document.createElement("div");
        dialog.style.cssText = "width:min(420px,calc(100vw - 32px));padding:18px;background:rgba(18,18,24,0.96);color:#f5f5f7;border:1px solid rgba(255,255,255,0.1);border-radius:8px;box-shadow:0 18px 60px rgba(0,0,0,0.65);font:13px/1.45 -apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;";
        const title = document.createElement("div");
        title.textContent = "Rename Backdrop";
        title.style.cssText = "font-size:15px;font-weight:700;margin-bottom:8px;";
        const copy = document.createElement("div");
        copy.textContent = "Enter the backdrop title.";
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
        confirm.textContent = "Rename";
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
// Dominant Discipline & Title Heuristics
// ─────────────────────────────────────────────────────────────────────────────

function analyzeDominantDiscipline(nodes) {
    const counts = {
        HDR: 0, Color: 0, Film: 0, VFX: 0, Generate: 0,
        IO: 0, Pipeline: 0, Training: 0, Image: 0, Utilities: 0
    };

    nodes.forEach(node => {
        const cls = (node.comfyClass || node.type || "").toUpperCase();
        
        if (cls.includes("HDR") || cls.includes("TURBO") || cls.includes("ENCODER"))
            counts.HDR++;
        else if (cls.includes("COLOR") || cls.includes("GRADE") || cls.includes("CURVES") ||
                 cls.includes("ACES")  || cls.includes("WHITEBALANCE") || cls.includes("OCIO") ||
                 cls.includes("COLORSPACE") || cls.includes("SCOPES") || cls.includes("CDL"))
            counts.Color++;
        else if (cls.includes("FILM")  || cls.includes("GRAIN") || cls.includes("OPTIC") ||
                 cls.includes("AESTHETIC") || cls.includes("CAMERA"))
            counts.Film++;
        else if (cls.includes("DEPTH") || cls.includes("OVERLAY") || cls.includes("COMPOSITE") ||
                 cls.includes("TEMPORAL") || cls.includes("DENOISE") || cls.includes("MOTION") ||
                 cls.includes("MULTIPASS") || cls.includes("MASK"))
            counts.VFX++;
        else if (cls.includes("LOADER") || cls.includes("SAMPLER") || cls.includes("PROMPT") ||
                 cls.includes("LORA")   || cls.includes("VAE")     || cls.includes("CONTROLNET") ||
                 cls.includes("REGIONAL") || cls.includes("STUDIO"))
            counts.Generate++;
        else if (cls.includes("IO")    || cls.includes("EXR")   || cls.includes("VIDEO") ||
                 cls.includes("NDI")   || cls.includes("RENDERQUEUE"))
            counts.IO++;
        else if (cls.includes("NUKE")  || cls.includes("RESOLVE") || cls.includes("WORKSPACE") ||
                 cls.includes("LAYOUT") || cls.includes("QUEUE")  || cls.includes("MASTERING") ||
                 cls.includes("METADATA"))
            counts.Pipeline++;
        else if (cls.includes("TRAIN") || cls.includes("SDRDEG") || cls.includes("TURBOTRAIN"))
            counts.Training++;
        else if (cls.includes("IMAGE") || cls.includes("RESOLUTION") || cls.includes("UPSCALE") ||
                 cls.includes("PANORAMA"))
            counts.Image++;
        else if (cls.includes("DNA")   || cls.includes("ENGINE") || cls.includes("TEXT") ||
                 cls.includes("VITALS") || cls.includes("QC"))
            counts.Utilities++;
    });

    let maxVal = 0;
    let dominant = "Default";
    for (const [key, val] of Object.entries(counts)) {
        if (val > maxVal) {
            maxVal = val;
            dominant = key;
        }
    }
    return dominant;
}

function getSuggestedTitle(discipline) {
    switch (discipline) {
        case "HDR":       return "◎ HDR Processing";
        case "Color":     return "◎ Color Grading";
        case "Film":      return "◎ Film Emulation";
        case "VFX":       return "◎ VFX Compositing";
        case "Generate":  return "◎ AI Generation";
        case "IO":        return "◎ Inputs / Outputs";
        case "Pipeline":  return "◎ Pipeline Infrastructure";
        case "Training":  return "◎ Network Training";
        case "Image":     return "◎ Image Processing";
        case "Utilities": return "◎ Utility Operations";
        default:          return "◎ Backdrop";
    }
}


// ─────────────────────────────────────────────────────────────────────────────
// Point-in-Group Geometry Checker
// ─────────────────────────────────────────────────────────────────────────────

function isPointInGroup(group, x, y) {
    const pos = group.pos;
    const size = group.size;
    return x >= pos[0] && y >= pos[1] &&
           x <= pos[0] + size[0] && y <= pos[1] + size[1];
}


// ─────────────────────────────────────────────────────────────────────────────
// Smart Auto-Backdrop Creator
// ─────────────────────────────────────────────────────────────────────────────

function createSmartBackdrop(selectedNodes) {
    if (!app.graph) return;

    let pos, size, title, color;

    if (selectedNodes && selectedNodes.length > 0) {
        // 1. Calculate Bounding Box bounds of selected nodes
        let minX = Infinity, minY = Infinity;
        let maxX = -Infinity, maxY = -Infinity;

        selectedNodes.forEach(node => {
            const [nx, ny] = node.pos;
            const [nw, nh] = node.size || [200, 100];
            minX = Math.min(minX, nx);
            minY = Math.min(minY, ny);
            maxX = Math.max(maxX, nx + nw);
            maxY = Math.max(maxY, ny + nh);
        });

        // Add visual padding
        const padding = 40;
        const headerPadding = 50;

        pos = [minX - padding, minY - headerPadding];
        size = [maxX - minX + (padding * 2), maxY - minY + padding + headerPadding];

        // 2. Perform dynamic snapping & auto-naming
        const dominant = analyzeDominantDiscipline(selectedNodes);
        color = RADIANCE_PALETTE[dominant];
        title = getSuggestedTitle(dominant);
    } else {
        // 3. Fallback: Create empty backdrop at mouse position
        const mouseCanvas = app.canvas.convertCanvasToOffset(app.canvas.last_mouse_position);
        pos = [mouseCanvas[0] - 150, mouseCanvas[1] - 100];
        size = [300, 200];
        color = RADIANCE_PALETTE.Default;
        title = "◎ Empty Backdrop";
    }

    // 4. Instantiate & Configure LGraphGroup
    const group = new LiteGraph.LGraphGroup(title);
    group.pos = pos;
    group.size = size;
    group.color = color;
    group.properties ??= {};
    group.properties.locked = false;

    app.graph.add(group);
    app.canvas.setDirty(true, true);
}


// ─────────────────────────────────────────────────────────────────────────────
// EXTENSION BOOTSTRAP: EVENT PATCHES & OVERRIDES (Warning-Free API)
// ─────────────────────────────────────────────────────────────────────────────

app.registerExtension({
    name: "FXTD.Radiance.Backdrop",

    // Modern ComfyUI Canvas Context Menu Hook (Warning-Free API)
    getCanvasMenuItems(canvas) {
        const items = [];
        const clickPos = canvas.convertCanvasToOffset(canvas.last_mouse_position);
        const cx = clickPos[0];
        const cy = clickPos[1];
        
        // ── Option A: Smart Backdrop Auto-creation ──
        const selected = Object.values(app.canvas.selected_nodes || {});
        items.push(null); // Separator
        items.push({
            content: selected.length > 0 ? "◎ Create Smart Backdrop" : "◎ Create Backdrop",
            callback: () => {
                createSmartBackdrop(selected);
            }
        });

        // ── Option B: Lock Backdrop context commands (for Unlocked groups under mouse) ──
        const unlockedGroups = (app.graph._groups || []).filter(g => {
            const isLocked = g.properties?.locked ?? false;
            return !isLocked && isPointInGroup(g, cx, cy);
        });

        if (unlockedGroups.length > 0) {
            unlockedGroups.forEach(g => {
                items.push({
                    content: `🔒 Lock Backdrop: ${g.title}`,
                    callback: () => {
                        g.properties ??= {};
                        g.properties.locked = true;
                        app.canvas.setDirty(true);
                    }
                });
            });
        }

        // ── Option C: Unlock Backdrop context commands (for Locked click-transparent groups under mouse) ──
        const lockedGroups = (app.graph._groups || []).filter(g => {
            const isLocked = g.properties?.locked ?? false;
            return isLocked && isPointInGroup(g, cx, cy);
        });

        if (lockedGroups.length > 0) {
            lockedGroups.forEach(g => {
                items.push({
                    content: `🔓 Unlock Backdrop: ${g.title}`,
                    callback: () => {
                        g.properties.locked = false;
                        app.canvas.setDirty(true);
                    }
                });
            });
        }

        return items;
    },

    async setup() {
        if (!window.LiteGraph || !window.LGraphCanvas || !window.LGraphGroup) return;

        // ─────────────────────────────────────────────────────────────────────
        // 1. PREMIUM GLASSMORPHIC RENDERING (LGraphGroup.prototype.draw)
        // ─────────────────────────────────────────────────────────────────────
        // Save the original before replacing it. This override previously
        // clobbered LGraphGroup.prototype.draw with no reference kept, which
        // destroyed any group rendering another extension had installed and
        // left no way back. The sibling patches below (getGroupAt,
        // onDoubleClick) already save and delegate -- this one now matches.
        const origGroupDraw = LGraphGroup.prototype.draw;
        LGraphGroup.prototype.draw = function() {
            // Escape hatch: lets a user or another pack opt out of Radiance's
            // group styling without uninstalling.
            if (window.RADIANCE_DISABLE_GROUP_STYLE ||
                localStorage.getItem("radiance.disableGroupStyle") === "1") {
                return origGroupDraw ? origGroupDraw.apply(this, arguments) : undefined;
            }
            // Bulletproof context and canvas argument resolution (supports all LiteGraph versions)
            let ctx = null;
            let canvas = null;
            for (let i = 0; i < arguments.length; i++) {
                const arg = arguments[i];
                if (arg && typeof arg === "object" && typeof arg.save === "function") {
                    ctx = arg;
                } else if (arg && typeof arg === "object") {
                    canvas = arg;
                }
            }
            if (!ctx) {
                ctx = arguments[0];
                canvas = arguments[1];
            }

            // Guard: Fallback safely if canvas or canvas.ds is undefined
            const scale = (canvas && canvas.ds && canvas.ds.scale !== undefined) ? canvas.ds.scale : 1.0;
            if (scale < 0.2) return; // Zoom-LOD Optimization

            ctx.save();

            const x = this.pos[0];
            const y = this.pos[1];
            const w = this.size[0];
            const h = this.size[1];
            const r = 10; // Rounded corner radius
            const isLocked = this.properties?.locked ?? false;
            const themeColor = this.color || RADIANCE_PALETTE.Default;

            // a) Draw Rounded Glassmorphic Fill & Border
            ctx.beginPath();
            ctx.moveTo(x + r, y);
            ctx.lineTo(x + w - r, y);
            ctx.quadraticCurveTo(x + w, y, x + w, y + r);
            ctx.lineTo(x + w, y + h - r);
            ctx.quadraticCurveTo(x + w, y + h, x + w - r, y + h);
            ctx.lineTo(x + r, y + h);
            ctx.quadraticCurveTo(x, y + h, x, y + h - r);
            ctx.lineTo(x, y + r);
            ctx.quadraticCurveTo(x, y, x + r, y);
            ctx.closePath();

            // Tinted glass: near-opaque dark base, then a soft top wash of the
            // group's category colour (ties the fill to the border, keeps depth).
            ctx.fillStyle = BASE_FILL;
            ctx.fill();
            const washTop = hexToRgba(themeColor, WASH_ALPHA);
            if (washTop) {
                const washGrad = ctx.createLinearGradient(x, y, x, y + Math.min(h, WASH_HEIGHT));
                washGrad.addColorStop(0, washTop);
                washGrad.addColorStop(1, hexToRgba(themeColor, 0));
                ctx.fillStyle = washGrad;
                ctx.fill();
            }

            // Glowing border
            ctx.strokeStyle = themeColor;
            ctx.lineWidth = isLocked ? 1.0 : 1.8;
            if (isLocked) {
                ctx.setLineDash([6, 4]); // Dashed border indicates locked state
            }
            ctx.stroke();
            ctx.setLineDash([]); // Reset line dash

            // b) Draw Top Sleek Colored Tab Accent Line
            ctx.fillStyle = themeColor;
            ctx.fillRect(x + 10, y + 4, w - 20, 2);

            // c) Draw Monospace Typography
            // Render title
            ctx.fillStyle = isLocked ? "#5c6b73" : "#e2e8f0";
            ctx.font = "bold 12px 'Courier New', monospace";
            ctx.textAlign = "left";
            ctx.fillText(this.title || "Group", x + 12, y + 20);

            // Render Lock Indicator Icon if locked
            if (isLocked) {
                ctx.fillStyle = "#ff4a4a";
                ctx.font = "10px 'Courier New', monospace";
                ctx.fillText("🔒 LOCKED", x + w - 75, y + 20);
            }

            // Render active node counter
            // Query LiteGraph elements contained inside group boundaries
            const containedNodes = app.graph._nodes?.filter(n => {
                const nx = n.pos[0];
                const ny = n.pos[1];
                return nx >= x && ny >= y && nx <= x + w && ny <= y + h;
            }) || [];
            
            if (containedNodes.length > 0) {
                ctx.fillStyle = "rgba(139, 160, 184, 0.4)";
                ctx.font = "8px 'Courier New', monospace";
                ctx.fillText(`${containedNodes.length} NODES NESTED`, x + 12, y + 32);
            }

            ctx.restore();
        };


        // ─────────────────────────────────────────────────────────────────────
        // 2. BACKDROP LOCKING (LGraphCanvas.prototype.getGroupAt)
        // ─────────────────────────────────────────────────────────────────────
        const origGetGroupAt = LGraphCanvas.prototype.getGroupAt;
        LGraphCanvas.prototype.getGroupAt = function(x, y) {
            const group = origGetGroupAt.apply(this, arguments);
            // If the group is locked in properties, make it click-transparent
            // EXCEPT when a modifier key (Ctrl, Shift, Alt) or Space (panning) is held.
            // This provides a professional UX bypass without unlocking.
            if (group && group.properties?.locked) {
                const e = window.event;
                const bypass = (e && (e.ctrlKey || e.shiftKey || e.altKey)) || 
                               (app.canvas && (app.canvas.space_pressed || app.canvas.pointer_is_selecting));
                if (bypass) {
                    return group;
                }
                return null;
            }
            return group;
        };


        // ─────────────────────────────────────────────────────────────────────
        // 4. QUICK RENAME VIA DOUBLE-CLICK (LGraphCanvas.prototype.onDoubleClick)
        // ─────────────────────────────────────────────────────────────────────
        const origOnDoubleClick = LGraphCanvas.prototype.onDoubleClick;
        if (origOnDoubleClick) {
            LGraphCanvas.prototype.onDoubleClick = function(e) {
                const clickPos = this.convertEventToCanvasOffset(e);
                const cx = clickPos[0];
                const cy = clickPos[1];

                // Find if double click fell specifically on a backdrop's header region (top 35px)
                const targetGroup = (app.graph._groups || []).find(g => {
                    const pos = g.pos;
                    const size = g.size;
                    return cx >= pos[0] && cy >= pos[1] &&
                           cx <= pos[0] + size[0] && cy <= pos[1] + 35;
                });

                if (targetGroup) {
                    promptBackdropTitle(targetGroup.title).then((newTitle) => {
                    if (newTitle !== null) {
                        targetGroup.title = newTitle;
                        this.setDirty(true, true);
                    }
                    });
                    return; // Suppress default action
                }

                return origOnDoubleClick.apply(this, arguments);
            };
        }


        // ─────────────────────────────────────────────────────────────────────
        // 5. GLOBAL KEYBOARD SHORTCUT (Alt + B) — Protected against duplicate listeners
        // ─────────────────────────────────────────────────────────────────────
        if (!window.__radianceBackdropListenerBound) {
            window.addEventListener("keydown", e => {
                // Alt + B binds backdrop auto-creation
                if (e.altKey && e.code === "KeyB") {
                    // Prevent defaults (e.g. browser menu bar highlights)
                    e.preventDefault();
                    const selected = Object.values(app.canvas.selected_nodes || {});
                    createSmartBackdrop(selected);
                }
            });
            window.__radianceBackdropListenerBound = true;
        }
    }
});
