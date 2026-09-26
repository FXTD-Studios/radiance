/**
 * radiance_gizmo.js
 * ═══════════════════════════════════════════════════════════════════════════
 * ◎ RADIANCE DYNAMIC GIZMOS  v1  —  SUBGRAPH COLLAPSE HARNESS
 * Enables packaging of any node subgraph into a styled dynamic custom node
 * ═══════════════════════════════════════════════════════════════════════════
 */

import { app } from "../../scripts/app.js";

import { escapeHtml } from "./radiance_dom_utils.js";

// ─── Constants ────────────────────────────────────────────────────────────────
const PANEL_ID = "radiance-gizmo-modal";
const ACCENT_COLOR = "#00a8ff"; // Radiance blue
const BACKDROP_COLOR = "rgba(15, 15, 20, 0.9)";
const GLASS_BORDER = "1.5px solid rgba(0, 168, 255, 0.25)";

function escapeAttr(value) {
    return escapeHtml(value).replace(/`/g, "&#96;");
}

// ─────────────────────────────────────────────────────────────────────────────
// Toast Notification Helper
// ─────────────────────────────────────────────────────────────────────────────
function showGizmoToast(msg, color = "#00a8ff") {
    const t = document.createElement("div");
    Object.assign(t.style, {
        position: "fixed", top: "24px", left: "50%",
        transform: "translateX(-50%)",
        background: "#0f0f14", color,
        padding: "14px 24px", borderRadius: "8px",
        border: `1.5px solid ${color}44`, zIndex: "10002",
        fontFamily: "'Courier New',monospace", fontSize: "12px",
        fontWeight: "bold", letterSpacing: "0.5px",
        boxShadow: "0 12px 40px rgba(0,0,0,0.9)",
        pointerEvents: "none",
        transition: "all 0.3s ease",
    });
    t.textContent = msg;
    document.body.appendChild(t);
    setTimeout(() => {
        t.style.opacity = "0";
        setTimeout(() => t.remove(), 400);
    }, 3500);
}

// ─────────────────────────────────────────────────────────────────────────────
// Modal UI Panel Builder
// ─────────────────────────────────────────────────────────────────────────────
function openGizmoCreationModal(selectedNodes) {
    // 1. Evict any existing panels
    document.getElementById(PANEL_ID)?.remove();

    if (selectedNodes.length === 0) {
        showGizmoToast("Select at least one node to create a Gizmo.", "#ff4a4a");
        return;
    }

    const selectedNodeIds = new Set(selectedNodes.map(n => n.id));
    const boundaryInputs = [];
    const boundaryOutputs = [];
    const promotableWidgets = [];

    // 2. Scan selected nodes for boundary ports & widgets
    selectedNodes.forEach(node => {
        const nodeTitle = node.title || node.comfyClass || node.type;

        // Boundary Inputs: unconnected, or linked from outside selection
        if (node.inputs) {
            node.inputs.forEach(input => {
                let isBoundary = false;
                if (input.link !== null && input.link !== undefined) {
                    const link = app.graph.links[input.link];
                    if (link && !selectedNodeIds.has(link.origin_id)) {
                        isBoundary = true;
                    }
                } else {
                    isBoundary = true;
                }

                if (isBoundary) {
                    boundaryInputs.push({
                        nodeId: node.id,
                        nodeTitle,
                        inputName: input.name,
                        inputType: input.type,
                        key: `inp_${node.id}_${input.name}`
                    });
                }
            });
        }

        // Boundary Outputs: unconnected, or linked to outside selection
        if (node.outputs) {
            node.outputs.forEach((output, index) => {
                let isBoundary = false;
                if (output.links && output.links.length > 0) {
                    for (const linkId of output.links) {
                        const link = app.graph.links[linkId];
                        if (link && !selectedNodeIds.has(link.target_id)) {
                            isBoundary = true;
                            break;
                        }
                    }
                } else {
                    isBoundary = true;
                }

                if (isBoundary) {
                    boundaryOutputs.push({
                        nodeId: node.id,
                        nodeTitle,
                        outputSlot: index,
                        outputName: output.name,
                        outputType: output.type,
                        key: `out_${node.id}_${index}`
                    });
                }
            });
        }

        // Promotable widgets / knobs
        if (node.widgets) {
            node.widgets.forEach(w => {
                if (w.type !== "button" && w.type !== "hidden") {
                    promotableWidgets.push({
                        nodeId: node.id,
                        nodeTitle,
                        widgetName: w.name,
                        widgetType: w.type,
                        widgetValue: w.value,
                        widgetOptions: w.options ? { ...w.options } : {},
                        key: `wid_${node.id}_${w.name}`
                    });
                }
            });
        }
    });

    // 3. Construct overlay backdrop
    const overlay = document.createElement("div");
    overlay.id = PANEL_ID;
    Object.assign(overlay.style, {
        position: "fixed", top: "0", left: "0", width: "100vw", height: "100vh",
        background: "rgba(5, 5, 8, 0.75)", backdropFilter: "blur(12px)",
        zIndex: "10000", display: "flex", justifyContent: "center", alignItems: "center",
        fontFamily: "'Courier New',monospace", color: "#d0d8e4"
    });

    // 4. Construct Glassmorphic Modal dialog
    const modal = document.createElement("div");
    Object.assign(modal.style, {
        background: BACKDROP_COLOR, border: GLASS_BORDER, borderRadius: "14px",
        padding: "30px", width: "700px", maxHeight: "88vh", overflowY: "auto",
        boxShadow: "0 24px 80px rgba(0,0,0,0.95)", display: "flex", flexDirection: "column",
        gap: "20px", transition: "all 0.3s ease"
    });

    const nodeSummaryText = selectedNodes.map(n => n.title || n.comfyClass || n.type).slice(0, 5).join(", ") +
                            (selectedNodes.length > 5 ? ` +${selectedNodes.length - 5} more` : "");

    modal.innerHTML = `
        <!-- Title & Metadata -->
        <div style="display:flex; justify-content:space-between; align-items:flex-start;">
            <div>
                <span style="font-size:16px; font-weight:bold; color:${ACCENT_COLOR}; letter-spacing:1px;">◎ COLLAPSE TO RADIANCE GIZMO</span>
                <div style="font-size:10px; color:#556677; margin-top:4px;">Collapsing subgraph: [ ${escapeHtml(nodeSummaryText)} ]</div>
            </div>
            <button id="gizmo-close-btn" style="background:none; border:none; color:#567; cursor:pointer; font-size:22px; line-height:1;">✕</button>
        </div>

        <!-- Node Setup Row -->
        <div style="display:flex; gap:16px;">
            <div style="flex:1;">
                <label style="font-size:11px; color:#8ba0b8; display:block; margin-bottom:6px;">GIZMO UNIQUE CLASS NAME</label>
                <input id="gizmo-class-name" placeholder="RadianceVFXChromeAb" value="RadianceGizmo_CustomTool"
                       style="width:100%; background:#0b0b0f; border:1px solid #2a3a4c; color:#fff; padding:8px 12px; border-radius:5px; font-family:monospace; font-size:12px;" />
            </div>
            <div style="width:200px;">
                <label style="font-size:11px; color:#8ba0b8; display:block; margin-bottom:6px;">DISCIPLINE / PALETTE</label>
                <select id="gizmo-category" style="width:100%; background:#0b0b0f; border:1px solid #2a3a4c; color:#fff; padding:7px 10px; border-radius:5px; font-family:monospace; font-size:12px;">
                    <option value="FXTD STUDIOS/Radiance/Core" selected>Core</option>
                    <option value="FXTD STUDIOS/Radiance/Load & Save">Load & Save</option>
                    <option value="FXTD STUDIOS/Radiance/Generate">Generate</option>
                    <option value="FXTD STUDIOS/Radiance/Color">Color</option>
                    <option value="FXTD STUDIOS/Radiance/HDR">HDR</option>
                    <option value="FXTD STUDIOS/Radiance/VFX">VFX</option>
                    <option value="FXTD STUDIOS/Radiance/Video">Video</option>
                    <option value="FXTD STUDIOS/Radiance/Upscale">Upscale</option>
                    <option value="FXTD STUDIOS/Radiance/Review">Review</option>
                    <option value="FXTD STUDIOS/Radiance/Pipeline">Pipeline</option>
                    <option value="FXTD STUDIOS/Radiance/Developer">Developer</option>
                </select>
            </div>
        </div>

        <div>
            <label style="font-size:11px; color:#8ba0b8; display:block; margin-bottom:6px;">GIZMO DESCRIPTION</label>
            <input id="gizmo-description" placeholder="Describe what this dynamic pipeline tool does..."
                   style="width:100%; background:#0b0b0f; border:1px solid #2a3a4c; color:#8ba0b8; padding:8px 12px; border-radius:5px; font-family:monospace; font-size:12px;" />
        </div>

        <hr style="border:none; border-top:1px solid #1a2533; margin:5px 0;" />

        <!-- BOUNDARY PORTS CONTAINER -->
        <div style="display:flex; gap:20px; max-height:300px; overflow-y:auto;">
            <!-- Inputs Column -->
            <div style="flex:1; border-right:1px solid #1a2533; padding-right:10px;">
                <span style="font-size:11px; font-weight:bold; color:#7eb8f7; display:block; margin-bottom:8px;">1. EXPOSE BOUNDARY INPUTS</span>
                <div id="gizmo-inputs-list" style="display:flex; flex-direction:column; gap:8px;"></div>
            </div>

            <!-- Outputs Column -->
            <div style="flex:1;">
                <span style="font-size:11px; font-weight:bold; color:#a27ef7; display:block; margin-bottom:8px;">2. EXPOSE BOUNDARY OUTPUTS</span>
                <div id="gizmo-outputs-list" style="display:flex; flex-direction:column; gap:8px;"></div>
            </div>
        </div>

        <hr style="border:none; border-top:1px solid #1a2533; margin:5px 0;" />

        <!-- PROMOTED PARAMETERS WIDGETS -->
        <div>
            <span style="font-size:11px; font-weight:bold; color:#f7c07e; display:block; margin-bottom:8px;">3. PROMOTE CONTROL WIDGETS / KNOBS</span>
            <div id="gizmo-widgets-list" style="display:flex; flex-direction:column; gap:6px; max-height:220px; overflow-y:auto; padding-right:6px;"></div>
        </div>

        <!-- Buttons Footer -->
        <div style="display:flex; gap:10px; justify-content:flex-end; padding-top:10px; border-top:1px solid #1a2533; margin-top:5px;">
            <button id="gizmo-cancel-btn" style="padding:8px 22px; background:#11141c; border:1px solid #2a3a4c; color:#88aabf; border-radius:5px; cursor:pointer; font-family:monospace; font-size:12px;">Cancel</button>
            <button id="gizmo-save-btn" style="padding:8px 26px; background:#005cbf; border:1px solid #00a8ff; color:#fff; border-radius:5px; cursor:pointer; font-weight:bold; font-family:monospace; font-size:12px;">◎ Collapse to Gizmo</button>
        </div>
    `;

    // 5. Populate Expose Inputs
    const inputsList = modal.querySelector("#gizmo-inputs-list");
    if (boundaryInputs.length === 0) {
        inputsList.innerHTML = `<span style="font-size:10px; color:#567;">No unconnected inputs available.</span>`;
    } else {
        boundaryInputs.forEach((item, index) => {
            const div = document.createElement("div");
            Object.assign(div.style, {
                display: "flex", alignItems: "center", gap: "6px", background: "#0e111a",
                padding: "6px 8px", borderRadius: "4px", border: "1px solid #1b2533"
            });
            div.innerHTML = `
                <input type="checkbox" id="${escapeAttr(item.key)}" checked style="accent-color:${ACCENT_COLOR}; cursor:pointer;" />
                <div style="flex:1; min-width:0;">
                    <div style="font-size:8px; color:#567; text-overflow:ellipsis; overflow:hidden; white-space:nowrap;">${escapeHtml(item.nodeTitle)} (#${escapeHtml(item.nodeId)})</div>
                    <span style="font-size:10px; color:#e0e8f0; font-weight:bold;">${escapeHtml(item.inputName)}</span>
                </div>
                <input class="rename-input" placeholder="${escapeAttr(item.inputName)}" value="${escapeAttr(item.inputName)}"
                       style="width:90px; background:#06080e; border:1px solid #233346; color:#00a8ff; font-family:monospace; font-size:9px; padding:3px 5px; border-radius:3px;" />
            `;
            // Keep boundary data references
            div.dataset.nodeId = item.nodeId;
            div.dataset.inputName = item.inputName;
            div.dataset.inputType = item.inputType;
            div.dataset.key = item.key;
            inputsList.appendChild(div);
        });
    }

    // 6. Populate Expose Outputs
    const outputsList = modal.querySelector("#gizmo-outputs-list");
    if (boundaryOutputs.length === 0) {
        outputsList.innerHTML = `<span style="font-size:10px; color:#567;">No unconnected outputs available.</span>`;
    } else {
        boundaryOutputs.forEach((item, index) => {
            const div = document.createElement("div");
            Object.assign(div.style, {
                display: "flex", alignItems: "center", gap: "6px", background: "#0e111a",
                padding: "6px 8px", borderRadius: "4px", border: "1px solid #1b2533"
            });
            div.innerHTML = `
                <input type="checkbox" id="${escapeAttr(item.key)}" checked style="accent-color:#a27ef7; cursor:pointer;" />
                <div style="flex:1; min-width:0;">
                    <div style="font-size:8px; color:#567; text-overflow:ellipsis; overflow:hidden; white-space:nowrap;">${escapeHtml(item.nodeTitle)} (#${escapeHtml(item.nodeId)})</div>
                    <span style="font-size:10px; color:#e0e8f0; font-weight:bold;">${escapeHtml(item.outputName)}</span>
                </div>
                <input class="rename-output" placeholder="${escapeAttr(item.outputName)}" value="${escapeAttr(item.outputName)}"
                       style="width:90px; background:#06080e; border:1px solid #233346; color:#a27ef7; font-family:monospace; font-size:9px; padding:3px 5px; border-radius:3px;" />
            `;
            // Keep boundary data references
            div.dataset.nodeId = item.nodeId;
            div.dataset.outputSlot = item.outputSlot;
            div.dataset.outputName = item.outputName;
            div.dataset.outputType = item.outputType;
            div.dataset.key = item.key;
            outputsList.appendChild(div);
        });
    }

    // 7. Populate Promotable Widgets
    const widgetsList = modal.querySelector("#gizmo-widgets-list");
    if (promotableWidgets.length === 0) {
        widgetsList.innerHTML = `<span style="font-size:10px; color:#567;">No controls available to promote.</span>`;
    } else {
        promotableWidgets.forEach((item, index) => {
            const div = document.createElement("div");
            Object.assign(div.style, {
                display: "flex", alignItems: "center", gap: "10px", background: "#0e111a",
                padding: "6px 12px", borderRadius: "4px", border: "1px solid #1b2533"
            });
            div.innerHTML = `
                <input type="checkbox" id="${escapeAttr(item.key)}" style="accent-color:#f7c07e; cursor:pointer;" />
                <div style="flex:1; min-width:0; display:flex; flex-direction:column;">
                    <span style="font-size:8px; color:#567;">${escapeHtml(item.nodeTitle)} (#${escapeHtml(item.nodeId)})</span>
                    <span style="font-size:11px; color:#e0e8f0; font-weight:bold;">${escapeHtml(item.widgetName)} <span style="font-weight:normal; color:#495f78; font-size:9px;">(${escapeHtml(item.widgetType)})</span></span>
                </div>
                <div style="display:flex; align-items:center; gap:6px;">
                    <span style="font-size:9px; color:#567;">Promoted Label:</span>
                    <input class="rename-widget" placeholder="${escapeAttr(item.widgetName)}" value="${escapeAttr(item.widgetName)}"
                           style="width:140px; background:#06080e; border:1px solid #233346; color:#f7c07e; font-family:monospace; font-size:10px; padding:3px 6px; border-radius:3px;" />
                </div>
            `;
            // Keep widget references
            div.dataset.nodeId = item.nodeId;
            div.dataset.widgetName = item.widgetName;
            div.dataset.widgetType = item.widgetType.toUpperCase();
            div.dataset.widgetValue = item.widgetValue;
            div.dataset.widgetOptions = JSON.stringify(item.widgetOptions);
            div.dataset.key = item.key;
            widgetsList.appendChild(div);
        });
    }

    // 8. Bind Control Button Actions
    const closePanel = () => overlay.remove();
    modal.querySelector("#gizmo-close-btn").addEventListener("click", closePanel);
    modal.querySelector("#gizmo-cancel-btn").addEventListener("click", closePanel);

    modal.querySelector("#gizmo-save-btn").addEventListener("click", async () => {
        const className = modal.querySelector("#gizmo-class-name").value.trim();
        const displayName = className.replace("RadianceGizmo_", "");
        const category = modal.querySelector("#gizmo-category").value;
        const description = modal.querySelector("#gizmo-description").value.trim();

        if (!className) {
            showGizmoToast("Gizmo Unique Class Name cannot be empty.", "#ff4a4a");
            return;
        }

        // a) Extract exposed inputs
        const inputs = [];
        inputsList.querySelectorAll("div[data-node-id]").forEach(el => {
            const chk = el.querySelector("input[type='checkbox']");
            if (chk && chk.checked) {
                const renameVal = el.querySelector(".rename-input").value.trim() || el.dataset.inputName;
                inputs.push({
                    name: renameVal,
                    type: el.dataset.inputType,
                    target_node_id: parseInt(el.dataset.nodeId),
                    target_input: el.dataset.inputName
                });
            }
        });

        // b) Extract exposed outputs
        const outputs = [];
        outputsList.querySelectorAll("div[data-node-id]").forEach(el => {
            const chk = el.querySelector("input[type='checkbox']");
            if (chk && chk.checked) {
                const renameVal = el.querySelector(".rename-output").value.trim() || el.dataset.outputName;
                outputs.push({
                    name: renameVal,
                    type: el.dataset.outputType,
                    target_node_id: parseInt(el.dataset.nodeId),
                    target_output_slot: parseInt(el.dataset.outputSlot)
                });
            }
        });

        // c) Extract promoted widgets
        const widgets = [];
        widgetsList.querySelectorAll("div[data-node-id]").forEach(el => {
            const chk = el.querySelector("input[type='checkbox']");
            if (chk && chk.checked) {
                const renameVal = el.querySelector(".rename-widget").value.trim() || el.dataset.widgetName;
                const opts = JSON.parse(el.dataset.widgetOptions || "{}");
                const widgetPayload = {
                    name: renameVal,
                    type: el.dataset.widgetType,
                    target_node_id: parseInt(el.dataset.nodeId),
                    target_widget: el.dataset.widgetName
                };

                // Add useful limits if they exist
                if (opts.min !== undefined) widgetPayload.min = opts.min;
                if (opts.max !== undefined) widgetPayload.max = opts.max;
                if (opts.step !== undefined) widgetPayload.step = opts.step;
                if (opts.values !== undefined) widgetPayload.choices = opts.values;

                // Capture current state as default
                if (el.dataset.widgetValue !== undefined && el.dataset.widgetValue !== "undefined") {
                    try {
                        widgetPayload.default = JSON.parse(el.dataset.widgetValue);
                    } catch (e) {
                        widgetPayload.default = el.dataset.widgetValue;
                    }
                }

                widgets.push(widgetPayload);
            }
        });

        // d) Extract internal nodes state & configuration
        const internalNodes = selectedNodes.map(node => {
            const nodeWidgetState = {};
            if (node.widgets) {
                node.widgets.forEach(w => {
                    if (w.name) {
                        nodeWidgetState[w.name] = w.value;
                    }
                });
            }
            return {
                id: node.id,
                type: node.type || node.comfyClass,
                widgets: nodeWidgetState
            };
        });

        // e) Extract internal links
        const internalLinks = [];
        app.graph.links.forEach(link => {
            if (link && selectedNodeIds.has(link.origin_id) && selectedNodeIds.has(link.target_id)) {
                // Resolve target slot integer to input port name
                const targetNode = app.graph.getNodeById(link.target_id);
                const targetInputName = targetNode?.inputs?.[link.target_slot]?.name;
                if (targetInputName) {
                    internalLinks.push({
                        origin_node_id: link.origin_id,
                        origin_output_slot: link.origin_slot,
                        target_node_id: link.target_id,
                        target_input: targetInputName
                    });
                }
            }
        });

        // f) Fire creation request to backend
        const requestPayload = {
            name: className,
            display_name: displayName,
            category,
            description,
            inputs,
            outputs,
            widgets,
            nodes: internalNodes,
            links: internalLinks
        };

        try {
            const response = await fetch("/radiance/gizmos/create", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(requestPayload)
            });

            const result = await response.json();
            if (result.success) {
                closePanel();
                showGizmoToast(`◎ Gizmo '${className}' compiled successfully! Please restart your ComfyUI server to load it.`, ACCENT_COLOR);
            } else {
                showGizmoToast(result.error || "Failed to create Gizmo.", "#ff4a4a");
            }
        } catch (err) {
            showGizmoToast(`Network Error: ${err.message}`, "#ff4a4a");
        }
    });

    overlay.appendChild(modal);
    document.body.appendChild(overlay);
}

// ─────────────────────────────────────────────────────────────────────────────
// ComfyUI Canvas Context Menu Registration (Warning-Free API)
// ─────────────────────────────────────────────────────────────────────────────
app.registerExtension({
    name: "FXTD.Radiance.Gizmo",

    // Modern ComfyUI Canvas Context Menu Hook
    getCanvasMenuItems(canvas) {
        const items = [];
        const selected = Object.values(app.canvas.selected_nodes || {});
        if (selected.length > 0) {
            items.push(null); // Separator
            items.push({
                content: "◎ Create Radiance Gizmo...",
                callback: () => {
                    openGizmoCreationModal(selected);
                }
            });
        }
        return items;
    },

    // Modern ComfyUI Node Context Menu Hook
    getNodeMenuItems(nodeType, node) {
        // Safe check: handle whether comfyui passes (node) or (nodeType, node)
        const activeNode = node || nodeType;
        const items = [];
        const selected = Object.values(app.canvas.selected_nodes || {});
        if (selected.length > 0) {
            items.push({
                content: "◎ Create Radiance Gizmo...",
                callback: () => {
                    openGizmoCreationModal(selected);
                }
            });
        }
        return items;
    }
});
