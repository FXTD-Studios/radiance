// radiance_workspace.js
// Handles the serialization of the entire ComfyUI app.graph into a secure .rad file

import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const HEADER = "RAD_WORKSPACE_V1::";
const SEARCH_DEBOUNCE_MS = 200;
const TOAST_DURATION_MS = 3000;

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
    .radiance-modal h2 { margin: 0; color: #fff; font-size: 1.5em; letter-spacing: -0.02em; }

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
`;
document.head.appendChild(style);


// ═══════════════════════════════════════════════════════════════════════════════
//                       NODE EXTENSION REGISTRATION
// ═══════════════════════════════════════════════════════════════════════════════

app.registerExtension({
    name: "Radiance.Workspace",

    async beforeRegisterNodeDef(nodeType, nodeData, _app) {
        const comfyClass = nodeType.comfyClass || nodeType.ComfyClass || nodeData.name;
        if (comfyClass !== "RadianceWorkspace") return;

        const origOnNodeCreated = nodeType.prototype.onNodeCreated;

        nodeType.prototype.onNodeCreated = function () {
            origOnNodeCreated?.apply(this, arguments);

            this.addWidget("button", "Export .rad", "export", () => this.exportRadWorkspace());
            this.addWidget("button", "Import .rad", "import", () => this.importRadWorkspace());
            this.addWidget("button", "Save to Library", "save_lib", () => this.saveToLibrary());
            this.addWidget("button", "Open Library", "open_lib", () => this.openLibrary());
            this.addWidget("button", "Docs", "docs_link", () => window.open("https://radiance.fxtd.org/", "_blank"));
            this.addWidget("button", "FXTD Studios", "site_link", () => window.open("https://www.fxtd.org", "_blank"));

            this.color = "#1a1c23";
            this.bgcolor = "#1a1c23";
            this.size = [300, 200];
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
                const defaultName = `workspace_${Date.now()}`;
                let chosenName = prompt("Enter a name for the workflow export:", defaultName);

                if (chosenName === null) return;
                chosenName = chosenName.trim() || defaultName;
                if (!chosenName.endsWith(".rad")) chosenName += ".rad";

                const description = prompt("Optional: Short description for this export:") || "";
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

        // ─── Library: Save ───────────────────────────────────────────

        nodeType.prototype.saveToLibrary = async function () {
            try {
                const defaultName = `workspace_${Date.now()}`;
                const artist = localStorage.getItem('radiance_artist') || "Guest";
                const sanitizedArtist = artist.replace(/[^a-z0-9]/gi, '_').toLowerCase();
                
                let chosenName = prompt("Enter filename (e.g., Experiments/Test01):", `workspace_${Date.now()}`);

                if (chosenName === null || chosenName.trim() === "") return;
                
                if (!chosenName.endsWith(".rad")) chosenName += ".rad";

                const description = prompt("Optional: Enter a short description for this workflow:") ?? "";
                const graphData = app.graph.serialize();

                const response = await api.fetchApi("/radiance/workflows/save", {
                    method: "POST",
                    body: JSON.stringify({
                        filename: chosenName,
                        content: JSON.stringify(graphData),
                        description: description,
                        author: artist
                    }),
                    headers: { "Content-Type": "application/json" },
                });

                if (response.ok) {
                    showToast(`Saved "${chosenName}" to library.`, "success");
                } else {
                    const err = await response.json().catch(() => ({}));
                    showToast(`Save failed: ${err.error || response.statusText}`, "error");
                }
            } catch (e) {
                console.error("[Radiance] Failed to save to library:", e);
                showToast("Save failed — check console.", "error");
            }
        };

        // ─── Portal: Sync ────────────────────────────────────────────


        // ─── Library: Open ───────────────────────────────────────────

        nodeType.prototype.openLibrary = async function () {
            try {
                showToast("Scanning Library...", "info");

                // Fetch local workflows
                const localResp = await api.fetchApi("/radiance/workflows/list");
                const localData = await localResp.json();
                let allWorkflows = (localData.workflows || []).map(w => ({ ...w, origin: "local" }));

                this.showLibraryModal(allWorkflows);
            } catch (e) {
                console.error("[Radiance] Failed to load library:", e);
                showToast("Failed to open Assets Library.", "error");
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

                const success = this.parseAndLoadRadContent(content, append);
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
            // Existing modal logic... 
            // I need to ensure the renderList part uses wf.origin/is_cloud
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

        nodeType.prototype.showLibraryModal = function (initialWorkflows) {
            // Mutable reference that the delete handler can update
            let workflows = initialWorkflows;

            const overlay = document.createElement("div");
            overlay.className = "radiance-modal-overlay";

            const modal = document.createElement("div");
            modal.className = "radiance-modal";

            // Header
            const headerDiv = document.createElement("div");
            headerDiv.className = "radiance-modal-header";
            const title = document.createElement("h2");
            title.textContent = "Radiance Assets Library";
            headerDiv.appendChild(title);

            const refreshBtn = document.createElement("button");
            refreshBtn.className = "radiance-btn";
            refreshBtn.style.marginLeft = "auto";
            refreshBtn.textContent = "↻ Refresh";
            refreshBtn.onclick = async () => {
                showToast("Refreshing Library...", "info");
                const resp = await api.fetchApi("/radiance/workflows/list");
                const data = await resp.json();
                workflows = (data.workflows || []).map(w => ({ ...w, origin: "local" }));
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

            // ── Render list ──
            const renderList = (filter = "") => {
                list.innerHTML = "";
                const lowerFilter = filter.toLowerCase();
                const filtered = workflows.filter((wf) => {
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
                    
                    let categoryLabel = folder === "Root" ? "General" : folder;
                    if (folder.toLowerCase().startsWith("official")) {
                        categoryLabel = "◎ Official Radiance";
                    } else if (folder.toLowerCase().startsWith("system")) {
                        categoryLabel = "⚙ System Workflows";
                    }
                    
                    header.textContent = categoryLabel;
                    groupDiv.appendChild(header);

                    const groupList = document.createElement("ul");
                    groupList.className = "radiance-workflow-list";

                    for (const wf of groups[folder]) {
                        const li = document.createElement("li");
                        li.className = "radiance-workflow-item";

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
                        li.appendChild(info);

                        // Action buttons
                        const actions = document.createElement("div");
                        actions.className = "radiance-actions";

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
                        loadBtn.onclick = () => {
                            if (confirm(`Replace current graph with "${displayName}"?`)) {
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
                                if (confirm(`Overwrite System Workflow "${displayName}" with your current canvas?`)) {
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
                            if (!confirm(`Permanently delete "${wf.filename}"?`)) return;
                            await this.deleteFromLibrary(wf.filename);

                            // Refresh list from server
                            try {
                                const response = await api.fetchApi("/radiance/workflows/list");
                                const data = await response.json();
                                workflows = data.workflows || [];
                            } catch {
                                // Remove locally as fallback
                                workflows = workflows.filter((w) => w.filename !== wf.filename);
                            }
                            renderList(searchInput.value);
                        };

                        actions.appendChild(appendBtn);
                        actions.appendChild(loadBtn);
                        actions.appendChild(delBtn);
                        li.appendChild(actions);
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
    },
});
