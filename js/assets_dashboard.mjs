// ◎ Radiance — Assets manager (dark minimal). Live data from /radiance/assets.

// ALBABIT-FIX: resolve extension base at runtime so asset paths work regardless of the install folder name
import { escapeHtml } from "./radiance_dom_utils.js";
const _EXT_BASE = import.meta.url.replace(/\/[^/]+$/, '');

const API = {
    list: "/radiance/assets",
    bins: "/radiance/assets/bins",
    bin: (id) => `/radiance/assets/bins/${id}`,
    thumb: (path) => `/radiance/assets/thumb?path=${encodeURIComponent(path)}`,
    upload: "/radiance/assets/upload",
};

const mock = { assets: [], bins: [], counts: { all: 0, image: 0, video: 0, sequence: 0 }, source: "Loading" };

let state = { data: mock, activeBin: "all", query: "" };

function playSvg() { return '<svg viewBox="0 0 24 24" fill="currentColor"><path d="M8 5v14l11-7z"/></svg>'; }
function searchSvg() { return '<svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="#52555c" stroke-width="1.6"><circle cx="11" cy="11" r="7"/><path d="M21 21l-4-4"/></svg>'; }

function tagClass(type) { return type === "video" ? "ast-video" : type === "sequence" ? "ast-sequence" : type === "hdri" ? "ast-hdri" : "ast-image"; }

function visibleAssets() {
    const d = state.data;
    let list = d.assets || [];
    const b = state.activeBin;
    if (b === "images") list = list.filter((a) => a.type === "image" || a.type === "hdri");
    else if (b === "videos") list = list.filter((a) => a.type === "video");
    else if (b === "sequences") list = list.filter((a) => a.type === "sequence");
    else if (b !== "all") {
        const bin = (d.bins || []).find((x) => x.id === b);
        const ids = new Set(bin ? bin.assets || [] : []);
        list = list.filter((a) => ids.has(a.id));
    }
    const q = state.query.trim().toLowerCase();
    if (q) list = list.filter((a) => String(a.name).toLowerCase().includes(q));
    return list;
}

function Sidebar() {
    const d = state.data, c = d.counts || {};
    const lib = [
        { key: "all", label: "All assets", n: c.all },
        { key: "images", label: "Images", n: c.image },
        { key: "videos", label: "Videos", n: c.video },
        { key: "sequences", label: "Sequences", n: c.sequence },
    ];
    const libHtml = lib.map((it) => `
        <button class="ast-nav-item ${state.activeBin === it.key ? "is-active" : ""}" data-action="Bin" data-bin="${it.key}">
            <span>${it.label}</span><span class="ast-nav-count">${it.n ?? 0}</span>
        </button>`).join("");
    const bins = (d.bins || []).map((b) => `
        <button class="ast-nav-item ${state.activeBin === b.id ? "is-active" : ""}" data-action="Bin" data-bin="${escapeHtml(b.id)}">
            <span>${escapeHtml(b.name)}</span><span class="ast-nav-count">${b.count ?? 0}</span>
        </button>`).join("");
    return `
        <aside class="ast-sidebar">
            <div class="ast-brand"><img src="${_EXT_BASE}/r_icon.png" alt="Radiance"><span>Radiance</span></div>
            <div class="ast-label">Library</div>
            <nav class="ast-nav">${libHtml}</nav>
            <div class="ast-label mt">Custom <button class="ast-newbin" data-action="New Bin" title="New bin">+</button></div>
            <nav class="ast-nav">${bins || `<div class="ast-label" style="color:var(--faint);padding-top:4px;">no bins yet</div>`}</nav>
            <div class="ast-foot">${c.all ?? 0} assets<br>${escapeHtml(d.source || "")}</div>
        </aside>`;
}

function Card(a) {
    const thumb = `<img src="${API.thumb(a.path)}" loading="lazy" alt="" onerror="this.remove()">`;
    const play = a.type === "video" ? `<span class="ast-play">${playSvg()}</span>` : "";
    const range = a.type === "sequence" ? `<span class="ast-range">${a.frame_start}–${a.frame_end}</span>` : "";
    return `
        <div class="ast" data-action="Open Asset" data-id="${escapeHtml(a.id)}">
            <div class="ast-thumb ${a.type === "sequence" ? "is-seq" : ""}">
                ${thumb}
                <span class="ast-tag ${tagClass(a.type)}">${escapeHtml(a.format || "")}</span>
                ${play}${range}
            </div>
            <div class="ast-name" title="${escapeHtml(a.name)}">${escapeHtml(a.name)}</div>
            <div class="ast-meta">${escapeHtml(a.meta || "")}</div>
            <div class="ast-sub">${escapeHtml(a.size || "")} · ${escapeHtml(a.date || "")}</div>
        </div>`;
}

function gridInner() {
    const list = visibleAssets();
    const cards = list.length ? list.map(Card).join("") : `<div class="ast-empty">No assets in this view.</div>`;
    return cards + `<label class="ast-drop" data-action="Drop">+ Drop files<input type="file" multiple hidden data-upload accept="image/*,video/*,.exr,.hdr,.tif,.tiff"></label>`;
}

function renderGrid() {
    const grid = document.querySelector(".ast-grid");
    if (grid) grid.innerHTML = gridInner();
}

function Main() {
    return `
        <main class="ast-main">
            <header class="ast-topbar">
                <div>
                    <div class="ast-crumb">Library</div>
                    <div class="ast-title dsp">Assets</div>
                </div>
                <div class="ast-tools">
                    <label class="ast-search">${searchSvg()}<input data-search type="search" placeholder="Search assets…" value="${escapeHtml(state.query)}"></label>
                    <button class="ast-import" data-action="Import">Import &rarr;</button>
                </div>
            </header>
            <div class="ast-content">
                <div class="ast-grid">${gridInner()}</div>
            </div>
        </main>`;
}

function render() {
    const mount = document.getElementById("radiance-assets");
    if (!mount) return;
    mount.innerHTML = Sidebar() + Main();
}

function showToast(msg) {
    let t = document.querySelector(".ast-toast");
    if (!t) { t = document.createElement("div"); t.className = "ast-toast"; document.body.appendChild(t); }
    t.textContent = msg; t.classList.add("is-visible");
    clearTimeout(showToast._t);
    showToast._t = setTimeout(() => t.classList.remove("is-visible"), 1900);
}

async function fetchJson(url, opt) {
    const r = await fetch(url, opt);
    if (!r.ok) throw new Error((await r.json().catch(() => ({}))).error || r.statusText);
    return r.json();
}

async function load() {
    try {
        state.data = await fetchJson(API.list);
    } catch (e) {
        state.data = { ...mock, source: "Offline" };
        showToast("Could not reach the asset scanner.");
    }
    render();
}

// ── Drawer ──
function AssetDrawer(a) {
    const d = state.data;
    const preview = `<img src="${API.thumb(a.path)}" alt="" onerror="this.style.display='none';this.nextElementSibling.style.display='block'"><span style="display:none">${escapeHtml(String(a.type))} preview not available</span>`;
    const kv = (k, v) => v ? `<div class="ast-kv"><span>${k}</span><span title="${escapeHtml(v)}">${escapeHtml(v)}</span></div>` : "";
    const seq = a.type === "sequence" ? kv("Frames", `${a.frame_start}–${a.frame_end} (${a.frames})`) : "";
    const binBtns = (d.bins || []).map((b) => `<button class="ast-binbtn" data-action="Add To Bin" data-bin="${escapeHtml(b.id)}" data-id="${escapeHtml(a.id)}">${escapeHtml(b.name)}</button>`).join("")
        || `<span style="color:var(--faint);font-size:12px;">No bins yet — create one in the sidebar.</span>`;
    return `
        <div class="ast-bd" data-action="Close Asset"></div>
        <aside class="ast-panel" role="dialog" aria-label="${escapeHtml(a.name)}">
            <header class="ast-ph">
                <div><div class="ast-crumb">${escapeHtml(String(a.type).toUpperCase())}</div><div class="ast-ph-title">${escapeHtml(a.name)}</div></div>
                <button class="ast-x" data-action="Close Asset" title="Close">&#10005;</button>
            </header>
            <div class="ast-pb">
                <div class="ast-preview">${preview}</div>
                ${kv("Format", a.format)}
                ${kv("Size", a.size)}
                ${kv("Updated", a.date)}
                ${seq}
                ${kv("Path", a.path)}
                <div class="ast-sec-label">Add to bin</div>
                <div>${binBtns}</div>
            </div>
        </aside>`;
}
function openAsset(id) {
    const a = (state.data.assets || []).find((x) => x.id === id);
    if (!a) return;
    const root = document.getElementById("ast-drawer");
    root.innerHTML = AssetDrawer(a);
    requestAnimationFrame(() => root.classList.add("is-open"));
}
function closeAsset() {
    const root = document.getElementById("ast-drawer");
    root.classList.remove("is-open");
    setTimeout(() => { root.innerHTML = ""; }, 220);
}

async function createBin() {
    const name = window.prompt("New bin name");
    if (!name || !name.trim()) return;
    try {
        await fetchJson(API.bins, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ name: name.trim() }) });
        await load();
        showToast(`Bin "${name.trim()}" created.`);
    } catch (e) { showToast(e.message); }
}
async function addToBin(binId, assetId) {
    try {
        await fetchJson(API.bin(binId), { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ action: "add", asset_id: assetId }) });
        await load();
        showToast("Added to bin.");
    } catch (e) { showToast(e.message); }
}
async function uploadFiles(files) {
    if (!files || !files.length) return;
    const fd = new FormData();
    [...files].forEach((f) => fd.append("file", f, f.name));
    showToast(`Uploading ${files.length} file(s)…`);
    try {
        const res = await fetchJson(API.upload, { method: "POST", body: fd });
        await load();
        showToast(`Imported ${(res.saved || []).length} file(s).`);
    } catch (e) { showToast(e.message); }
}

// ── Events ──
document.addEventListener("click", (e) => {
    const btn = e.target.closest("[data-action]");
    if (!btn) return;
    const action = btn.dataset.action;
    if (action === "Bin") { state.activeBin = btn.dataset.bin; render(); return; }
    if (action === "New Bin") { createBin(); return; }
    if (action === "Open Asset") { openAsset(btn.dataset.id); return; }
    if (action === "Close Asset") { closeAsset(); return; }
    if (action === "Add To Bin") { addToBin(btn.dataset.bin, btn.dataset.id); closeAsset(); return; }
    if (action === "Import") { document.querySelector("[data-upload]")?.click(); return; }
});
document.addEventListener("input", (e) => {
    if (e.target.matches("[data-search]")) { state.query = e.target.value; renderGrid(); }
});
document.addEventListener("change", (e) => {
    if (e.target.matches("[data-upload]")) { uploadFiles(e.target.files); }
});
document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") {
        const dr = document.getElementById("ast-drawer");
        if (dr && dr.classList.contains("is-open")) { closeAsset(); return; }
        if (window.parent && window.parent !== window && window.parent.closeRadianceDashboard) window.parent.closeRadianceDashboard();
    }
});

render();
load();
