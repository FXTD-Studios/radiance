// ◎ Radiance — one page for Home, Workflows, Assets, Shots and Renders.
//
// Replaces the Workflow Library, Project Manager and Assets dashboards. It
// opens in ComfyUI's dashboard overlay (window.showRadianceDashboard) and
// talks to the canvas with the same postMessage protocol they used:
// radiance_load_workflow / radiance_append_workflow / radiance_save_project_version.
// Data shaping lives in studio_model.mjs.

import { escapeHtml as h } from "./radiance_dom_utils.js";
import * as M from "./studio_model.mjs";

const ICON = {
    home: "M3 11 12 4l9 7v9h-6v-6H9v6H3z",
    workflows: "M4 5h7v6H4zM13 5h7v6h-7zM4 13h7v6H4zM13 13h7v6h-7z",
    assets: "M4 5h16v14H4zM4 15l5-5 4 4 3-3 4 4",
    shots: "M3 6h18v12H3zM7 6v12M17 6v12",
    renders: "M12 3v4M12 17v4M3 12h4M17 12h4M6 6l2.5 2.5M15.5 15.5 18 18M6 18l2.5-2.5M15.5 8.5 18 6",
    upload: "M12 15V3M7 8l5-5 5 5M4 15v5h16v-5",
};
const NAV = [["home", "Home"], ["workflows", "Workflows"], ["assets", "Assets"], ["shots", "Shots"], ["renders", "Renders"]];
const svg = (d, size = 18) => `<svg width="${size}" height="${size}" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="${d}"/></svg>`;
const $ = (id) => document.getElementById(id);
const store = {
    get(k) { try { return localStorage.getItem(k); } catch { return null; } },
    set(k, v) { try { localStorage.setItem(k, v); } catch { /* private mode */ } },
};

const S = {
    screen: M.screenFromHash(location.hash),
    query: "",
    wfFilter: "all", asFilter: "all", shFilter: "all",
    wf: null, asset: null, shot: null,
    workflows: [], assets: [], bins: [],
    dashboard: null, projectId: store.get("radiance_studio_project") || "",
    project: { versions: [], outputs: [], notes: [] },
    renders: [], gpu: null,
    loaded: { workflows: false, assets: false, dashboard: false, renders: false },
    errors: {},
};

// ── io ───────────────────────────────────────────────────────────────────────

async function getJSON(url, init) {
    const res = await fetch(url, init);
    let data = null;
    try { data = await res.json(); } catch { /* not JSON */ }
    if (!res.ok) throw new Error(data?.error || `${res.status} ${res.statusText}`);
    return data;
}
const postJSON = (url, body) => getJSON(url, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body || {}) });

let toastTimer = 0;
function toast(text, bad = false) {
    const el = $("stToast");
    el.textContent = text;
    el.classList.toggle("bad", !!bad);
    el.classList.add("show");
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => el.classList.remove("show"), 3200);
}

const canvas = () => (window.parent && window.parent !== window ? window.parent : window.opener) || null;
function tellCanvas(msg) {
    const target = canvas();
    if (!target) { toast("Open Radiance from ComfyUI to use the canvas.", true); return false; }
    target.postMessage(msg, location.origin);
    return true;
}
const closeStudio = () => { canvas()?.postMessage({ type: "radiance_close_dashboard" }, location.origin); };

async function loadWorkflows() {
    try {
        const data = await getJSON("/radiance/workflows/list");
        S.workflows = M.workflowCards(data.workflows);
        S.errors.workflows = "";
    } catch (e) { S.errors.workflows = e.message; }
    S.loaded.workflows = true;
}
async function loadAssets() {
    try {
        const data = await getJSON("/radiance/assets");
        S.assets = data.assets || [];
        S.bins = data.bins || [];
        S.errors.assets = "";
    } catch (e) { S.errors.assets = e.message; }
    S.loaded.assets = true;
}
async function loadDashboard() {
    try {
        S.dashboard = await getJSON("/radiance/projects/dashboard");
        const ids = (S.dashboard.projects || []).map((p) => p.id);
        if (!ids.includes(S.projectId)) S.projectId = S.dashboard.activeProjectId || ids[0] || "";
        await loadProject();
        S.errors.dashboard = "";
    } catch (e) { S.errors.dashboard = e.message; }
    S.loaded.dashboard = true;
}
async function loadProject() {
    const d = S.dashboard;
    if (!S.projectId) { S.project = { versions: [], outputs: [], notes: [] }; return; }
    if (S.projectId === d?.activeProjectId) {
        S.project = { versions: d.versions || [], outputs: d.outputs || [], notes: d.notes || [] };
        return;
    }
    const base = `/radiance/projects/${encodeURIComponent(S.projectId)}`;
    const [v, o, n] = await Promise.all([getJSON(`${base}/versions`), getJSON(`${base}/outputs`), getJSON(`${base}/notes`)]);
    S.project = { versions: v.versions || [], outputs: o.outputs || [], notes: n.notes || [] };
}
async function loadRenders() {
    try {
        S.renders = M.renderRows(await getJSON("/history?max_items=60"));
        S.errors.renders = "";
    } catch (e) { S.errors.renders = e.message; }
    S.loaded.renders = true;
}
async function loadGpu() {
    try { S.gpu = M.gpuSummary(await getJSON("/system_stats")); } catch { S.gpu = null; }
    renderGpu();
}

// ── urls ─────────────────────────────────────────────────────────────────────

const coverUrl = (filename) => `/radiance/workflows/preview?filename=${encodeURIComponent(filename)}`;
const thumbUrl = (path) => `/radiance/assets/thumb?path=${encodeURIComponent(path)}`;
const img = (src, alt = "", label = "") => `<img src="${h(src)}" alt="${h(alt)}" loading="lazy" decoding="async" data-ph="${h(label)}">`;
const ph = (label) => `<span class="st-ph">${h(label)}</span>`;
const shortName = (s) => String(s || "").replace(/\.[^.]+$/, "").slice(0, 2).toUpperCase();

// ── chrome ───────────────────────────────────────────────────────────────────

function shotsList() { return M.shotRows(S.project.versions, S.project.outputs, S.project.notes); }

function renderNav() {
    const shots = shotsList();
    const review = shots.filter((r) => r.status.key === "Review").length;
    const failed = S.renders.filter((r) => r.failed).length;
    $("stNav").innerHTML = NAV.map(([id, label]) => {
        const count = id === "shots" ? review : id === "renders" ? failed : 0;
        const badge = count ? `<span class="st-count${id === "renders" ? " bad" : ""}" aria-label="${count} ${id === "renders" ? "failed" : "in review"}">${count}</span>` : "";
        return `<button type="button" data-go="${id}" ${S.screen === id ? 'aria-current="page"' : ""}>${svg(ICON[id])}<span class="st-label">${label}</span>${badge}</button>`;
    }).join("");
}

function renderProjects() {
    const sel = $("stProject");
    const projects = S.dashboard?.projects || [];
    sel.innerHTML = projects.length
        ? projects.map((p) => `<option value="${h(p.id)}" ${p.id === S.projectId ? "selected" : ""}>${h(p.name)} · ${h(M.plural(p.shots || 0, "shot"))}</option>`).join("")
        : `<option value="">No projects yet</option>`;
    sel.disabled = !projects.length;
}

function renderGpu() {
    const g = S.gpu;
    $("stGpu").hidden = !g;
    if (!g) return;
    $("stGpuName").textContent = g.name;
    $("stGpuName").title = g.name;
    $("stGpuBar").style.width = `${g.percent}%`;
    $("stGpuLabel").textContent = g.label;
}

function renderMe() {
    const name = store.get("radiance_artist") || "";
    $("stMe").textContent = name || "Radiance artist";
    $("stAvatar").textContent = M.initials(name || "Radiance");
}

function render() {
    renderNav();
    renderProjects();
    const body = $("stBody");
    const fn = { home: homeView, workflows: workflowsView, assets: assetsView, shots: shotsView, renders: rendersView }[S.screen];
    body.innerHTML = fn();
    $("studio").setAttribute("aria-busy", "false");
}

const loading = () => `<div class="st-empty"><b>Loading…</b></div>`;
const failed = (what, err) => `<div class="st-empty"><b>Could not load ${h(what)}</b>${h(err)}<br><br><button type="button" class="st-btn st-sm" data-act="reload">Try again</button></div>`;
const empty = (title, text) => `<div class="st-empty"><b>${h(title)}</b>${h(text)}</div>`;
const chips = (defs, cur, key, counts) => `<div class="st-chips" role="group">${defs.map(([id, label]) =>
    `<button type="button" class="st-chip" data-filter="${key}" data-id="${h(id)}" aria-pressed="${cur === id}">${h(label)}${counts && counts[id] != null ? `<span class="n">${counts[id]}</span>` : ""}</button>`).join("")}</div>`;

// ── home ─────────────────────────────────────────────────────────────────────

function homeView() {
    const d = S.dashboard;
    const name = store.get("radiance_artist") || "";
    const shots = shotsList();
    const todo = M.needsYou({ shots, renders: S.renders });
    const cw = d?.continueWorking;
    const projectName = (d?.projects || []).find((p) => p.id === S.projectId)?.name || "";
    // The shot's own latest output behind the title; a workflow cover has text baked in.
    const heroOut = cw?.shot && cw.shot !== "GENERAL"
        ? S.project.outputs.find((o) => String(o.name || "").toUpperCase().includes(cw.shot.toUpperCase())) : null;
    const hero = cw?.filename
        ? `<div class="st-card st-hero">
            ${heroOut ? img(thumbUrl(heroOut.path), heroOut.name, "") : `<div class="st-hero-bg"></div>`}
            <div class="st-hero-text">
              <span class="st-sec" style="color:#cfe5ff">Continue</span>
              <div class="st-hero-title">${h(cw.workflow)}</div>
              <div class="st-hero-sub">${h(cw.shot && cw.shot !== "GENERAL" ? cw.shot + " · " : "")}${h(cw.project)} · edited ${h(String(cw.lastModified).toLowerCase())}</div>
              <div class="st-grow"></div>
              <div class="st-actions" style="max-width:340px"><button type="button" class="st-btn st-primary" data-act="open-wf" data-file="${h(cw.filename)}">Open in ComfyUI</button><button type="button" class="st-btn" data-act="show-wf" data-file="${h(cw.filename)}" style="background:rgba(8,8,10,.5)">Details</button></div>
            </div></div>`
        : `<div class="st-card st-hero"><div class="st-hero-bg"></div>
            <div class="st-hero-text">
              <span class="st-sec" style="color:#cfe5ff">Start here</span>
              <div class="st-hero-title">Make your first HDR image</div>
              <div class="st-hero-sub">Open the Starter workflow, press Run, and look at the result in the Viewer.</div>
              <div class="st-grow"></div>
              <div><button type="button" class="st-btn st-primary" data-act="new">Open the Starter</button></div>
            </div></div>`;
    const todoHtml = todo.length
        ? todo.map((t) => `<div class="st-todo-item"><span class="st-dot ${t.tone}"></span><div class="st-todo-text"><span>${h(t.title)}</span><span title="${h(t.text)}">${h(t.text)}</span></div>
            <button type="button" class="st-btn st-sm" data-go="${t.go.screen}" data-shot="${h(t.go.shot || "")}">${h(t.action)}</button></div>`).join("")
        : `<div class="st-todo-item"><span class="st-dot ok"></span><div class="st-todo-text"><span>Nothing is waiting</span><span>No shots in review, no failed renders</span></div></div>`;
    const starts = ["official/Local/sdr_to_hdr.rad", "official/Local/sdr_to_hdr_analytic.rad", "official/Local/start.rad"]
        .map((f) => S.workflows.find((w) => w.filename.replace(/\\/g, "/") === f)).filter(Boolean);
    const startText = { sdr_to_hdr: ["From a photo or video", "SDR to HDR with RUDRA ACES 2"], sdr_to_hdr_analytic: ["Without a model", "The analytic route, any GPU"], start: ["Generate one", "The Starter workflow"] };
    const startsHtml = starts.map((w) => {
        const [t, s] = startText[w.filename.split("/").pop().replace(".rad", "")] || [w.title, w.kicker];
        return `<button type="button" class="st-tile st-start" data-act="show-wf" data-file="${h(w.filename)}"><span class="st-thumb">${img(coverUrl(w.filename), "", "")}</span>
          <span class="st-tile-text" style="padding:0"><span class="st-tile-title">${h(t)}</span><span class="st-tile-sub">${h(s)}</span></span></button>`;
    }).join("");
    const recent = (S.project.outputs.length ? S.project.outputs : S.assets.filter((a) => a.previewable || M.isHdrAsset(a)).slice(0, 10)).slice(0, 10);
    const recentHtml = recent.length
        ? `<div class="st-recent">${recent.map((o) => {
            const hdr = /\.(exr|hdr)$/i.test(o.name || "") || M.isHdrAsset(o);
            return `<div><div class="st-thumb v16">${img(thumbUrl(o.path), o.name, shortName(o.name))}<span class="st-badge${hdr ? " hdr" : ""}">${hdr ? "HDR" : h(String(o.type || o.format || "").toUpperCase() || "FILE")}</span></div><div class="st-recent-name st-mono" title="${h(o.name)}">${h(o.name)}</div></div>`;
        }).join("")}</div>`
        : empty("No outputs yet", "What you render shows up here.");
    const summary = [projectName, todo.length ? M.plural(todo.length, "thing") + " need you" : ""].filter(Boolean).join(" · ");
    return `<div class="st-page">
      <div><div class="st-h1"><h1>${h(M.greeting())}${name ? ", " + h(name) : ""}</h1></div><div class="st-dim" style="margin-top:4px">${h(summary || "Radiance")}</div></div>
      <div class="st-home-top">${S.loaded.dashboard ? hero : `<div class="st-card st-hero"><div class="st-hero-bg"></div></div>`}
        <div class="st-card st-todo"><div class="st-sec" style="margin-bottom:6px">Needs you</div>${todoHtml}</div></div>
      <section style="display:flex;flex-direction:column;gap:12px"><div class="st-section-head"><span class="st-sec">Start something</span><button type="button" class="st-link" data-go="workflows">All ${S.workflows.length || ""} workflows</button></div>
        ${S.loaded.workflows ? `<div class="st-starts">${startsHtml}</div>` : loading()}</section>
      <section style="display:flex;flex-direction:column;gap:12px"><div class="st-section-head"><span class="st-sec">Recent outputs</span><button type="button" class="st-link" data-go="assets">All assets</button></div>${recentHtml}</section>
    </div>`;
}

// ── workflows ────────────────────────────────────────────────────────────────

function workflowsView() {
    if (!S.loaded.workflows) return `<div class="st-page">${loading()}</div>`;
    if (S.errors.workflows) return `<div class="st-page">${failed("workflows", S.errors.workflows)}</div>`;
    const shown = M.filterWorkflows(S.workflows, S.wfFilter, S.query);
    const sel = S.workflows.find((w) => w.id === S.wf) || shown[0] || null;
    if (sel) S.wf = sel.id;
    const grid = shown.length
        ? `<div class="st-grid">${shown.map((w) => `<button type="button" class="st-tile" data-act="pick-wf" data-file="${h(w.filename)}" aria-pressed="${sel && w.id === sel.id}">
            <span class="st-thumb">${w.hasPreview ? img(coverUrl(w.filename), "", shortName(w.title)) : ph(shortName(w.title))}</span>
            <span class="st-tile-text"><span class="st-kicker">${h(w.kicker)}</span><span class="st-tile-title" title="${h(w.title)}">${h(w.title)}</span><span class="st-tile-sub">${h(w.description || w.filename)}</span></span></button>`).join("")}</div>`
        : empty(S.query ? "Nothing matches your search" : "No workflows here", S.query ? "Try another word, or clear the search." : "Import a .rad or .json file, or save one from the canvas.");
    return `<div class="st-page">
      <div class="st-h1"><h1>Workflows</h1><span class="st-dim">${M.plural(shown.length, "workflow")} · open one to put it on the canvas</span></div>
      ${chips(M.WORKFLOW_FILTERS, S.wfFilter, "wf", M.workflowCounts(S.workflows))}
      ${grid}
    </div>${sel ? workflowRail(sel) : ""}`;
}

function workflowRail(w) {
    const facts = [
        ["Type", w.official ? "Official" : "Saved by you"],
        w.nodes ? ["Nodes", String(w.nodes)] : null,
        w.author ? ["By", w.author] : null,
        !w.official && w.mtime ? ["Saved", M.relativeTime(w.mtime)] : null,
        w.size ? ["Size", M.formatBytes(w.size)] : null,
    ].filter(Boolean);
    return `<aside class="st-rail" aria-label="Workflow details">
      <div class="st-thumb">${w.hasPreview ? img(coverUrl(w.filename), "", shortName(w.title)) : ph(shortName(w.title))}</div>
      <div class="st-rail-body">
        <div><div class="st-kicker">${h(w.kicker)}</div><h2>${h(w.title)}</h2>${w.description ? `<p>${h(w.description)}</p>` : ""}</div>
        ${w.tags.length ? `<div class="st-rail-section"><div class="st-sec">Inside</div><div class="st-tags">${w.tags.map((t) => `<span class="st-tag">${h(t)}</span>`).join("")}</div></div>` : ""}
        <div class="st-facts">${facts.map(([k, v]) => `<div><span>${h(k)}</span><span>${h(v)}</span></div>`).join("")}</div>
        <div class="st-actions"><button type="button" class="st-btn st-primary st-lg" data-act="open-wf" data-file="${h(w.filename)}">Open workflow</button><button type="button" class="st-btn st-lg" data-act="append-wf" data-file="${h(w.filename)}" title="Add its nodes next to what is on the canvas">Add to canvas</button></div>
        ${w.official ? "" : `<div class="st-actions"><button type="button" class="st-btn" data-act="history-wf" data-file="${h(w.filename)}">Earlier saves</button><button type="button" class="st-btn st-danger" data-act="delete-wf" data-file="${h(w.filename)}">Delete</button></div><div id="stHistory"></div>`}
        <div class="st-path st-mono">${h(w.filename)}</div>
      </div></aside>`;
}

// ── assets ───────────────────────────────────────────────────────────────────

function assetsView() {
    if (!S.loaded.assets) return `<div class="st-page">${loading()}</div>`;
    if (S.errors.assets) return `<div class="st-page">${failed("assets", S.errors.assets)}</div>`;
    const shown = M.filterAssets(S.assets, S.asFilter, S.query);
    const sel = S.assets.find((a) => a.id === S.asset) || shown[0] || null;
    if (sel) S.asset = sel.id;
    const counts = Object.fromEntries(M.ASSET_FILTERS.map(([id]) => [id, M.filterAssets(S.assets, id).length]));
    const grid = shown.length
        ? `<div class="st-grid">${shown.map((a) => {
            const hdr = M.isHdrAsset(a);
            return `<button type="button" class="st-tile" data-act="pick-asset" data-id="${h(a.id)}" aria-pressed="${sel && a.id === sel.id}">
              <span class="st-thumb v16">${img(thumbUrl(a.path), a.name, shortName(a.name))}<span class="st-badge${hdr ? " hdr" : ""}">${hdr ? "HDR" : "SDR"} · ${h(a.format)}</span>${a.type === "sequence" ? `<span class="st-badge br st-mono">${h(M.plural(a.frames || 0, "frame"))}</span>` : ""}</span>
              <span class="st-tile-text"><span class="st-tile-title st-mono" style="font-size:12px;font-weight:500" title="${h(a.name)}">${h(a.name)}</span><span class="st-tile-sub">${h(a.size)} · ${h(a.date)}</span></span></button>`;
        }).join("")}</div>`
        : empty(S.query ? "Nothing matches your search" : "No assets yet", "Files in ComfyUI's input and output folders show up here.");
    return `<div class="st-page">
      <div class="st-h1"><h1>Assets</h1><span class="st-dim">${M.plural(shown.length, "file")} · input and output folders, newest first</span></div>
      ${chips(M.ASSET_FILTERS, S.asFilter, "as", counts)}
      <button type="button" class="st-drop" id="stDrop" data-act="upload">${svg(ICON.upload, 16)}Drop images, videos or EXR files here, or click to add them</button>
      ${grid}
    </div>${sel ? assetRail(sel) : ""}`;
}

function assetRail(a) {
    const hdr = M.isHdrAsset(a);
    const stats = a.type === "sequence"
        ? [["Frames", String(a.frames || 0)], ["First", String(a.frame_start ?? "")], ["Last", String(a.frame_end ?? "")]]
        : [["Format", a.format], ["Range", hdr ? "HDR" : "SDR"], ["Size", a.size]];
    const bins = S.bins.filter((b) => (b.assets || []).includes(a.id));
    return `<aside class="st-rail" aria-label="Asset details">
      <div class="st-thumb v16">${img(thumbUrl(a.path), a.name, shortName(a.name))}</div>
      <div class="st-rail-body">
        <div><h2 class="st-mono" style="font-size:15px;font-weight:500">${h(a.name)}</h2><p>${h(M.assetKind(a))}</p></div>
        <div class="st-stats">${stats.map(([k, v]) => `<div class="st-stat"><div>${h(k)}</div><div title="${h(v)}">${h(v)}</div></div>`).join("")}</div>
        <div class="st-facts"><div><span>Added</span><span>${h(a.date)}</span></div><div><span>Bins</span><span>${h(bins.map((b) => b.name).join(", ") || "None")}</span></div></div>
        <div class="st-actions"><button type="button" class="st-btn st-primary st-lg" data-act="copy-path" data-path="${h(a.path)}">Copy path</button>${a.previewable ? `<button type="button" class="st-btn st-lg" data-act="open-asset" data-path="${h(a.path)}">Open full size</button>` : ""}</div>
        <div class="st-rail-section"><div class="st-sec">Add to a bin</div><div class="st-tags">${S.bins.map((b) => {
            const inBin = (b.assets || []).includes(a.id);
            return `<button type="button" class="st-chip" data-act="bin" data-bin="${h(b.id)}" data-id="${h(a.id)}" aria-pressed="${inBin}">${h(b.name)}</button>`;
        }).join("")}<button type="button" class="st-chip" data-act="new-bin" data-id="${h(a.id)}">+ New bin</button></div></div>
        <div class="st-path st-mono">${h(a.path)}</div>
      </div></aside>`;
}

// ── shots ────────────────────────────────────────────────────────────────────

function shotThumb(r) {
    const o = r.outputs[0];
    return o ? img(thumbUrl(o.path), r.shot, r.shot) : ph(r.shot);
}

function shotsView() {
    if (!S.loaded.dashboard) return `<div class="st-page">${loading()}</div>`;
    if (S.errors.dashboard) return `<div class="st-page">${failed("shots", S.errors.dashboard)}</div>`;
    const rows = shotsList();
    const shown = M.filterShots(rows, S.shFilter, S.query);
    const sel = rows.find((r) => r.shot === S.shot) || shown[0] || null;
    if (sel) S.shot = sel.shot;
    const st = M.shotStats(rows);
    const projectName = (S.dashboard?.projects || []).find((p) => p.id === S.projectId)?.name || "";
    const kpi = (k, v, color) => `<div class="st-card st-kpi"><span>${k}</span><span style="color:${color}">${v}</span></div>`;
    const table = shown.length
        ? `<div class="st-card st-table"><div class="st-tr st-head st-sec"><span></span><span>Shot</span><span>Version</span><span>Status</span><span>Updated</span></div>
            ${shown.map((r) => `<button type="button" class="st-tr" data-act="pick-shot" data-shot="${h(r.shot)}" aria-pressed="${sel && r.shot === sel.shot}">
              <span class="st-thumb">${shotThumb(r)}</span>
              <span style="min-width:0;display:flex;flex-direction:column;gap:3px"><span class="st-shot-name">${h(r.shot)}</span><span class="st-dim" style="font-size:12px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis">${h(r.workflow)}</span></span>
              <span class="st-mono" style="font-size:12px">${h(r.version)}</span>
              <span class="st-pill ${r.status.tone}">${h(r.status.label)}</span>
              <span class="st-dim" style="font-size:12px">${h(r.updated)}</span></button>`).join("")}</div>`
        : rows.length ? empty("No shots match", "Try another filter or clear the search.")
            : empty("No shots in this project yet", "Save a workflow with a shot name (for example SH010_grade_v001) and it shows up here.");
    return `<div class="st-page">
      <div class="st-h1"><h1>Shots</h1><span class="st-dim">${h(projectName)}</span><span class="st-grow"></span>
        ${S.projectId ? `<button type="button" class="st-btn st-sm" data-act="save-version">Save canvas as a version</button><button type="button" class="st-btn st-sm" data-act="export">Export package</button>` : ""}</div>
      <div class="st-kpis">${kpi("Shots", st.total, "var(--text)")}${kpi("In progress", st.wip, "var(--warn)")}${kpi("Waiting for review", st.review, "var(--info)")}${kpi("Approved", st.approved, "var(--ok)")}</div>
      ${chips(M.SHOT_FILTERS, S.shFilter, "sh")}
      ${table}
    </div>${sel ? shotRail(sel) : ""}`;
}

function shotRail(r) {
    const notes = r.notes.length ? r.notes : [];
    return `<aside class="st-rail" aria-label="Shot details">
      <div class="st-thumb v16">${shotThumb(r)}</div>
      <div class="st-rail-body">
        <div class="st-row-between"><h2>${h(r.shot)}</h2><span class="st-pill ${r.status.tone}">${h(r.status.label)}</span></div>
        <div class="st-rail-section"><div class="st-sec">Versions</div><div class="st-versions">${r.versions.slice(0, 6).map((v, i) => `
          <div class="st-version"><span class="st-mono" style="width:52px">${h(v.version)}</span><span class="st-dim" style="flex:1;font-size:12px">${h(i === 0 ? "Latest · " : "")}${h(v.updated)}</span>
          <button type="button" class="st-btn st-sm" data-act="open-wf" data-file="${h(v.filename)}">Open</button></div>`).join("")}</div></div>
        ${r.outputs.length ? `<div class="st-rail-section"><div class="st-sec">Outputs</div><div class="st-outs">${r.outputs.slice(0, 6).map((o) => `<div class="st-thumb v16" title="${h(o.name)}">${img(thumbUrl(o.path), o.name, shortName(o.name))}</div>`).join("")}</div></div>` : ""}
        <div class="st-rail-section"><div class="st-sec">Notes</div>${notes.length ? notes.map((n) => `<div class="st-note">${h(n)}</div>`).join("") : `<div class="st-dim" style="font-size:13px">No notes for this shot.</div>`}</div>
        <div class="st-actions"><button type="button" class="st-btn st-primary st-lg" data-act="status" data-shot="${h(r.shot)}" data-status="Approved">Approve</button><button type="button" class="st-btn st-lg" data-act="status" data-shot="${h(r.shot)}" data-status="Retake">Ask for changes</button></div>
        <div class="st-actions"><button type="button" class="st-btn" data-act="status" data-shot="${h(r.shot)}" data-status="Review">Send to review</button><button type="button" class="st-btn" data-act="status" data-shot="${h(r.shot)}" data-status="WIP">Back to in progress</button></div>
      </div></aside>`;
}

// ── renders ──────────────────────────────────────────────────────────────────

function rendersView() {
    if (!S.loaded.renders) return `<div class="st-page">${loading()}</div>`;
    if (S.errors.renders) return `<div class="st-page">${failed("renders", S.errors.renders)}</div>`;
    const rows = S.renders;
    const list = rows.length
        ? `<div class="st-card st-table"><div class="st-render st-head st-sec"><span></span><span>Result</span><span>Took</span><span>When</span></div>
          ${rows.map((r) => {
            const im = r.images[0];
            const pill = r.failed ? `<span class="st-pill bad">Failed</span>` : `<span class="st-pill ok">Done</span>`;
            return `<div class="st-render"><span class="st-thumb">${im ? img(M.viewUrl(im), im.filename, "") : ph(r.failed ? "!" : "—")}</span>
              <span style="min-width:0;display:flex;flex-direction:column;gap:4px">${pill}<span class="st-dim" style="font-size:12px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis" title="${h(r.error || (im ? im.filename : ""))}">${h(r.error || (im ? `${M.plural(r.images.length, "image")} · ${im.filename}` : "No images"))}</span></span>
              <span class="st-mono" style="font-size:12px">${r.seconds ? h(r.seconds < 60 ? r.seconds.toFixed(1) + " s" : Math.round(r.seconds / 60) + " min") : "—"}</span>
              <span class="st-dim" style="font-size:12px">${r.started ? h(M.relativeTime(r.started / 1000)) : ""}</span></div>`;
          }).join("")}</div>`
        : empty("No renders yet", "Press Run in ComfyUI and each render shows up here.");
    return `<div class="st-page"><div class="st-h1"><h1>Renders</h1><span class="st-dim">This ComfyUI session, newest first</span><span class="st-grow"></span><button type="button" class="st-btn st-sm" data-act="reload">Refresh</button></div>${list}</div>`;
}

// ── actions ──────────────────────────────────────────────────────────────────

async function openWorkflow(filename, append = false) {
    try {
        const data = await getJSON(`/radiance/workflows/get?filename=${encodeURIComponent(filename)}`);
        if (!data?.success) throw new Error(data?.error || "not found");
        const name = filename.split(/[\\/]/).pop().replace(/\.(rad|json)$/i, "");
        if (tellCanvas({ type: append ? "radiance_append_workflow" : "radiance_load_workflow", content: data.content, name })) {
            toast(append ? `Added ${name} to the canvas` : `Opened ${name}`);
            if (!append) setTimeout(closeStudio, 350);
        }
    } catch (e) { toast(`Could not open the workflow: ${e.message}`, true); }
}

async function showHistory(filename) {
    const box = $("stHistory");
    if (!box) return;
    box.innerHTML = `<div class="st-dim" style="font-size:13px">Loading…</div>`;
    try {
        const data = await getJSON(`/radiance/workflows/history?filename=${encodeURIComponent(filename)}`);
        const list = data.history || [];
        box.innerHTML = list.length
            ? `<div class="st-versions">${list.slice(0, 8).map((v) => `<div class="st-version"><span style="flex:1;min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap" title="${h(v.message || v.version_file)}">${h(v.message || v.version_file)}</span>
                <button type="button" class="st-btn st-sm" data-act="restore-wf" data-file="${h(filename)}" data-version="${h(v.version_file)}">Restore</button></div>`).join("")}</div>`
            : `<div class="st-dim" style="font-size:13px">No earlier saves.</div>`;
    } catch (e) { box.innerHTML = `<div class="st-dim" style="font-size:13px">${h(e.message)}</div>`; }
}

async function importFile(file) {
    const ext = file.name.split(".").pop().toLowerCase();
    if (!["rad", "json"].includes(ext)) { toast("Import takes .rad or .json workflow files", true); return; }
    try {
        let content;
        if (ext === "rad") {
            const res = await getJSON("/radiance/workflows/unpack", { method: "POST", headers: { "Content-Type": "application/octet-stream" }, body: await file.arrayBuffer() });
            if (!res.success) throw new Error(res.error || "could not read the file");
            content = typeof res.content === "string" ? res.content : JSON.stringify(res.content);
        } else {
            content = await file.text();
        }
        const parsed = JSON.parse(content);
        const base = file.name.replace(/\.(rad|json)$/i, "").replace(/[^\w.-]+/g, "_") || "workflow";
        await postJSON("/radiance/workflows/save", {
            filename: `IMPORTED/${base}.rad`, content, description: `Imported from ${file.name}`,
            message: "Imported in Radiance", author: store.get("radiance_artist") || "Radiance artist",
            stats: { node_count: Array.isArray(parsed.nodes) ? parsed.nodes.length : 0, saved_at: Date.now() }, preview_image: "",
        });
        toast(`Imported ${file.name}`);
        await loadWorkflows();
        S.screen = "workflows"; S.wfFilter = "mine"; S.wf = `IMPORTED/${base}.rad`;
        go("workflows");
    } catch (e) { toast(`Import failed: ${e.message}`, true); }
}

async function uploadAssets(files) {
    const list = [...files];
    if (!list.length) return;
    const form = new FormData();
    for (const f of list) form.append("file", f, f.name);
    try {
        const res = await getJSON("/radiance/assets/upload", { method: "POST", body: form });
        const n = (res.saved || []).length;
        toast(n ? `Added ${M.plural(n, "file")}` : "None of those files is an image, video or EXR", !n);
        await loadAssets();
        render();
    } catch (e) { toast(`Upload failed: ${e.message}`, true); }
}

async function setShotStatus(shot, status) {
    try {
        await postJSON(`/radiance/projects/${encodeURIComponent(S.projectId)}/shots/${encodeURIComponent(shot)}/status`, { status });
        toast(`${shot}: ${M.shotStatus(status).label}`);
        await loadDashboard();
        render();
    } catch (e) { toast(`Could not change the status: ${e.message}`, true); }
}

async function exportPackage() {
    try {
        const res = await fetch(`/radiance/projects/${encodeURIComponent(S.projectId)}/export-package`, { method: "POST" });
        if (!res.ok) throw new Error((await res.json().catch(() => ({}))).error || res.statusText);
        const blob = await res.blob();
        const a = document.createElement("a");
        a.href = URL.createObjectURL(blob);
        a.download = `${S.projectId}_radiance_package.zip`;
        a.click();
        setTimeout(() => URL.revokeObjectURL(a.href), 2000);
    } catch (e) { toast(`Export failed: ${e.message}`, true); }
}

async function binToggle(binId, assetId) {
    const b = S.bins.find((x) => x.id === binId);
    const inBin = (b?.assets || []).includes(assetId);
    try {
        await postJSON(`/radiance/assets/bins/${encodeURIComponent(binId)}`, { action: inBin ? "remove" : "add", asset_id: assetId });
        await loadAssets();
        render();
    } catch (e) { toast(`Could not change the bin: ${e.message}`, true); }
}

async function newBin(assetId) {
    const name = (prompt("Name the new bin") || "").trim();
    if (!name) return;
    try {
        const res = await postJSON("/radiance/assets/bins", { name });
        if (res.bin?.id && assetId) await postJSON(`/radiance/assets/bins/${encodeURIComponent(res.bin.id)}`, { action: "add", asset_id: assetId });
        await loadAssets();
        render();
    } catch (e) { toast(`Could not make the bin: ${e.message}`, true); }
}

async function reloadScreen() {
    const jobs = { home: [loadDashboard, loadWorkflows, loadRenders], workflows: [loadWorkflows], assets: [loadAssets], shots: [loadDashboard], renders: [loadRenders] }[S.screen] || [];
    await Promise.all(jobs.map((f) => f()));
    render();
}

function go(screen, extra = {}) {
    S.screen = M.SCREENS.includes(screen) ? screen : "home";
    Object.assign(S, extra);
    if (location.hash !== `#${S.screen}`) history.replaceState(null, "", `#${S.screen}`);
    render();
    $("stBody").querySelector(".st-page")?.scrollTo?.(0, 0);
}

async function onClick(e) {
    const t = e.target.closest("[data-go],[data-act],[data-filter]");
    if (!t) return;
    if (t.dataset.go) {
        go(t.dataset.go, t.dataset.shot ? { shot: t.dataset.shot, shFilter: "all" } : {});
        return;
    }
    if (t.dataset.filter) {
        S[{ wf: "wfFilter", as: "asFilter", sh: "shFilter" }[t.dataset.filter]] = t.dataset.id;
        render();
        return;
    }
    const { act, file, id, path, shot, status } = t.dataset;
    switch (act) {
        case "pick-wf": S.wf = file; render(); break;
        case "show-wf": S.wf = file; S.wfFilter = "all"; go("workflows"); break;
        case "open-wf": await openWorkflow(file); break;
        case "append-wf": await openWorkflow(file, true); break;
        case "history-wf": await showHistory(file); break;
        case "restore-wf":
            if (!confirm("Restore this earlier save? What is saved now is kept as a backup.")) break;
            try { await postJSON("/radiance/workflows/restore", { filename: file, version_file: t.dataset.version }); toast("Restored"); await loadWorkflows(); render(); }
            catch (err) { toast(`Restore failed: ${err.message}`, true); }
            break;
        case "delete-wf":
            if (!confirm(`Delete ${file.split(/[\\/]/).pop()} from the library?`)) break;
            try { await postJSON("/radiance/workflows/delete", { filename: file }); toast("Deleted"); S.wf = null; await loadWorkflows(); render(); }
            catch (err) { toast(`Delete failed: ${err.message}`, true); }
            break;
        case "pick-asset": S.asset = id; render(); break;
        case "upload": $("stUploadFile").click(); break;
        case "copy-path":
            try { await navigator.clipboard.writeText(path); toast("Path copied"); } catch { prompt("Copy the path", path); }
            break;
        case "open-asset": window.open(thumbUrl(path), "_blank", "noopener"); break;
        case "bin": await binToggle(t.dataset.bin, id); break;
        case "new-bin": await newBin(id); break;
        case "pick-shot": S.shot = shot; render(); break;
        case "status": await setShotStatus(shot, status); break;
        case "save-version":
            if (tellCanvas({ type: "radiance_save_project_version", projectId: S.projectId })) toast("Saving the canvas…");
            break;
        case "export": await exportPackage(); break;
        case "new": await openWorkflow("official/Local/start.rad"); break;
        case "reload": await reloadScreen(); break;
        default: break;
    }
}

// ── wire up ──────────────────────────────────────────────────────────────────

function wire() {
    document.addEventListener("click", (e) => { onClick(e); });
    // A thumbnail that will not load becomes a lettered placeholder.
    document.addEventListener("error", (e) => {
        const el = e.target;
        if (el?.tagName === "IMG" && el.dataset.ph != null) el.replaceWith(Object.assign(document.createElement("span"), { className: "st-ph", textContent: el.dataset.ph }));
    }, true);
    $("stImport").onclick = () => $("stImportFile").click();
    $("stNew").onclick = () => openWorkflow("official/Local/start.rad");
    $("stImportFile").onchange = (e) => { const f = e.target.files?.[0]; e.target.value = ""; if (f) importFile(f); };
    $("stUploadFile").onchange = (e) => { const fs = e.target.files; if (fs?.length) uploadAssets(fs).finally(() => { e.target.value = ""; }); };
    $("stFree").onclick = async () => {
        try { await getJSON("/free", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ unload_models: true, free_memory: true }) }); toast("Freed GPU memory"); setTimeout(loadGpu, 600); }
        catch (err) { toast(`Could not free memory: ${err.message}`, true); }
    };
    $("stProject").onchange = async (e) => {
        S.projectId = e.target.value;
        store.set("radiance_studio_project", S.projectId);
        S.shot = null;
        try { await loadProject(); } catch (err) { toast(err.message, true); }
        render();
    };
    let searchTimer = 0;
    $("stSearch").addEventListener("input", (e) => {
        clearTimeout(searchTimer);
        searchTimer = setTimeout(() => {
            S.query = e.target.value;
            if (S.screen === "home" && S.query) S.screen = "workflows";
            render();
        }, 150);
    });
    document.addEventListener("keydown", (e) => {
        if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "k") { e.preventDefault(); $("stSearch").focus(); $("stSearch").select(); return; }
        if (e.key === "Escape") {
            if (document.activeElement === $("stSearch") && $("stSearch").value) { $("stSearch").value = ""; S.query = ""; render(); return; }
            closeStudio();
        }
    });
    window.addEventListener("hashchange", () => go(M.screenFromHash(location.hash)));
    // Files dropped anywhere on the Assets screen are uploaded.
    document.addEventListener("dragover", (e) => { if (S.screen === "assets" && e.dataTransfer?.types?.includes("Files")) { e.preventDefault(); $("stDrop")?.classList.add("over"); } });
    document.addEventListener("dragleave", (e) => { if (!e.relatedTarget) $("stDrop")?.classList.remove("over"); });
    document.addEventListener("drop", (e) => {
        if (S.screen !== "assets" || !e.dataTransfer?.files?.length) return;
        e.preventDefault();
        $("stDrop")?.classList.remove("over");
        uploadAssets(e.dataTransfer.files);
    });
    window.addEventListener("message", (e) => {
        if (e.origin !== location.origin || e.data?.type !== "radiance_project_action_result") return;
        if (e.data.success) { toast("Saved the canvas as a new version"); loadDashboard().then(render); }
        else toast(`Save failed: ${e.data.error || "unknown error"}`, true);
    });
}

async function start() {
    wire();
    renderMe();
    render();
    loadGpu();
    // Each source renders as soon as it arrives; a slow one does not hold up the rest.
    const after = (p) => p.then(render, render);
    await Promise.all([after(loadWorkflows()), after(loadDashboard()), after(loadAssets()), after(loadRenders())]);
}

start();
