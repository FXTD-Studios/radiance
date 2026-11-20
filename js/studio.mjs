// ◎ Radiance — one page for Home, Workflows, Gizmos, Assets, Shots, Renders and Models.
//
// Replaces the Workflow Library, Project Manager and Assets dashboards. It
// opens in ComfyUI's dashboard overlay (window.showRadianceDashboard) and
// talks to the canvas with the same postMessage protocol they used:
// radiance_load_workflow / radiance_append_workflow / radiance_save_project_version.
// Data shaping lives in studio_model.mjs.

import { escapeHtml as h } from "./radiance_dom_utils.js";
import * as M from "./studio_model.mjs";
import { displayCheck, probeDisplay } from "./radiance_welcome_logic.js";

const ICON = {
    home: "M3 11 12 4l9 7v9h-6v-6H9v6H3z",
    workflows: "M4 5h7v6H4zM13 5h7v6h-7zM4 13h7v6H4zM13 13h7v6h-7z",
    assets: "M4 5h16v14H4zM4 15l5-5 4 4 3-3 4 4",
    shots: "M3 6h18v12H3zM7 6v12M17 6v12",
    renders: "M12 3v4M12 17v4M3 12h4M17 12h4M6 6l2.5 2.5M15.5 15.5 18 18M6 18l2.5-2.5M15.5 8.5 18 6",
    upload: "M12 15V3M7 8l5-5 5 5M4 15v5h16v-5",
    gizmos: "M12 3 4 7.5v9L12 21l8-4.5v-9zM4 7.5 12 12l8-4.5M12 12v9",
    models: "M4 6c0-1.7 3.6-3 8-3s8 1.3 8 3-3.6 3-8 3-8-1.3-8-3zM4 6v12c0 1.7 3.6 3 8 3s8-1.3 8-3V6M4 12c0 1.7 3.6 3 8 3s8-1.3 8-3",
};
const NAV = [["home", "Home"], ["workflows", "Workflows"], ["gizmos", "Gizmos"], ["assets", "Assets"], ["shots", "Shots"], ["renders", "Renders"], ["models", "Models"]];
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
    gizmos: [], gizmo: null, gizmoFolder: "",
    setup: null, downloads: [], scanning: false,
    loaded: { workflows: false, assets: false, dashboard: false, renders: false, gizmos: false, setup: false },
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
async function loadGizmos() {
    try {
        const data = await getJSON("/radiance/gizmos/list");
        if (data && data.success === false) throw new Error(data.error || "could not list gizmos");
        S.gizmos = M.gizmoCards(data.gizmos);
        S.gizmoFolder = data.folder || "";
        S.errors.gizmos = "";
    } catch (e) { S.errors.gizmos = e.message; }
    S.loaded.gizmos = true;
}
let displayProbe = null;
async function loadSetup(refresh = false) {
    S.scanning = true;
    try {
        S.setup = await postJSON("/radiance/setup/scan", { refresh });
        S.downloads = S.setup.downloads || [];
        S.errors.setup = "";
    } catch (e) { S.errors.setup = e.message; }
    S.scanning = false;
    S.loaded.setup = true;
}
let pollTimer = 0;
function pollDownloads() {
    clearTimeout(pollTimer);
    pollTimer = setTimeout(async () => {
        try {
            const before = S.downloads.filter((j) => j.status === "done").length;
            S.downloads = (await getJSON("/radiance/setup/downloads")).downloads || [];
            const active = M.downloadsSummary(S.downloads).active;
            // A download finished: scan again so the workflow rows update.
            if (S.downloads.filter((j) => j.status === "done").length !== before || !active) await loadSetup();
            if (S.screen === "models") render();
            if (active) pollDownloads();
        } catch { pollDownloads(); }
    }, 1000);
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
    const fn = { home: homeView, workflows: workflowsView, gizmos: gizmosView, assets: assetsView, shots: shotsView, renders: rendersView, models: modelsView }[S.screen];
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

// ── brand covers ─────────────────────────────────────────────────────────────
// Drawn the way the workflow cards are: the RADIANCE 4 lock-up, a blue kicker,
// the title, one line under it and a mono footer, art on the right. Sized in
// container units, so a tile and the rail show the same picture.

const LOGO = `<svg viewBox="0 0 100 100" aria-hidden="true"><rect width="100" height="100" rx="22.5" fill="#1C1C1E"/><path d="M52.2 28.4A24 24 0 1 0 71.6 47.8" fill="none" stroke="#F5F5F7" stroke-width="14"/><rect x="58" y="28" width="14" height="14" rx="3" fill="#0A84FF"/></svg>`;

function cover({ kicker, title, sub = "", foot = [], badge = "", art = "", cls = "" }) {
    return `<span class="st-cover ${cls}" aria-hidden="true"><span class="st-cover-art">${art}</span>
      <span class="st-cover-lock">${LOGO}<b>RADIANCE 4</b></span>${badge ? `<span class="st-cover-badge">${h(badge)}</span>` : ""}
      <span class="st-cover-text"><span class="st-cover-kick">${h(kicker)}</span><span class="st-cover-title">${h(title)}</span>${sub ? `<span class="st-cover-sub">${h(sub)}</span>` : ""}</span>
      ${foot.length ? `<span class="st-cover-foot">${foot.map(h).join("<i>/</i>")}</span>` : ""}</span>`;
}

/** A gizmo as what it is: wires in, its nodes packed into one, wires out. */
function gizmoArt(g) {
    const n = Math.max(1, Math.min(4, g.nodes || 1));
    const ins = Math.max(1, Math.min(3, g.inputs.length || 1));
    const outs = Math.max(1, Math.min(3, g.outputs.length || 1));
    const box = { x: 92, y: 18, w: 76, h: 64 };
    const pin = (count, i) => box.y + 14 + ((box.h - 28) * (count === 1 ? 0.5 : i / (count - 1)));
    const wiresIn = Array.from({ length: ins }, (_, i) => `<path d="M20 ${pin(ins, i)}C56 ${pin(ins, i)} 56 ${pin(ins, i)} ${box.x} ${pin(ins, i)}"/><circle cx="20" cy="${pin(ins, i)}" r="2.6"/>`).join("");
    const wiresOut = Array.from({ length: outs }, (_, i) => `<path d="M${box.x + box.w} ${pin(outs, i)}C${box.x + box.w + 18} ${pin(outs, i)} ${box.x + box.w + 18} ${pin(outs, i)} 196 ${pin(outs, i)}"/><circle cx="196" cy="${pin(outs, i)}" r="2.6"/>`).join("");
    const cols = n > 2 ? 2 : n, rows = Math.ceil(n / cols);
    const cw = (box.w - 14 - (cols - 1) * 5) / cols, ch = (box.h - 22 - (rows - 1) * 5) / rows;
    const inner = Array.from({ length: n }, (_, i) => {
        const x = box.x + 7 + (i % cols) * (cw + 5), y = box.y + 15 + Math.floor(i / cols) * (ch + 5);
        return `<rect x="${x}" y="${y}" width="${cw}" height="${ch}" rx="2.5" fill="#26262a" stroke="#3a3a3c" stroke-width=".6"/><rect x="${x}" y="${y}" width="${cw}" height="4.5" rx="2" fill="#303034"/>`;
    }).join("");
    return `<svg viewBox="0 0 200 100" preserveAspectRatio="xMaxYMid meet">
      <defs><radialGradient id="gzGlow" cx="65%" cy="50%" r="50%"><stop offset="0" stop-color="#0A84FF" stop-opacity=".30"/><stop offset="1" stop-color="#0A84FF" stop-opacity="0"/></radialGradient></defs>
      <rect width="200" height="100" fill="url(#gzGlow)"/>
      <g fill="none" stroke="#409cff" stroke-width="1.4" stroke-linecap="round">${wiresIn}${wiresOut}</g>
      <rect x="${box.x}" y="${box.y}" width="${box.w}" height="${box.h}" rx="6" fill="#141419" stroke="#0A84FF" stroke-width="1.2"/>
      <rect x="${box.x}" y="${box.y}" width="${box.w}" height="10" rx="5" fill="#0A84FF" fill-opacity=".28"/>
      <text x="${box.x + 6}" y="${box.y + 7.4}" font-size="5.6" font-family="JetBrains Mono, monospace" fill="#cfe5ff" letter-spacing=".6">◎ GIZMO</text>
      ${inner}
      <g fill="#409cff">${Array.from({ length: ins }, (_, i) => `<circle cx="${box.x}" cy="${pin(ins, i)}" r="2.2"/>`).join("")}${Array.from({ length: outs }, (_, i) => `<circle cx="${box.x + box.w}" cy="${pin(outs, i)}" r="2.2"/>`).join("")}</g>
    </svg>`;
}

function gizmoCover(g, cls = "") {
    const types = (list) => [...new Set(list.map((p) => String(p.type || "").toUpperCase()).filter(Boolean))].join("+") || "—";
    return cover({ kicker: "Gizmo", title: g.title, sub: g.description && !g.description.startsWith(g.title + ":") ? g.description : "",
                   foot: [M.plural(g.nodes, "node").toUpperCase(), `${types(g.inputs)} → ${types(g.outputs)}`, M.plural(g.knobs.length, "knob").toUpperCase()],
                   badge: "GIZMO", art: gizmoArt(g), cls });
}

/** RUDRA's card, in the SDR | HDR split every RUDRA cover uses. */
function rudraCover(installed) {
    return cover({ kicker: "HDR · RUDRA", title: "RUDRA ACES 2", sub: "SDR to a scene-linear ACES 2.0 master",
                   foot: ["MODELS/RADIANCE", "SHA-256", installed ? "INSTALLED" : "3 MB"], badge: "HDR", cls: "is-rudra",
                   art: `<span class="st-split"><span class="st-split-sdr"><em>SDR</em></span><span class="st-split-hdr"><em>HDR</em></span><span class="st-split-line"></span></span>` });
}

// ── gizmos ───────────────────────────────────────────────────────────────────

function gizmosView() {
    if (!S.loaded.gizmos) return `<div class="st-page">${loading()}</div>`;
    if (S.errors.gizmos) return `<div class="st-page">${failed("gizmos", S.errors.gizmos)}</div>`;
    const shown = M.filterGizmos(S.gizmos, S.query);
    const sel = S.gizmos.find((g) => g.id === S.gizmo) || shown[0] || null;
    if (sel) S.gizmo = sel.id;
    const restart = S.gizmos.filter((g) => g.status.key === "restart").length;
    const grid = shown.length
        ? `<div class="st-grid">${shown.map((g) => `<button type="button" class="st-tile" data-act="pick-gizmo" data-id="${h(g.id)}" aria-pressed="${sel && g.id === sel.id}">
            <span class="st-thumb">${gizmoCover(g)}</span>
            <span class="st-tile-text"><span class="st-kicker">Gizmo</span><span class="st-tile-title" title="${h(g.title)}">${h(g.title)}</span>
            <span class="st-tile-sub"><span class="st-pill ${g.status.tone}">${h(g.status.label)}</span></span></span></button>`).join("")}</div>`
        : S.query ? empty("Nothing matches your search", "Try another word, or clear the search.")
            : `<div class="st-empty"><b>No gizmos yet</b>Select some nodes on the canvas, right-click and choose ◎ Create Radiance Gizmo… It is saved here, and in the node menu after a restart.<br><br><button type="button" class="st-btn st-sm" data-act="import-gizmo">Import a .gizmo file</button></div>`;
    return `<div class="st-page">
      <div class="st-h1"><h1>Gizmos</h1><span class="st-dim">${M.plural(shown.length, "gizmo")} · your node setups packed into one node</span><span class="st-grow"></span>
        <button type="button" class="st-btn st-sm" data-act="import-gizmo">Import .gizmo</button><button type="button" class="st-btn st-sm" data-act="reload">Refresh</button></div>
      ${restart ? `<div class="st-card st-notice"><span class="st-dot warn"></span>${h(M.plural(restart, "gizmo"))} saved since ComfyUI started: restart ComfyUI to find ${restart === 1 ? "it" : "them"} in the node menu under Radiance › ◎ Gizmos.</div>` : ""}
      ${grid}
      ${S.gizmoFolder ? `<div class="st-path st-mono">${h(S.gizmoFolder)}</div>` : ""}
    </div>${sel ? gizmoRail(sel) : ""}`;
}

function gizmoRail(g) {
    const ports = (list) => list.length ? list.map((p) => `<span class="st-tag" title="${h(p.type || "")}">${h(p.name)}${p.type ? ` · ${h(p.type)}` : ""}</span>`).join("") : `<span class="st-dim" style="font-size:13px">None</span>`;
    const facts = [["Nodes inside", String(g.nodes)], ["Saved", g.mtime ? M.relativeTime(g.mtime) : ""], ["Size", g.size ? M.formatBytes(g.size) : ""]].filter(([, v]) => v);
    return `<aside class="st-rail" aria-label="Gizmo details">
      <div class="st-thumb">${gizmoCover(g)}</div>
      <div class="st-rail-body">
        <div><div class="st-kicker">Gizmo</div><h2>${h(g.title)}</h2>${g.description ? `<p>${h(g.description)}</p>` : ""}<span class="st-pill ${g.status.tone}">${h(g.status.label)}</span></div>
        ${g.error ? `<div class="st-note">${h(g.error)}</div>` : ""}
        <div class="st-rail-section"><div class="st-sec">Inputs</div><div class="st-tags">${ports(g.inputs)}</div></div>
        <div class="st-rail-section"><div class="st-sec">Outputs</div><div class="st-tags">${ports(g.outputs)}</div></div>
        <div class="st-rail-section"><div class="st-sec">Knobs</div><div class="st-tags">${g.knobs.length ? g.knobs.map((k) => `<span class="st-tag">${h(k)}</span>`).join("") : `<span class="st-dim" style="font-size:13px">None</span>`}</div></div>
        ${g.nodeTypes.length ? `<div class="st-rail-section"><div class="st-sec">Made of</div><div class="st-tags">${g.nodeTypes.map((t) => `<span class="st-tag">${h(t)}</span>`).join("")}</div></div>` : ""}
        <div class="st-facts">${facts.map(([k, v]) => `<div><span>${h(k)}</span><span>${h(v)}</span></div>`).join("")}</div>
        <div class="st-actions"><button type="button" class="st-btn st-primary st-lg" data-act="add-gizmo" data-id="${h(g.id)}" ${g.status.key === "ready" ? "" : `disabled title="Restart ComfyUI first"`}>Add to canvas</button><button type="button" class="st-btn st-lg" data-act="export-gizmo" data-id="${h(g.id)}">Export .gizmo</button></div>
        <div class="st-actions"><button type="button" class="st-btn st-danger" data-act="delete-gizmo" data-id="${h(g.id)}">Delete</button></div>
        <div class="st-path st-mono">${h(g.file)}</div>
      </div></aside>`;
}

// ── models (setup check) ─────────────────────────────────────────────────────

// First-start checks: the server's four, plus the display, which only the browser can see.
function checksCard(sc) {
    if (!displayProbe) displayProbe = probeDisplay(window);
    const list = [...(sc.checks || []), displayCheck(displayProbe)];
    const label = { ok: "OK", warn: "Check", bad: "Fix needed" };
    const tone = { ok: "ok", warn: "warn", bad: "bad" };
    const fix = (c) => !c.fix ? ""
        : c.fix.action ? `<button type="button" class="st-btn st-sm" data-act="setup-fix" data-id="${h(c.fix.action)}">${h(c.fix.label)}</button>`
            : c.fix.url ? `<a class="st-link" href="${h(c.fix.url)}" target="_blank" rel="noopener">${h(c.fix.label)}</a>` : "";
    const issues = list.filter((c) => c.status !== "ok").length;
    return `<div class="st-card st-checks"><div class="st-row-between" style="margin-bottom:8px"><div class="st-sec">First start</div>
        <span class="st-dim st-small">${issues ? h(M.plural(issues, "thing")) + " to look at" : "Everything checks out"}</span></div>
      ${list.map((c) => `<div class="st-check" data-check="${h(c.id)}"><span class="st-dot ${tone[c.status] || "info"}"></span>
        <div class="st-grow"><div class="st-check-title">${h(c.title)} <span class="st-pill ${tone[c.status] || "info"}">${h(label[c.status] || c.status)}</span></div>
        <div class="st-dim st-small">${h(c.detail)}</div></div>${fix(c)}</div>`).join("")}</div>`;
}

function modelsView() {
    if (!S.loaded.setup) return `<div class="st-page"><div class="st-h1"><h1>Models</h1></div>${loading()}</div>`;
    if (S.errors.setup) return `<div class="st-page">${failed("the setup check", S.errors.setup)}</div>`;
    const sc = S.setup;
    const hw = sc.hardware || {};
    const tier = M.gpuTier(hw);
    const jobs = S.downloads || [];
    const byId = Object.fromEntries(jobs.map((j) => [j.id, j]));
    const rows = M.setupRows(sc);
    const core = (sc.core || [])[0];
    const dl = M.downloadsSummary(jobs);
    const off = !sc.downloads_allowed;
    const jobLine = (id) => {
        const j = byId[id];
        if (!j) return "";
        const p = M.jobProgress(j);
        return `<div class="st-job"><div class="st-meter"><div style="width:${p.pct ?? 0}%"></div></div><span class="st-dim st-small">${h(p.text)}</span>
          ${j.status === "running" || j.status === "queued" ? `<button type="button" class="st-link" data-act="cancel-dl" data-id="${h(id)}">Cancel</button>` : ""}
          ${j.error ? `<div class="st-note">${h(j.error)}</div>` : ""}</div>`;
    };
    const model = (id) => {
        const m = sc.models[id] || { title: id };
        return `<div class="st-model"><div class="st-model-head"><span class="st-dot ${m.installed ? "ok" : m.gated ? "warn" : "info"}"></span>
          <span class="st-mono st-model-name" title="${h(m.path || "")}">${h(m.title)}</span>
          <span class="st-dim st-small">${h(m.kind === "radiance" ? "models/radiance" : "models/" + (m.folder || ""))}</span><span class="st-grow"></span>
          <span class="st-dim st-small">${h(m.installed ? "Installed" : M.formatSize(m.size) + (m.size && !m.size_exact ? " (about)" : ""))}</span>
          ${m.installed || off ? "" : `<button type="button" class="st-btn st-sm" data-act="dl-ids" data-id="${h(id)}">Download</button>`}
          ${m.gated && !m.installed ? `<a class="st-link" href="${h(m.page)}" target="_blank" rel="noopener">Licence</a>` : ""}</div>${jobLine(id)}</div>`;
    };
    const card = (r) => `<div class="st-card st-setup-row${r.status.key === "ready" ? " is-ready" : ""}">
        <div class="st-setup-head"><span class="st-thumb">${img(coverUrl(`official/Local/${r.id}.rad`), "", shortName(r.title))}</span>
        <div class="st-row-between st-grow"><div><div class="st-tile-title">${h(r.title)}</div>
          <div class="st-dim st-small">${r.needsNothing ? "Needs no models" : r.status.key === "ready" ? `${M.plural(r.required.length, "model")} installed` : `${h(M.plural(r.missing.length, "model"))} missing${r.size ? " · " + h(r.size) : ""}`}</div></div>
          <div class="st-actions" style="margin:0"><span class="st-pill ${r.status.tone}">${h(r.status.label)}</span>
          ${r.status.key !== "ready" && !off ? `<button type="button" class="st-btn st-primary st-sm" data-act="dl-workflow" data-id="${h(r.id)}">Download missing</button>` : ""}</div></div></div>
        ${r.note ? `<div class="st-note">${h(r.note)}</div>` : ""}
        ${r.missing.length ? `<details ${r.missing.some((m) => byId[m.id]) ? "open" : ""}><summary class="st-dim st-small">What is missing</summary>${r.missing.map((m) => model(m.id)).join("")}</details>` : ""}
      </div>`;
    return `<div class="st-page">
      <div class="st-h1"><h1>Models</h1><span class="st-dim">This computer, and what each workflow needs</span><span class="st-grow"></span>
        <button type="button" class="st-btn st-sm" data-act="rescan" ${S.scanning ? "disabled" : ""}>${S.scanning ? "Checking…" : "Check again"}</button></div>
      <div class="st-card st-rudra-hero"><div class="st-thumb">${rudraCover(core && sc.models[core.id]?.installed)}</div>
        <div class="st-rudra-body"><div class="st-sec">Radiance's own models</div>
          ${core ? model(core.id) : ""}
          <div class="st-facts"><div><span>Folder</span><span class="st-mono st-tail" title="${h(sc.paths.radiance)}"><bdi>${h(sc.paths.radiance)}</bdi></span></div>
          <div><span>Hugging Face token</span><span>${sc.hf_token ? "Found" : "Not set"}</span></div>
          <div><span>Downloads</span><span>${off ? "Off (RADIANCE_ALLOW_DOWNLOADS=0 or offline)" : "On"}</span></div></div>
          ${dl.active ? `<div class="st-note">${h(M.plural(dl.active, "download"))} running${dl.percent != null ? ` · ${dl.percent}%` : ""}</div>` : ""}</div>
      </div>
      <div class="st-home-top">
        <div class="st-card"><div class="st-sec" style="margin-bottom:8px">This computer</div>
          <div class="st-facts">${M.hardwareRows(hw).map(([k, v]) => `<div><span>${h(k)}</span><span title="${h(v)}">${h(v)}</span></div>`).join("")}</div>
          <div class="st-note"><span class="st-dot ${tier.tone}"></span> ${h(tier.text)}</div></div>
        <div class="st-kpis-col"><div class="st-card st-kpi"><span>Local workflows</span><span>${sc.summary.workflows}</span></div><div class="st-card st-kpi"><span>Ready to run</span><span style="color:var(--ok)">${sc.summary.ready}</span></div>
          <div class="st-card st-kpi"><span>Need downloads</span><span style="color:var(--info)">${sc.summary.workflows - sc.summary.ready}</span></div></div>
      </div>
      ${checksCard(sc)}
      <div class="st-dim st-small">Files go where ComfyUI looks for them (models/diffusion_models, text_encoders, vae, loras…); Radiance's own models go in models/radiance. Each file is checked against its SHA-256 before it is installed. 🔒 models need their licence accepted on Hugging Face and a token (HF_TOKEN, or <span class="st-mono">hf auth login</span>), then a restart.</div>
      <div class="st-setup">${rows.map(card).join("")}</div>
    </div>`;
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
        ${r.outputs.length ? `<div class="st-rail-section"><div class="st-sec">Outputs</div><div class="st-outs">${r.outputs.slice(0, 6).map((o) => `<div class="st-thumb v16" title="${h(M.outputLabel(o))}">${img(thumbUrl(o.path), o.name, shortName(o.name))}${o.frames ? `<span class="st-badge br st-mono">${h(M.plural(o.frames, "frame"))}</span>` : ""}</div>`).join("")}</div></div>` : ""}
        <div class="st-rail-section"><div class="st-sec">Notes</div>${notes.length ? notes.map((n) => `<div class="st-note">${h(n)}</div>`).join("") : `<div class="st-dim" style="font-size:13px">No notes for this shot.</div>`}
          <label class="st-visually-hidden" for="stNoteText">Note for ${h(r.shot)}</label>
          <textarea id="stNoteText" class="st-note-input" rows="3" placeholder="What needs to change? A retake sends this note with it."></textarea>
          <div class="st-actions" style="margin-top:6px"><button type="button" class="st-btn st-sm" data-act="add-note" data-shot="${h(r.shot)}">Add note</button></div></div>
        <div class="st-actions"><button type="button" class="st-btn st-primary st-lg" data-act="status" data-shot="${h(r.shot)}" data-status="Approved">Approve</button><button type="button" class="st-btn st-lg" data-act="retake" data-shot="${h(r.shot)}">Ask for changes</button></div>
        <div class="st-actions"><button type="button" class="st-btn" data-act="status" data-shot="${h(r.shot)}" data-status="Review">Send to review</button><button type="button" class="st-btn" data-act="status" data-shot="${h(r.shot)}" data-status="WIP">Back to in progress</button></div>
        ${r.status.key === "Approved" ? `<div class="st-actions"><button type="button" class="st-btn" data-act="status" data-shot="${h(r.shot)}" data-status="Final">Mark final</button></div>` : ""}
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

/** Add the typed note to a shot; with `retake`, the note is required and the shot goes to Retake. */
async function addNote(shot, retake) {
    const box = $("stNoteText");
    const text = (box?.value || "").trim();
    if (!text) {
        if (retake) { toast("Write what needs to change first; the note goes with the retake.", true); box?.focus(); }
        return;
    }
    try {
        await postJSON(`/radiance/projects/${encodeURIComponent(S.projectId)}/notes`,
            { shot, text, author: store.get("radiance_artist") || "" });
        if (retake) await postJSON(`/radiance/projects/${encodeURIComponent(S.projectId)}/shots/${encodeURIComponent(shot)}/status`, { status: "Retake" });
        toast(retake ? `${shot}: retake, with your note` : `${shot}: note added`);
        await loadDashboard();
        render();
    } catch (e) { toast(`Could not save the note: ${e.message}`, true); }
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

async function startDownloads(body) {
    try {
        const res = await postJSON("/radiance/setup/download", body);
        S.downloads = res.downloads || S.downloads;
        toast(res.queued?.length ? `Downloading ${M.plural(res.queued.length, "model")}` : "Nothing to download");
        render();
        pollDownloads();
    } catch (e) { toast(`Could not start the download: ${e.message}`, true); }
}

async function importGizmo(file) {
    try {
        const data = JSON.parse(await file.text());
        const res = await postJSON("/radiance/gizmos/import", data);
        toast(res.message || `Imported ${file.name}`);
        await loadGizmos();
        S.gizmo = res.name || S.gizmo;
        render();
    } catch (e) { toast(`Import failed: ${e.message}`, true); }
}

async function reloadScreen() {
    const jobs = { home: [loadDashboard, loadWorkflows, loadRenders], workflows: [loadWorkflows], gizmos: [loadGizmos], assets: [loadAssets], shots: [loadDashboard], renders: [loadRenders], models: [loadSetup] }[S.screen] || [];
    await Promise.all(jobs.map((f) => f()));
    render();
}

function go(screen, extra = {}) {
    S.screen = M.SCREENS.includes(screen) ? screen : "home";
    if (S.screen === "models" && !S.loaded.setup && !S.scanning) loadSetup().then(() => { render(); if (M.downloadsSummary(S.downloads).active) pollDownloads(); });
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
        case "add-note": await addNote(shot, false); break;
        case "retake": await addNote(shot, true); break;
        case "save-version":
            if (tellCanvas({ type: "radiance_save_project_version", projectId: S.projectId, shot: S.shot || "" })) toast("Saving the canvas…");
            break;
        case "export": await exportPackage(); break;
        case "new": await openWorkflow("official/Local/start.rad"); break;
        case "pick-gizmo": S.gizmo = id; render(); break;
        case "add-gizmo": {
            const g = S.gizmos.find((x) => x.id === id);
            if (g && tellCanvas({ type: "radiance_add_node", nodeType: g.className, title: g.title })) toast(`Adding ${g.title}…`);
            break;
        }
        case "export-gizmo": {
            const a = document.createElement("a");
            a.href = `/radiance/gizmos/file?name=${encodeURIComponent(id)}`;
            a.download = `${id}.gizmo`;
            a.click();
            break;
        }
        case "delete-gizmo":
            if (!confirm(`Delete the gizmo ${id}? It is moved to the gizmos/.trash folder; its node stays until ComfyUI restarts.`)) break;
            try { await postJSON("/radiance/gizmos/delete", { name: id }); toast("Gizmo deleted"); S.gizmo = null; await loadGizmos(); render(); }
            catch (err) { toast(`Delete failed: ${err.message}`, true); }
            break;
        case "import-gizmo": $("stGizmoFile").click(); break;
        case "rescan": displayProbe = null; render(); await loadSetup(true); render(); break;
        case "setup-fix":
            try {
                const res = await postJSON("/radiance/setup/fix", { action: id });
                S.setup.checks = (S.setup.checks || []).map((c) => (c.id === res.check.id ? res.check : c));
                render();
            } catch (err) { toast(err.message, true); }
            break;
        case "dl-workflow": await startDownloads({ workflow: id }); break;
        case "dl-ids": await startDownloads({ ids: [id] }); break;
        case "cancel-dl":
            try { S.downloads = (await postJSON("/radiance/setup/cancel", { id })).downloads || S.downloads; render(); }
            catch (err) { toast(err.message, true); }
            break;
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
    $("stGizmoFile").onchange = (e) => { const f = e.target.files?.[0]; e.target.value = ""; if (f) importGizmo(f); };
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
    window.addEventListener("hashchange", () => {
        go(M.screenFromHash(location.hash));
        if (S.screen === "models" && !S.loaded.setup) loadSetup().then(render);
    });
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
    // The setup check scans the disk and the GPU: only when its page is opened.
    if (S.screen === "models") after(loadSetup()).then(() => { if (M.downloadsSummary(S.downloads).active) pollDownloads(); });
    await Promise.all([after(loadWorkflows()), after(loadDashboard()), after(loadAssets()), after(loadRenders()), after(loadGizmos())]);
}

start();
