// ALBABIT-FIX: resolve extension base at runtime so asset paths work regardless of the install folder name
import { escapeHtml } from "./radiance_dom_utils.js";
const _EXT_BASE = import.meta.url.replace(/\/[^/]+$/, '');

const futureApi = {
    dashboard: "/radiance/projects/dashboard",
    projects: "/radiance/projects",
    recentProjects: "/radiance/projects/recent",
    versions: (projectId) => `/radiance/projects/${projectId}/versions`,
    outputs: (projectId) => `/radiance/projects/${projectId}/outputs`,
    notes: (projectId) => `/radiance/projects/${projectId}/notes`,
    saveVersion: (projectId) => `/radiance/projects/${projectId}/save-version`,
    exportPackage: (projectId) => `/radiance/projects/${projectId}/export-package`,
};

const mockProjectManagerData = {
    user: {
        name: "Ahmed",
        role: "Artist",
    },
    storage: {
        usedLabel: "412 GB",
        totalLabel: "640 GB",
        percent: 64,
    },
    continueWorking: {
        workflow: "SH010_HDR_Grade_v004",
        project: "HDR_Test",
        shot: "SH010",
        status: "Approved",
        lastModified: "2 hours ago",
    },
    projects: [
        { id: "hdr-test", name: "HDR_Test", shots: 12, workflows: 28, updated: "Today", favorite: true },
        { id: "commercial-01", name: "Commercial_01", shots: 18, workflows: 41, updated: "Yesterday", favorite: false },
        { id: "lookdev-project", name: "LookDev_Project", shots: 6, workflows: 13, updated: "3 days ago", favorite: true },
        { id: "personal-studies", name: "Personal_Studies", shots: 9, workflows: 19, updated: "Last week", favorite: false },
    ],
    versions: [
        { version: "v004", status: "Approved", shot: "SH010" },
        { version: "v003", status: "Review", shot: "SH020" },
        { version: "v002", status: "WIP", shot: "SH030" },
        { version: "v001", status: "On Hold", shot: "SH040" },
    ],
    outputs: [
        { name: "SH010_beauty_v004.exr", type: "EXR", size: "164 MB", date: "Today" },
        { name: "SH010_preview_v004.png", type: "PNG", size: "8 MB", date: "Today" },
        { name: "SH020_comp_v003.exr", type: "EXR", size: "142 MB", date: "Yesterday" },
        { name: "SH020_preview_v003.png", type: "PNG", size: "7 MB", date: "Yesterday" },
        { name: "SH030_light_v002.exr", type: "EXR", size: "188 MB", date: "3 days ago" },
    ],
    notes: [
        "Highlight clipping in SH010",
        "Check skin tones in SH020",
        "Reduce noise in shadows",
    ],
    quickActions: [
        "Save Version",
        "Compare Versions",
        "Import Workflow",
        "Export Package",
        "Project Settings",
    ],
    nav: [
        "Dashboard",
        "Projects",
        "Shots",
        "Workflows",
        "Versions",
        "Review",
        "Assets",
        "Settings",
    ],
    activeProjectId: "hdr-test",
    source: "Mock fallback",
};

const NAV = [
    { key: "Dashboard", icon: "dashboard", active: true },
    { key: "Projects", icon: "folders" },
    { key: "Shots", icon: "shots" },
    { key: "Assets", icon: "assets" },
    { key: "Reviews", icon: "review", badge: "review" },
    { key: "Renders", icon: "renders" },
    { key: "Library", icon: "library" },
];

let activeView = "Dashboard";

function svg(name) {
    const paths = {
        dashboard: '<rect x="3" y="3" width="7" height="9" rx="1"/><rect x="14" y="3" width="7" height="5" rx="1"/><rect x="14" y="12" width="7" height="9" rx="1"/><rect x="3" y="16" width="7" height="5" rx="1"/>',
        folders: '<path d="M3 7h5l2 2h9a1 1 0 0 1 1 1v8a1 1 0 0 1-1 1H4a1 1 0 0 1-1-1V7z"/>',
        shots: '<rect x="3" y="5" width="18" height="14" rx="2"/><path d="M3 9h18M8 5v14"/>',
        assets: '<path d="M12 3l8 4.5v9L12 21l-8-4.5v-9L12 3z"/><path d="M12 12l8-4.5M12 12v9"/>',
        review: '<circle cx="12" cy="12" r="3"/><path d="M2 12s3.5-7 10-7 10 7 10 7-3.5 7-10 7-10-7-10-7z"/>',
        renders: '<rect x="3" y="4" width="18" height="6" rx="1"/><rect x="3" y="14" width="18" height="6" rx="1"/>',
        library: '<rect x="3" y="5" width="3" height="14"/><rect x="8" y="5" width="3" height="14"/><path d="M14 6l4 13"/>',
        search: '<circle cx="11" cy="11" r="7"/><path d="M21 21l-4-4"/>',
        play: '<path d="M8 5v14l11-7z"/>',
        resume: '<path d="M9 14l-4-4 4-4"/><path d="M5 10h11a4 4 0 0 1 0 8h-1"/>',
        save: '<path d="M5 4h11l3 3v13H5z"/><path d="M8 4v5h7V4M8 20v-6h8v6"/>',
    };
    const fill = name === "play" ? "currentColor" : "none";
    return `<svg viewBox="0 0 24 24" fill="${fill}" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round">${paths[name] || ""}</svg>`;
}

function statusMeta(status) {
    const s = String(status || "").toLowerCase();
    if (s.includes("approve")) return { cls: "is-approved", label: "APPROVED" };
    if (s.includes("review")) return { cls: "is-review", label: "REVIEW" };
    if (s.includes("retake") || s.includes("reject")) return { cls: "is-retake", label: "RETAKE" };
    if (s.includes("final") || s.includes("deliver")) return { cls: "is-final", label: "FINAL" };
    if (s.includes("wip") || s.includes("progress")) return { cls: "is-wip", label: "WIP" };
    return { cls: "is-neutral", label: (status ? String(status) : "—").toUpperCase() };
}

function EmptyState(message) {
    return `<div class="rpm-empty">${escapeHtml(message)}</div>`;
}

function Sidebar(data) {
    const versions = data.versions || [];
    const reviewCount = versions.filter((v) => statusMeta(v.status).cls === "is-review").length;
    const navHtml = NAV.map((n) => {
        const badge = (n.badge === "review" && reviewCount) ? `<span class="rpm-nav-badge">${reviewCount}</span>` : "";
        return `<button class="rpm-nav-item ${n.key === activeView ? "is-active" : ""}" type="button" data-action="Nav" data-nav="${escapeHtml(n.key)}">
            <span class="rpm-nav-icon">${svg(n.icon)}</span><span>${escapeHtml(n.key)}</span>${badge}
        </button>`;
    }).join("");
    const initials = String(data.user?.name || "?").slice(0, 2).toUpperCase();
    return `
        <aside class="rpm-sidebar">
            <div class="rpm-brand">
                <img class="rpm-brand-mark" src="${_EXT_BASE}/r_icon.png" alt="Radiance" />
                <div class="rpm-brand-title">Radiance</div>
            </div>
            <nav class="rpm-nav" aria-label="Project Manager">${navHtml}</nav>
            <div class="rpm-user">
                <span class="rpm-user-avatar">${escapeHtml(initials)}</span>
                <span class="rpm-user-meta">
                    <span class="rpm-user-name">${escapeHtml(data.user?.name || "Artist")}</span>
                    <span class="rpm-user-role">${escapeHtml(data.user?.role || "")}</span>
                </span>
            </div>
        </aside>
    `;
}

function TopBar(data) {
    const projects = data.projects || [];
    const proj = projects.find((p) => p.id === data.activeProjectId) || projects[0];
    const projName = proj?.name || "No project";
    const menu = projects.length ? projects.map((p) => `
        <button class="rpm-proj-opt ${p.id === proj?.id ? "is-active" : ""}" type="button" data-action="Select Project" data-project-id="${escapeHtml(p.id)}">
            <span class="rpm-proj-name">${escapeHtml(p.name)}</span><span class="rpm-dim">${escapeHtml(p.shots)} shots</span>
        </button>`).join("") : `<div class="rpm-empty">No projects yet — save a .rad workflow.</div>`;
    return `
        <header class="rpm-topbar">
            <div class="rpm-show-wrap">
                <button class="rpm-show" type="button" data-action="Toggle Projects">
                    <span class="rpm-nav-icon">${svg("folders")}</span><span>${escapeHtml(projName)}</span><span class="rpm-caret">&#9662;</span>
                </button>
                <div class="rpm-proj-menu" data-proj-menu hidden>${menu}</div>
            </div>
            <label class="rpm-search">${svg("search")}<input data-search type="search" placeholder="Search shots, assets, versions…"></label>
            <button class="rpm-button is-primary rpm-topbtn" type="button" data-action="Save Version">${svg("save")}Save version</button>
            <span class="rpm-sync"><span class="rpm-sync-dot"></span>Synced</span>
        </header>
    `;
}

function Hero(item) {
    const st = statusMeta(item.status);
    const verMatch = String(item.workflow || "").match(/v\d{2,}/i);
    const ver = (verMatch && verMatch[0]) || item.version || "";
    return `
        <section class="rpm-hero">
            <div class="rpm-hero-thumb">
                <span class="rpm-hero-range">1001–1142</span>
                <span class="rpm-hero-play">${svg("play")}</span>
                <span class="rpm-pill ${st.cls} rpm-hero-pill">${st.label}</span>
            </div>
            <div class="rpm-hero-info">
                <div class="rpm-kicker">Continue working</div>
                <div class="rpm-hero-title">${escapeHtml(item.shot || "—")} <span class="rpm-hero-sub">· ${escapeHtml(item.project || "")}</span></div>
                <div class="rpm-hero-meta">${escapeHtml(item.workflow || "No workflow saved yet")}${ver ? ` · <span class="rpm-ver">${escapeHtml(ver)}</span>` : ""}${item.lastModified ? ` · ${escapeHtml(item.lastModified)}` : ""}</div>
                <button class="rpm-button is-primary" type="button" data-action="Open Workflow">${svg("resume")}Resume session</button>
            </div>
        </section>
    `;
}

function StorageCard(storage) {
    const pct = Math.max(0, Math.min(100, Number(storage?.percent) || 0));
    return `
        <section class="rpm-storage-card">
            <div class="rpm-storage-head"><span>Project storage</span><span class="rpm-storage-val">${escapeHtml(storage?.usedLabel || "")} / ${escapeHtml(storage?.totalLabel || "")}</span></div>
            <div class="rpm-storage-track"><span class="rpm-storage-fill" style="width:${pct}%"></span></div>
            <div class="rpm-storage-foot">${pct}% used</div>
        </section>
    `;
}

function Stats(data) {
    const versions = data.versions || [];
    const approved = versions.filter((v) => statusMeta(v.status).cls === "is-approved").length;
    const inReview = versions.filter((v) => statusMeta(v.status).cls === "is-review").length;
    const shots = (data.projects || []).reduce((a, p) => a + (Number(p.shots) || 0), 0) || versions.length;
    const outputs = (data.outputs || []).length;
    const cell = (label, val, cls = "") => `<div class="rpm-stat"><div class="rpm-stat-label">${label}</div><div class="rpm-stat-val ${cls}">${val}</div></div>`;
    return `<div class="rpm-stats">${cell("Active shots", shots)}${cell("In review", inReview, "is-review-text")}${cell("Approved", approved, "is-approved-text")}${cell("Outputs", outputs)}</div>`;
}

function Legend() {
    return `<div class="rpm-legend">
        <span class="rpm-pill is-wip">WIP</span>
        <span class="rpm-pill is-review">REVIEW</span>
        <span class="rpm-pill is-approved">APPROVED</span>
        <span class="rpm-pill is-retake">RETAKE</span>
    </div>`;
}

function ShotsTable(versions) {
    const list = versions || [];
    const rows = list.length ? list.map((v) => {
        const st = statusMeta(v.status);
        return `<div class="rpm-tr rpm-tr-click" data-action="Open Shot" data-shot="${escapeHtml(v.shot || "")}" data-searchable="${escapeHtml(((v.shot || "") + " " + (v.version || "")).toLowerCase())}">
            <span class="rpm-thumb-sm"></span>
            <span class="rpm-shot dsp">${escapeHtml(v.shot || "—")}</span>
            <span class="rpm-pill ${st.cls}">${st.label}</span>
            <span class="rpm-ver">${escapeHtml(v.version || "")}</span>
        </div>`;
    }).join("") : EmptyState("No versions tracked yet. Save a .rad workflow to start.");
    return `
        <section>
            <div class="rpm-section-head"><h2 class="rpm-section-title">Shots</h2>${Legend()}</div>
            <div class="rpm-table">
                <div class="rpm-tr rpm-thead"><span></span><span>Shot</span><span>Status</span><span>Ver</span></div>
                ${rows}
            </div>
        </section>
    `;
}

function OutputsTable(outputs) {
    const list = outputs || [];
    const rows = list.length ? list.map((o) => `
        <div class="rpm-tr rpm-tr-out" data-searchable="${escapeHtml(String(o.name || "").toLowerCase())}">
            <span class="rpm-shot" title="${escapeHtml(o.name || "")}">${escapeHtml(o.name || "")}</span>
            <span class="rpm-tag">${escapeHtml(o.type || "")}</span>
            <span class="rpm-dim">${escapeHtml(o.size || "")}</span>
            <span class="rpm-dim">${escapeHtml(o.date || "")}</span>
        </div>
    `).join("") : EmptyState("No render outputs found in the ComfyUI output folder.");
    return `
        <section>
            <div class="rpm-section-head"><h2 class="rpm-section-title">Recent outputs</h2></div>
            <div class="rpm-table rpm-table-out">
                <div class="rpm-tr rpm-tr-out rpm-thead"><span>File</span><span>Type</span><span>Size</span><span>Updated</span></div>
                ${rows}
            </div>
        </section>
    `;
}

function PageHead(data, title) {
    const projects = data.projects || [];
    const proj = projects.find((p) => p.id === data.activeProjectId) || projects[0];
    return `<div class="rpm-page-head">
        <div class="rpm-crumb">${escapeHtml(proj?.name || "No project")}</div>
        <div class="rpm-page-title dsp">${escapeHtml(title || "Dashboard")}</div>
    </div>`;
}

function ProjectsView(projects) {
    const list = projects || [];
    if (!list.length) return EmptyState("No projects yet — save a .rad workflow to create one.");
    const rows = list.map((p) => `<div class="rpm-tr rpm-tr-proj rpm-tr-click" data-action="Select Project" data-project-id="${escapeHtml(p.id)}">
        <span class="rpm-shot">${escapeHtml(p.name)}</span><span class="rpm-dim">${escapeHtml(p.shots)} shots</span><span class="rpm-dim">${escapeHtml(p.workflows || 0)} wf</span><span class="rpm-dim">${escapeHtml(p.updated || "")}</span>
    </div>`).join("");
    return `<div class="rpm-table">
        <div class="rpm-tr rpm-tr-proj rpm-thead"><span>Project</span><span>Shots</span><span>Workflows</span><span>Updated</span></div>
        ${rows}
    </div>`;
}

function ViewBody(data) {
    const versions = data.versions || [];
    if (activeView === "Shots") return `${PageHead(data, "Shots")}${Stats(data)}${ShotsTable(versions)}`;
    if (activeView === "Reviews") {
        const review = versions.filter((v) => { const c = statusMeta(v.status).cls; return c === "is-review" || c === "is-retake"; });
        return `${PageHead(data, "Reviews")}${ShotsTable(review)}`;
    }
    if (activeView === "Renders") return `${PageHead(data, "Outputs")}${OutputsTable(data.outputs)}`;
    if (activeView === "Projects") return `${PageHead(data, "Projects")}${ProjectsView(data.projects)}`;
    if (activeView !== "Dashboard") return `${PageHead(data, activeView)}<div class="rpm-empty">${escapeHtml(activeView)} — no data for this view yet.</div>`;
    return `${PageHead(data, "Dashboard")}
            <div class="rpm-top-grid">${Hero(data.continueWorking || {})}${StorageCard(data.storage || {})}</div>
            ${Stats(data)}
            ${ShotsTable(versions)}
            ${OutputsTable(data.outputs)}`;
}

function ProjectManagerDashboard(data) {
    return `
        ${Sidebar(data)}
        <main class="rpm-main">
            ${TopBar(data)}
            <div class="rpm-content">
                ${ViewBody(data)}
            </div>
        </main>
        <div class="rpm-toast" data-toast></div>
        <div id="rpm-drawer-root"></div>
    `;
}

function ShotDetail(shot, data) {
    const vers = (data.versions || []).filter((v) => String(v.shot || "") === String(shot));
    const latest = vers[0] || {};
    const st = statusMeta(latest.status);
    const outs = (data.outputs || []).filter((o) => String(o.name || "").toUpperCase().includes(String(shot).toUpperCase()));
    const allNotes = data.notes || [];
    const shotNotes = allNotes.filter((n) => String(n).toUpperCase().includes(String(shot).toUpperCase()));
    const notes = shotNotes.length ? shotNotes : allNotes;

    const verRows = vers.length ? vers.map((v, i) => {
        const s = statusMeta(v.status);
        return `<div class="rpm-drow ${i === 0 ? "is-latest" : ""}">
            <span class="rpm-ver">${escapeHtml(v.version || "")}</span>
            <span class="rpm-pill ${s.cls}">${s.label}</span>
            ${i === 0 ? `<span class="rpm-drow-tag">latest</span>` : ""}
        </div>`;
    }).join("") : EmptyState("No versions for this shot yet.");

    const outRows = outs.length ? outs.map((o) => `<div class="rpm-drow">
        <span class="rpm-shot" title="${escapeHtml(o.name || "")}">${escapeHtml(o.name || "")}</span>
        <span class="rpm-tag">${escapeHtml(o.type || "")}</span>
        <span class="rpm-dim">${escapeHtml(o.size || "")}</span>
    </div>`).join("") : EmptyState("No outputs for this shot.");

    const noteRows = notes.length ? notes.map((n) => `<div class="rpm-note"><span class="rpm-note-dot"></span>${escapeHtml(n)}</div>`).join("") : EmptyState("No review notes.");

    return `
        <div class="rpm-drawer-backdrop" data-action="Close Shot"></div>
        <aside class="rpm-drawer" role="dialog" aria-label="Shot ${escapeHtml(shot)}">
            <header class="rpm-drawer-head">
                <div>
                    <div class="rpm-kicker">Shot</div>
                    <div class="rpm-drawer-title dsp">${escapeHtml(shot)}</div>
                </div>
                <span class="rpm-pill ${st.cls}">${st.label}</span>
                <button class="rpm-icon-btn" type="button" data-action="Close Shot" title="Close">&#10005;</button>
            </header>
            <div class="rpm-drawer-body">
                <div class="rpm-drawer-preview">
                    <span class="rpm-hero-range">1001–1142</span>
                    <span class="rpm-hero-play">${svg("play")}</span>
                </div>
                <div class="rpm-drawer-actions">
                    <button class="rpm-button is-primary" type="button" data-action="Open Shot Workflow" data-filename="${escapeHtml(latest.filename || "")}">${svg("resume")}Open in canvas</button>
                    <button class="rpm-button" type="button" data-action="Approve Shot" data-shot="${escapeHtml(shot)}">Approve</button>
                    <button class="rpm-button" type="button" data-action="Request Changes" data-shot="${escapeHtml(shot)}">Request changes</button>
                </div>
                <div class="rpm-drawer-section"><div class="rpm-drawer-label">Version history</div>${verRows}</div>
                <div class="rpm-drawer-section"><div class="rpm-drawer-label">Outputs</div>${outRows}</div>
                <div class="rpm-drawer-section"><div class="rpm-drawer-label">Review notes</div>${noteRows}</div>
            </div>
        </aside>
    `;
}

function openShotDrawer(shot, data) {
    const root = document.getElementById("rpm-drawer-root");
    if (!root) return;
    root.innerHTML = ShotDetail(shot, data);
    requestAnimationFrame(() => root.classList.add("is-open"));
}

function closeShotDrawer() {
    const root = document.getElementById("rpm-drawer-root");
    if (!root) return;
    root.classList.remove("is-open");
    window.setTimeout(() => { if (root) root.innerHTML = ""; }, 220);
}

function showToast(message) {
    const toast = document.querySelector("[data-toast]");
    if (!toast) return;
    toast.textContent = message;
    toast.classList.add("is-visible");
    window.clearTimeout(showToast.timer);
    showToast.timer = window.setTimeout(() => {
        toast.classList.remove("is-visible");
    }, 1800);
}

async function fetchJson(url, options = {}) {
    const response = await fetch(url, options);
    if (!response.ok) {
        const data = await response.json().catch(() => ({}));
        throw new Error(data.error || response.statusText);
    }
    return response.json();
}

async function loadDashboardData() {
    try {
        const data = await fetchJson(futureApi.dashboard);
        return {
            ...mockProjectManagerData,
            ...data,
            projects: data.projects || [],
            versions: data.versions || [],
            outputs: data.outputs || [],
            notes: data.notes || [],
            quickActions: data.quickActions || mockProjectManagerData.quickActions,
            nav: data.nav || mockProjectManagerData.nav,
            source: data.source || "Live API",
        };
    } catch (error) {
        console.warn("[Radiance Project Manager] Falling back to mock data:", error);
        return mockProjectManagerData;
    }
}

async function loadProjectDetails(state, projectId) {
    const [versions, outputs, notes] = await Promise.all([
        fetchJson(futureApi.versions(projectId)),
        fetchJson(futureApi.outputs(projectId)),
        fetchJson(futureApi.notes(projectId)),
    ]);

    const project = state.data.projects.find((item) => item.id === projectId);
    const latestVersion = versions.versions?.[0];

    return {
        ...state.data,
        activeProjectId: projectId,
        versions: versions.versions || [],
        outputs: outputs.outputs || [],
        notes: notes.notes || [],
        continueWorking: latestVersion ? {
            workflow: latestVersion.workflow || latestVersion.filename || "No workflow saved yet",
            project: project?.name || state.data.continueWorking.project,
            shot: latestVersion.shot || "GENERAL",
            status: latestVersion.status || "WIP",
            lastModified: latestVersion.updated || "",
            filename: latestVersion.filename || "",
        } : {
            workflow: "No workflow saved yet",
            project: project?.name || state.data.continueWorking.project,
            shot: "GENERAL",
            status: "No Versions",
            lastModified: "",
            filename: "",
        },
    };
}

async function openWorkflow(filename) {
    if (!filename) {
        showToast("No workflow is available to open yet.");
        return;
    }

    const data = await fetchJson(`/radiance/workflows/get?filename=${encodeURIComponent(filename)}`);
    const target = window.opener || (window.parent !== window ? window.parent : null);
    if (target) {
        target.postMessage({ type: "radiance_load_workflow", content: data.content }, "*");
        showToast("Workflow sent to the active ComfyUI canvas.");
    } else {
        showToast("Open this dashboard from ComfyUI to load workflows into the canvas.");
    }
}

async function saveVersion(projectId) {
    const target = window.opener || (window.parent !== window ? window.parent : null);
    if (target) {
        target.postMessage({ type: "radiance_save_project_version", projectId }, "*");
        showToast("Save requested from the active ComfyUI canvas.");
        return;
    }

    const result = await fetchJson(futureApi.saveVersion(projectId), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: "Dashboard save version", author: "Ahmed" }),
    });
    showToast(result.message || "Version saved.");
}

async function exportPackage(projectId) {
    const response = await fetch(futureApi.exportPackage(projectId), { method: "POST" });
    if (!response.ok) {
        const data = await response.json().catch(() => ({}));
        throw new Error(data.error || response.statusText);
    }

    const blob = await response.blob();
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = `${projectId || "radiance"}_package.zip`;
    document.body.appendChild(anchor);
    anchor.click();
    setTimeout(() => {
        document.body.removeChild(anchor);
        URL.revokeObjectURL(url);
    }, 100);
}

function bindInteractions(root, state) {
    root.querySelector("[data-search]")?.addEventListener("input", (event) => {
        const query = event.target.value.trim().toLowerCase();
        root.querySelectorAll("[data-searchable]").forEach((row) => {
            row.hidden = query.length > 0 && !row.dataset.searchable.includes(query);
        });
    });

    root.addEventListener("click", (event) => {
        const button = event.target.closest("[data-action]");
        if (!button) return;

        const action = button.dataset.action;
        const activeProjectId = state.data.activeProjectId || state.data.projects[0]?.id || "";

        if (action === "Select Project") {
            activeView = "Dashboard";
            const projectId = button.dataset.projectId;
            loadProjectDetails(state, projectId)
                .then((nextData) => render(nextData))
                .catch((error) => showToast(error.message));
            return;
        }

        if (action === "Open Workflow") {
            openWorkflow(state.data.continueWorking.filename).catch((error) => showToast(error.message));
            return;
        }

        if (action === "Nav") {
            activeView = button.dataset.nav || "Dashboard";
            render(state.data);
            return;
        }

        if (action === "Toggle Projects") {
            const menu = root.querySelector("[data-proj-menu]");
            if (menu) menu.hidden = !menu.hidden;
            return;
        }

        if (action === "Open Shot") {
            openShotDrawer(button.dataset.shot, state.data);
            return;
        }

        if (action === "Close Shot") {
            closeShotDrawer();
            return;
        }

        if (action === "Open Shot Workflow") {
            const filename = button.dataset.filename;
            if (filename) {
                openWorkflow(filename).catch((error) => showToast(error.message));
            } else {
                showToast("No saved workflow for this version yet.");
            }
            return;
        }

        if (action === "Approve Shot" || action === "Request Changes") {
            const shot = button.dataset.shot;
            const status = action === "Approve Shot" ? "Approved" : "Retake";
            if (!activeProjectId) { showToast("Save a project first."); return; }
            fetchJson(`/radiance/projects/${encodeURIComponent(activeProjectId)}/shots/${encodeURIComponent(shot)}/status`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ status }),
            })
                .then(() => loadProjectDetails(state, activeProjectId))
                .then((nextData) => { render(nextData); showToast(`${shot} → ${status}`); })
                .catch((error) => showToast(error.message));
            closeShotDrawer();
            return;
        }

        if (action === "Save Version") {
            if (!activeProjectId) {
                showToast("Save a workflow first to create a project.");
                return;
            }
            saveVersion(activeProjectId)
                .then(() => loadDashboardData())
                .then((nextData) => render(nextData))
                .catch((error) => showToast(error.message));
            return;
        }

        if (action === "Export Package") {
            if (!activeProjectId) {
                showToast("Save a workflow first to create a project.");
                return;
            }
            exportPackage(activeProjectId)
                .then(() => showToast("Package export started."))
                .catch((error) => showToast(error.message));
            return;
        }

        showToast(`${action} is not wired to a backend action yet.`);
    });
}

const mount = document.getElementById("radiance-project-manager");

function render(data) {
    if (!mount) return;
    const state = { data };
    mount.innerHTML = ProjectManagerDashboard(data);
    bindInteractions(mount, state);
}

if (mount) {
    mount.innerHTML = ProjectManagerDashboard({
        ...mockProjectManagerData,
        projects: [],
        versions: [],
        outputs: [],
        notes: [],
        source: "Loading",
    });
    loadDashboardData().then(render);
}

window.addEventListener("message", (event) => {
    if (event.origin !== window.location.origin) return;
    if (event.data?.type !== "radiance_project_action_result") return;
    if (event.data.success) {
        showToast("Current canvas saved to the project.");
        loadDashboardData().then(render);
    } else {
        showToast(event.data.error || "Project action failed.");
    }
});

window.addEventListener("keydown", (e) => {
    if (e.key === "Escape") {
        const drawer = document.getElementById("rpm-drawer-root");
        if (drawer && drawer.classList.contains("is-open")) {
            closeShotDrawer();
            return;
        }
        if (window.parent && window.parent !== window && window.parent.closeRadianceDashboard) {
            window.parent.closeRadianceDashboard();
        }
    }
});
