// ◎ Radiance — the data side of the Radiance page (studio.html).
//
// Pure functions only: they turn what the Radiance routes return into what
// the page shows, so they run under `node --test` without a browser. The DOM
// side is studio.mjs.

export const SCREENS = ["home", "workflows", "gizmos", "assets", "shots", "renders", "models"];

export function screenFromHash(hash) {
    const id = String(hash || "").replace(/^#/, "").split(/[?&]/)[0].toLowerCase();
    // The old dashboards' names still land somewhere sensible.
    const alias = { library: "workflows", manager: "shots", projects: "shots", dashboard: "home", setup: "models" };
    const screen = alias[id] || id;
    return SCREENS.includes(screen) ? screen : "home";
}

export function greeting(date = new Date()) {
    const h = date.getHours();
    if (h < 5) return "Good evening";
    if (h < 12) return "Good morning";
    if (h < 18) return "Good afternoon";
    return "Good evening";
}

export function initials(name) {
    const parts = String(name || "").trim().split(/\s+/).filter(Boolean);
    if (!parts.length) return "R";
    return (parts[0][0] + (parts.length > 1 ? parts[parts.length - 1][0] : "")).toUpperCase();
}

export function plural(n, one, many = one + "s") {
    return `${Number(n).toLocaleString("en-US")} ${Number(n) === 1 ? one : many}`;
}

export function formatBytes(n) {
    const v = Number(n) || 0;
    if (v < 1024) return `${v} B`;
    const units = ["KB", "MB", "GB", "TB"];
    let x = v / 1024, i = 0;
    while (x >= 1024 && i < units.length - 1) { x /= 1024; i += 1; }
    return `${x < 10 ? x.toFixed(1) : Math.round(x)} ${units[i]}`;
}

export function relativeTime(epochSeconds, now = Date.now()) {
    const t = Number(epochSeconds) * 1000;
    if (!Number.isFinite(t) || t <= 0) return "";
    const s = Math.max(0, (now - t) / 1000);
    if (s < 60) return "just now";
    if (s < 3600) return plural(Math.floor(s / 60), "minute") + " ago";
    if (s < 86400) return plural(Math.floor(s / 3600), "hour") + " ago";
    if (s < 2 * 86400) return "yesterday";
    if (s < 7 * 86400) return plural(Math.floor(s / 86400), "day") + " ago";
    return new Date(t).toLocaleDateString("en-GB", { day: "numeric", month: "short", year: "numeric" });
}

// ── Workflows ────────────────────────────────────────────────────────────────

export const WORKFLOW_FILTERS = [
    ["all", "All"], ["hdr", "Make HDR"], ["generate", "Generate"], ["video", "Video"],
    ["deliver", "Deliver"], ["api", "API"], ["mine", "Saved by you"],
];

const baseName = (filename) => String(filename || "").split(/[\\/]/).pop().replace(/\.(rad|json)$/i, "");

export function isOfficial(filename) {
    return /^official[\\/]/i.test(String(filename || ""));
}

/** The filter groups a workflow belongs to, from its card kicker and its path. */
export function workflowGroups(wf) {
    const meta = wf?.metadata || {};
    const kicker = String(meta.stats?.kicker || meta.category || "").toLowerCase();
    const path = String(wf?.filename || "").toLowerCase();
    const groups = new Set();
    if (!isOfficial(wf?.filename)) groups.add("mine");
    if (/(^|[\\/])api[\\/]/.test(path) || /^api\b/.test(kicker)) groups.add("api");
    if (/video|ltx|wan|plates|remaster/.test(kicker + " " + path)) groups.add("video");
    if (/generate|t2i|flux|ltx|wan22/.test(kicker + " " + path) || groups.has("api")) groups.add("generate");
    if (/deliver/.test(kicker + " " + path)) groups.add("deliver");
    if (/hdr|remaster|sdr/.test(kicker) || /sdr_to_hdr|remaster/.test(path)) groups.add("hdr");
    return [...groups];
}

/** One card per workflow, official ones first in the order a newcomer needs them. */
export function workflowCards(list) {
    const order = ["start", "sdr_to_hdr", "sdr_to_hdr_analytic", "sdr_remaster_ai_fill", "hdr_delivery",
        "colour_grade", "shot_look_match", "t2i_i2i_hdr"];
    const rank = (wf) => {
        const i = order.indexOf(baseName(wf.filename));
        if (!isOfficial(wf.filename)) return 1000;
        if (i >= 0) return i;
        return /[\\/]api[\\/]/i.test(wf.filename) ? 500 : 100;
    };
    return (Array.isArray(list) ? list : [])
        .filter((wf) => wf && wf.filename)
        .map((wf) => {
            const meta = wf.metadata || {};
            const stats = meta.stats || {};
            const official = isOfficial(wf.filename);
            const title = String(stats.title || meta.title || baseName(wf.filename).replace(/[_-]+/g, " "));
            return {
                id: wf.filename,
                filename: wf.filename,
                title,
                kicker: String(stats.kicker || (official ? "Official" : "Saved by you")),
                description: String(meta.description || ""),
                tags: Array.isArray(stats.tags) ? stats.tags.map(String) : [],
                nodes: Number(stats.node_count) || 0,
                official,
                author: String(meta.author || ""),
                hasPreview: meta.has_preview !== false && meta.format !== "v2",
                groups: workflowGroups(wf),
                mtime: Number(wf.mtime) || 0,
                size: Number(wf.size) || 0,
                rank: rank(wf),
            };
        })
        .sort((a, b) => a.rank - b.rank || (a.official ? a.title.localeCompare(b.title) : b.mtime - a.mtime));
}

export function filterWorkflows(cards, filter = "all", query = "") {
    const q = String(query || "").trim().toLowerCase();
    return cards.filter((c) => (filter === "all" || c.groups.includes(filter))
        && (!q || [c.title, c.kicker, c.description, c.filename, ...c.tags].join(" ").toLowerCase().includes(q)));
}

export function workflowCounts(cards) {
    const counts = Object.fromEntries(WORKFLOW_FILTERS.map(([id]) => [id, 0]));
    counts.all = cards.length;
    for (const c of cards) for (const g of c.groups) if (g in counts) counts[g] += 1;
    return counts;
}

// ── Assets ───────────────────────────────────────────────────────────────────

export const ASSET_FILTERS = [["all", "All"], ["image", "Images"], ["hdr", "HDR"], ["video", "Video"], ["sequence", "Sequences"]];

const HDR_FORMATS = new Set(["EXR", "HDR", "PFM", "TIF", "TIFF"]);

export function isHdrAsset(a) {
    return a?.type === "hdri" || HDR_FORMATS.has(String(a?.format || "").toUpperCase());
}

export function filterAssets(assets, filter = "all", query = "") {
    const q = String(query || "").trim().toLowerCase();
    return (Array.isArray(assets) ? assets : []).filter((a) => {
        const okType = filter === "all"
            || (filter === "hdr" ? isHdrAsset(a)
                : filter === "image" ? (a.type === "image" || a.type === "hdri")
                    : a.type === filter);
        return okType && (!q || `${a.name} ${a.format} ${a.path}`.toLowerCase().includes(q));
    });
}

export function assetKind(a) {
    if (!a) return "";
    if (a.type === "sequence") return `${a.format} sequence · ${plural(a.frames || 0, "frame")}`;
    if (a.type === "video") return `${a.format} video`;
    return `${isHdrAsset(a) ? "HDR" : "SDR"} image · ${a.format}`;
}

// ── Shots ────────────────────────────────────────────────────────────────────

// The server's statuses, in the words the page uses.
export const SHOT_STATUS = {
    WIP: { label: "In progress", tone: "warn" },
    Review: { label: "In review", tone: "info" },
    Approved: { label: "Approved", tone: "ok" },
    Final: { label: "Final", tone: "ok" },
    Retake: { label: "Retake", tone: "bad" },
};

export function shotStatus(raw) {
    const key = Object.keys(SHOT_STATUS).find((k) => k.toLowerCase() === String(raw || "").toLowerCase());
    return key ? { key, ...SHOT_STATUS[key] } : { key: "WIP", ...SHOT_STATUS.WIP };
}

const versionNumber = (v) => {
    const m = String(v || "").match(/(\d+)/);
    return m ? parseInt(m[1], 10) : 0;
};

/**
 * One row per shot from a project's version list: its newest version, its
 * status and every version it has, newest first.
 */
/** A shot code as a whole token: SH040 is in "SH040_comp" and "comp.SH040", not in "SH0400". */
export function hasToken(text, token) {
    if (!token) return false;
    const t = String(token).toUpperCase().replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
    return new RegExp(`(^|[^A-Z0-9])${t}([^A-Z0-9]|$)`).test(String(text || "").toUpperCase());
}

/** An output's name with its frame range when it is a sequence. */
export function outputLabel(o) {
    const name = String(o?.name || "");
    return o?.frames && Array.isArray(o.range) ? `${name} · ${o.frames} frames · ${o.range[0]}-${o.range[1]}` : name;
}

export function shotRows(versions, outputs = [], notes = []) {
    const byShot = new Map();
    for (const v of Array.isArray(versions) ? versions : []) {
        const shot = String(v?.shot || "").trim();
        if (!shot || shot === "GENERAL") continue;
        if (!byShot.has(shot)) byShot.set(shot, []);
        byShot.get(shot).push(v);
    }
    const rows = [];
    for (const [shot, list] of byShot) {
        // Backups carry their save message in `status`; a real status wins.
        const sorted = [...list].sort((a, b) => versionNumber(b.version) - versionNumber(a.version));
        const latest = sorted[0];
        const real = sorted.find((v) => Object.keys(SHOT_STATUS).some((k) => k.toLowerCase() === String(v.status || "").toLowerCase()));
        const up = shot.toUpperCase();
        rows.push({
            shot,
            version: String(latest.version || ""),
            status: shotStatus(real?.status),
            workflow: String(latest.workflow || ""),
            filename: String(latest.filename || ""),
            updated: String(latest.updated || ""),
            versions: sorted.map((v) => ({ version: String(v.version || ""), filename: String(v.filename || ""), updated: String(v.updated || ""), note: String(v.status || "") })),
            outputs: (Array.isArray(outputs) ? outputs : []).filter((o) => hasToken(o?.name, up) || hasToken(o?.path, up)),
            notes: (Array.isArray(notes) ? notes : []).map(String).filter((n) => hasToken(n, up)),
        });
    }
    return rows.sort((a, b) => a.shot.localeCompare(b.shot, "en", { numeric: true }));
}

export const SHOT_FILTERS = [["all", "All"], ["Review", "In review"], ["WIP", "In progress"], ["Retake", "Retakes"], ["Approved", "Approved"]];

export function filterShots(rows, filter = "all", query = "") {
    const q = String(query || "").trim().toLowerCase();
    return rows.filter((r) => (filter === "all" || r.status.key === filter || (filter === "Approved" && r.status.key === "Final"))
        && (!q || `${r.shot} ${r.workflow}`.toLowerCase().includes(q)));
}

export function shotStats(rows) {
    const n = (k) => rows.filter((r) => r.status.key === k).length;
    return { total: rows.length, wip: n("WIP"), review: n("Review"), approved: n("Approved") + n("Final"), retake: n("Retake") };
}

// ── Renders (ComfyUI /history) ───────────────────────────────────────────────

/** Recent renders from ComfyUI's /history, newest first. */
export function renderRows(history, limit = 40) {
    const rows = [];
    for (const [id, item] of Object.entries(history && typeof history === "object" ? history : {})) {
        const status = item?.status || {};
        const msgs = Array.isArray(status.messages) ? status.messages : [];
        const stamp = (type) => msgs.find((m) => Array.isArray(m) && m[0] === type)?.[1]?.timestamp;
        const err = msgs.find((m) => Array.isArray(m) && m[0] === "execution_error")?.[1];
        const images = [];
        for (const out of Object.values(item?.outputs || {})) {
            for (const im of out?.images || []) if (im?.filename) images.push(im);
        }
        const started = Number(stamp("execution_start")) || 0;
        const ended = Number(stamp("execution_success") || stamp("execution_error") || stamp("execution_interrupted")) || 0;
        rows.push({
            id,
            ok: status.status_str ? status.status_str === "success" : !err,
            failed: status.status_str === "error" || !!err,
            error: err ? `${err.node_type || "A node"}: ${String(err.exception_message || "failed").trim().split("\n")[0]}` : "",
            started,
            seconds: started && ended ? Math.max(0, (ended - started) / 1000) : 0,
            images,
            number: Array.isArray(item?.prompt) ? Number(item.prompt[0]) || 0 : 0,
        });
    }
    rows.sort((a, b) => (b.started - a.started) || (b.number - a.number));
    return rows.slice(0, limit);
}

export function viewUrl(image) {
    const p = new URLSearchParams({ filename: image.filename, type: image.type || "output", subfolder: image.subfolder || "" });
    return `/view?${p}`;
}

// ── Home ─────────────────────────────────────────────────────────────────────

/** The few things that need the artist, from real data only. */
export function needsYou({ shots = [], renders = [] } = {}) {
    const items = [];
    for (const r of shots.filter((s) => s.status.key === "Review").slice(0, 2)) {
        items.push({ tone: "info", title: `${r.shot} is waiting for review`, text: `${r.version} · ${r.workflow}`, action: "Review", go: { screen: "shots", shot: r.shot } });
    }
    for (const r of shots.filter((s) => s.status.key === "Retake").slice(0, 1)) {
        items.push({ tone: "bad", title: `${r.shot} needs a retake`, text: `${r.version} · ${r.workflow}`, action: "Open", go: { screen: "shots", shot: r.shot } });
    }
    const failed = renders.find((r) => r.failed);
    if (failed) items.push({ tone: "bad", title: "A render failed", text: failed.error || "See Renders for the details", action: "See", go: { screen: "renders" } });
    return items.slice(0, 4);
}

export function gpuSummary(stats) {
    const d = Array.isArray(stats?.devices) ? stats.devices[0] : null;
    if (!d) return null;
    const total = Number(d.vram_total) || 0;
    const free = Number(d.vram_free) || 0;
    const used = Math.max(0, total - free);
    const name = String(d.name || "GPU").replace(/^cuda:\d+\s*/, "").replace(/\s*:\s*cudaMallocAsync$/, "");
    return { name, total, used, percent: total ? Math.round((used / total) * 100) : 0, label: total ? `${formatBytes(used)} / ${formatBytes(total)}` : "" };
}

// ── Gizmos ───────────────────────────────────────────────────────────────────

/** Saved gizmos as cards, by name; one that this start-up has not loaded says so. */
export function gizmoCards(list) {
    return (Array.isArray(list) ? list : []).map((g) => {
        const title = String(g.display_name || g.name || "Gizmo");
        const status = g.error ? { key: "error", label: "Cannot be read", tone: "bad" }
            : g.loaded ? { key: "ready", label: "In the node menu", tone: "ok" }
                : { key: "restart", label: "Restart ComfyUI to use", tone: "warn" };
        return {
            id: String(g.name || title), title, status,
            category: String(g.category || "FXTD STUDIOS/Radiance/◎ Gizmos").split("/").pop(),
            description: String(g.description || ""),
            inputs: g.inputs || [], outputs: g.outputs || [], knobs: g.knobs || [],
            nodes: Number(g.node_count) || 0, nodeTypes: g.node_types || [],
            className: String(g.class_name || ""), file: String(g.file || ""),
            size: Number(g.size) || 0, mtime: Number(g.mtime) || 0, error: g.error || "",
        };
    }).sort((a, b) => a.title.localeCompare(b.title));
}

export function filterGizmos(cards, query = "") {
    const q = String(query || "").trim().toLowerCase();
    if (!q) return cards;
    return cards.filter((g) => [g.title, g.id, g.description, g.category, ...g.nodeTypes].join(" ").toLowerCase().includes(q));
}

// ── Models (the setup check) ─────────────────────────────────────────────────

/** Bytes in decimal GB/MB, as model downloads are quoted. */
export function formatSize(n) {
    const v = Number(n) || 0;
    if (v >= 1e9) return `${(v / 1e9).toFixed(v >= 1e10 ? 0 : 1)} GB`;
    if (v >= 1e6) return `${Math.round(v / 1e6)} MB`;
    if (v > 0) return `${Math.max(1, Math.round(v / 1e3))} KB`;
    return "";
}

/** Memory in binary GB, as GPU and RAM sizes are quoted (a 24 GB card reads 24 GB). */
export function formatMemory(n) {
    const v = Number(n) || 0;
    if (!v) return "";
    const gb = v / 1024 ** 3;
    return gb >= 1 ? `${gb.toFixed(gb >= 10 ? 0 : 1)} GB` : `${Math.round(v / 1024 ** 2)} MB`;
}

/** [label, value] rows describing this machine, from /radiance/setup/scan's hardware. */
export function hardwareRows(hw = {}) {
    const rows = [];
    const gpus = hw.gpus || [];
    if (!gpus.length) rows.push(["GPU", "None found: models run on the CPU (very slow)"]);
    gpus.forEach((g, i) => rows.push([gpus.length > 1 ? `GPU ${i}` : "GPU",
        g.vram ? `${g.name} · ${formatMemory(g.vram)}` : g.name]));
    if (hw.ram) rows.push(["Memory", `${formatMemory(hw.ram)}${hw.ram_free ? ` · ${formatMemory(hw.ram_free)} free` : ""}`]);
    if (hw.disk_free != null) rows.push(["Free disk (models)", formatSize(hw.disk_free) || "0"]);
    rows.push(["PyTorch", [hw.torch, hw.cuda ? `CUDA ${hw.cuda}` : ""].filter(Boolean).join(" · ") || "—"]);
    if (hw.comfyui) rows.push(["ComfyUI", hw.comfyui]);
    if (hw.radiance) rows.push(["Radiance", hw.radiance]);
    return rows;
}

/** What one GPU can comfortably run, in words, from its memory. */
export function gpuTier(hw = {}) {
    const vram = Math.max(0, ...(hw.gpus || []).map((g) => Number(g.vram) || 0));
    if (!vram) return { tone: "bad", text: "No CUDA GPU: SDR to HDR and grading work; generation is very slow." };
    const gb = vram / 1024 ** 3;
    if (gb >= 22) return { tone: "ok", text: "Runs every local workflow, including the 22B video models and the AI fill." };
    if (gb >= 15) return { tone: "ok", text: "Runs the image workflows and the AI fill; the large video models offload to RAM." };
    if (gb >= 7.5) return { tone: "warn", text: "Runs SDR to HDR, the Starter and FLUX.2 Klein 4B; larger models offload to RAM (slow)." };
    return { tone: "warn", text: "Fine for SDR to HDR and grading; generation models will be slow." };
}

/** One row per workflow: ready, or what is missing and how big it is. */
export function setupRows(scan = {}) {
    const models = scan.models || {};
    return (scan.workflows || []).map((w) => {
        const missing = (w.missing || []).map((id) => models[id] || { id, title: id });
        // A partial total ("3 MB+" for four checkpoints) misleads, so it is shown only when all sizes are known or it is already large.
        const bytes = Number(w.missing_bytes) || 0;
        const size = !bytes ? "" : w.missing_bytes_known ? formatSize(bytes) : bytes >= 1e9 ? `${formatSize(bytes)}+` : "";
        let status;
        if (w.ready) status = { key: "ready", label: "Ready", tone: "ok" };
        else if ((w.gated || []).length) status = { key: "gated", label: "Needs a licence", tone: "warn" };
        else status = { key: "missing", label: `${missing.length} to download`, tone: "info" };
        return { id: w.id, title: w.title, status, missing, size, note: w.gpu_note || "",
                 required: w.required || [], optional: w.optional || [], needsNothing: !(w.required || []).length };
    }).sort((a, b) => (a.status.key === "ready") - (b.status.key === "ready") || a.title.localeCompare(b.title));
}

/** The downloads list's headline: how many run, and the overall progress. */
export function downloadsSummary(jobs = []) {
    const active = jobs.filter((j) => j.status === "running" || j.status === "queued");
    const done = jobs.filter((j) => j.status === "done").length;
    const failed = jobs.filter((j) => j.status === "failed").length;
    const total = active.reduce((s, j) => s + (Number(j.total) || 0), 0);
    const got = active.reduce((s, j) => s + Math.min(Number(j.done) || 0, Number(j.total) || Infinity), 0);
    return { active: active.length, done, failed, percent: total ? Math.min(100, Math.round((got / total) * 100)) : null };
}

/** Progress of one download, as a percentage and a line of text. */
export function jobProgress(j = {}) {
    const total = Number(j.total) || 0;
    const done = Number(j.done) || 0;
    const pct = total ? Math.min(100, Math.round((done / total) * 100)) : null;
    const text = j.status === "running"
        ? (total ? `${formatSize(done) || "0 MB"} of ${formatSize(total)}` : (done ? formatSize(done) : "Starting…"))
        : { queued: "Waiting", done: "Installed", failed: "Failed", cancelled: "Cancelled" }[j.status] || "";
    return { pct, text };
}
