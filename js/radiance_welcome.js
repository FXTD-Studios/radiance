/**
 * The Radiance welcome: a six-page tour on first install, a What's new page
 * after an update, and Help > Radiance Tour to open it again. When it shows
 * is decided in radiance_welcome_logic.js.
 */
import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import { RADIANCE_VERSION } from "./radiance_version.js";
import { LINKS, TOUR_PAGES, notesFor, startupView, stepPage } from "./radiance_welcome_logic.js";

const BASE = import.meta.url.replace(/\/[^/]+$/, "");
const SHOW_ID = "Radiance.Welcome.ShowAtStartup";
const LAST_ID = "Radiance.Welcome.LastVersion";
const LS_LAST = "radiance.welcome.lastVersion";
const LS_SHOW = "radiance.welcome.showAtStartup";

// ── Settings: ComfyUI's (per user, on the server) with localStorage as fallback ──
function getSetting(id, fallback) {
    try {
        const s = app.extensionManager?.setting;
        if (s?.get) { const v = s.get(id); if (v !== undefined && v !== null) return v; }
        const v = app.ui?.settings?.getSettingValue?.(id);
        if (v !== undefined && v !== null) return v;
    } catch { /* settings not ready */ }
    try {
        const raw = localStorage.getItem(id === SHOW_ID ? LS_SHOW : LS_LAST);
        if (raw !== null) return id === SHOW_ID ? raw !== "false" : raw;
    } catch { /* storage blocked */ }
    return fallback;
}

async function setSetting(id, value) {
    try { localStorage.setItem(id === SHOW_ID ? LS_SHOW : LS_LAST, String(value)); } catch { /* storage blocked */ }
    try {
        const s = app.extensionManager?.setting;
        if (s?.set) { await s.set(id, value); return; }
        await app.ui?.settings?.setSettingValueAsync?.(id, value);
    } catch { /* the localStorage copy stands */ }
}

function toast(summary, detail) {
    try {
        if (app.extensionManager?.toast?.add) {
            app.extensionManager.toast.add({ severity: "info", summary, detail, life: 8000 });
            return;
        }
    } catch { /* fall through */ }
    console.info(`[Radiance] ${summary}: ${detail}`);
}

// ── Styles ──────────────────────────────────────────────────────────────────
const CSS = `
.rw-back{position:fixed;inset:0;z-index:10050;background:rgba(0,0,0,.62);display:flex;align-items:center;justify-content:center;padding:16px;
  font-family:Inter,system-ui,-apple-system,"Segoe UI",sans-serif;color:#e8e8ea}
.rw-card{width:920px;max-width:100%;height:600px;max-height:calc(100vh - 32px);display:flex;border-radius:16px;overflow:hidden;background:#1c1c1e;
  border:1px solid #2f2f32;box-shadow:0 40px 120px rgba(0,0,0,.7)}
.rw-card.is-update{width:580px;height:auto;min-height:480px;flex-direction:column;padding:28px 32px 24px;box-sizing:border-box}
.rw-art{flex:0 0 380px;background:#141415;border-right:1px solid #2a2a2d;display:flex;flex-direction:column;align-items:center;justify-content:center;gap:16px;padding:28px;box-sizing:border-box}
.rw-main{flex:1;min-width:0;display:flex;flex-direction:column;padding:28px 32px 24px;box-sizing:border-box;overflow:auto}
.rw-head{display:flex;align-items:center;gap:10px}
.rw-brand{font-family:"JetBrains Mono",ui-monospace,Menlo,Consolas,monospace;font-size:12px;letter-spacing:.18em;font-weight:500}
.rw-count{margin-left:auto;font-family:"JetBrains Mono",ui-monospace,monospace;font-size:12px;color:#aeaeb3}
.rw-close{width:32px;height:32px;border:0;border-radius:8px;background:transparent;color:#bcbcc1;cursor:pointer;display:flex;align-items:center;justify-content:center}
.rw-close:hover{background:#2c2c2e;color:#fff}
.rw-kicker{margin-top:26px;font-family:"JetBrains Mono",ui-monospace,monospace;font-size:12px;letter-spacing:.16em;color:#409cff}
.rw-title{margin:8px 0 0;font-size:30px;line-height:1.2;font-weight:600;color:#e8e8ea}
.rw-lead{margin:10px 0 0;font-size:15px;line-height:1.55;color:#bcbcc1}
.rw-list{margin:22px 0 0;padding:0;list-style:none;display:flex;flex-direction:column;gap:14px}
.rw-item{display:grid;grid-template-columns:30px minmax(0,1fr);gap:12px;align-items:start}
.rw-mark{width:30px;height:30px;border-radius:8px;background:#2c2c2e;display:flex;align-items:center;justify-content:center;
  font-family:"JetBrains Mono",ui-monospace,monospace;font-size:12px;color:#409cff}
.rw-step .rw-mark{border-radius:50%;font-size:13px}
.rw-item>div>b{display:block;font-size:14px;font-weight:600}
.rw-item>div>span{display:block;margin-top:2px;font-size:13px;line-height:1.5;color:#aeaeb3}
.rw-step>div>span{margin-top:0;padding-top:5px;font-size:14px;color:#e8e8ea}
.rw-card:focus{outline:none}
.rw-actions{margin-top:20px;display:flex;gap:10px;flex-wrap:wrap}
.rw-btn{height:38px;padding:0 18px;border-radius:9px;border:1px solid #3a3a3c;background:transparent;color:#e8e8ea;font:inherit;font-size:14px;cursor:pointer}
.rw-btn:hover{background:#2c2c2e}
.rw-btn.is-soft{height:36px;padding:0 14px;border:0;background:#2c2c2e;font-size:13px;font-weight:500}
.rw-btn.is-soft:hover{background:#363639}
.rw-btn.is-primary{border:0;background:#0a84ff;color:#ffffff;font-weight:600;padding:0 20px}
.rw-btn.is-primary:hover{background:#409cff}
.rw-btn:focus-visible,.rw-close:focus-visible,.rw-link:focus-visible{outline:2px solid #409cff;outline-offset:2px}
.rw-foot{margin-top:auto;padding-top:18px;border-top:1px solid #2a2a2d;display:flex;align-items:center;gap:12px}
.rw-foot label{font-size:13px;color:#aeaeb3;display:flex;align-items:center;gap:10px;cursor:pointer}
.rw-foot input{width:16px;height:16px;margin:0;accent-color:#0a84ff}
.rw-dots{margin-left:auto;margin-right:8px;display:flex;gap:6px}
.rw-dots i{width:6px;height:6px;border-radius:3px;background:#48484a;transition:width .15s}
.rw-dots i.on{width:18px;background:#0a84ff}
.rw-img{display:block;border-radius:10px;border:1px solid #2f2f32;object-fit:cover}
.rw-chip{padding:4px 10px;border-radius:6px;background:#2c2c2e;font-family:"JetBrains Mono",ui-monospace,monospace;font-size:11px;letter-spacing:.08em;color:#bcbcc1}
.rw-seg{display:flex;padding:3px;border-radius:9px;background:#2c2c2e}
.rw-seg span{padding:6px 14px;border-radius:6px;font-size:13px;color:#bcbcc1}
.rw-seg span.on{background:#48484a;color:#e8e8ea;font-weight:500}
.rw-keys{margin-top:18px;display:flex;align-items:center;gap:8px;flex-wrap:wrap;font-size:13px;color:#aeaeb3}
.rw-keys kbd{min-width:24px;height:24px;padding:0 7px;box-sizing:border-box;border-radius:6px;border:1px solid #48484a;background:#2c2c2e;
  font-family:"JetBrains Mono",ui-monospace,monospace;font-size:12px;color:#e8e8ea;display:inline-flex;align-items:center;justify-content:center}
.rw-keys kbd:not(:first-child){margin-left:10px}
.rw-links{margin-top:20px;display:flex;flex-direction:column;gap:8px}
.rw-link{display:grid;grid-template-columns:minmax(0,1fr) 16px;align-items:center;gap:12px;padding:12px 14px;border-radius:10px;background:#232325;
  border:1px solid #2f2f32;color:#e8e8ea;text-decoration:none}
.rw-link:hover{background:#2a2a2d;color:#fff}
.rw-link b{display:block;font-size:14px;font-weight:600}
.rw-link span{display:block;margin-top:2px;font-size:13px;color:#aeaeb3}
.rw-note{margin:14px 0 0;font-size:13px;color:#8e8e93}
.rw-ver{margin-top:22px;display:flex;align-items:center;gap:10px;font-family:"JetBrains Mono",ui-monospace,monospace;font-size:12px;letter-spacing:.12em;color:#aeaeb3}
.rw-ver b{padding:3px 8px;border-radius:6px;background:rgba(10,132,255,.16);color:#409cff;font-weight:500}
.rw-bul{margin:20px 0 0;padding:0;list-style:none;display:flex;flex-direction:column;gap:12px}
.rw-bul li{display:grid;grid-template-columns:8px minmax(0,1fr);gap:12px;font-size:14px;line-height:1.5}
.rw-bul li::before{content:"";width:8px;height:8px;margin-top:7px;border-radius:50%;background:#0a84ff}
.rw-a{color:#409cff}.rw-a:hover{color:#7ab8ff}
.rw-back-frame{width:316px;height:230px;box-sizing:border-box;border-radius:8px;background:rgba(179,134,0,.2);border:1px solid rgba(240,168,48,.45);display:flex;flex-direction:column}
.rw-back-frame>div:first-child{height:34px;padding:0 12px;display:flex;align-items:center;justify-content:space-between;background:rgba(179,134,0,.35);border-radius:7px 7px 0 0;font-size:13px;font-weight:600}
.rw-node{position:absolute;border-radius:6px;background:#26262a;border:1px solid #3a3a3c}
.rw-node>div{height:18px;border-radius:6px 6px 0 0;background:#303034;font-size:9px;padding:3px 6px;box-sizing:border-box;color:#bcbcc1}
.rw-dot{display:inline-flex;align-items:center;gap:6px;padding:4px 9px;border-radius:6px;background:#2c2c2e;font-size:12px;color:#bcbcc1}
.rw-dot i{width:8px;height:8px;border-radius:2px}
@media (max-width: 760px){.rw-art{display:none}.rw-card{height:auto}}
`;

function ensureCSS() {
    if (document.getElementById("radiance-welcome-css")) return;
    const s = document.createElement("style");
    s.id = "radiance-welcome-css";
    s.textContent = CSS;
    document.head.appendChild(s);
}

// ── Small DOM helpers ───────────────────────────────────────────────────────
function el(tag, cls, text) {
    const e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text != null) e.textContent = text;
    return e;
}
function svg(markup) {
    const t = document.createElement("template");
    t.innerHTML = markup.trim();
    return t.content.firstChild;
}
const RING = '<svg width="18" height="18" viewBox="0 0 100 100" aria-hidden="true"><rect width="100" height="100" rx="22.5" fill="#1C1C1E"/><path d="M50.6 32.3A21 21 0 1 0 67.7 49.4" fill="none" stroke="#F5F5F7" stroke-width="17"/><rect x="54" y="29" width="17" height="17" rx="4" fill="#0A84FF"/></svg>';
const CLOSE = '<svg width="14" height="14" viewBox="0 0 14 14" aria-hidden="true"><path d="M2 2l10 10M12 2L2 12" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"/></svg>';
const OUT = '<svg width="14" height="14" viewBox="0 0 14 14" aria-hidden="true"><path d="M5 2h7v7M12 2L3 11" fill="none" stroke="#aeaeb3" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/></svg>';

function img(file, alt, w, h, extra = "") {
    const i = el("img", "rw-img");
    i.src = `${BASE}/welcome/${file}`;
    i.alt = alt;
    i.width = w; i.height = h;
    i.style.cssText = `width:${w}px;height:${h}px;${extra}`;
    return i;
}
function items(list, step = false) {
    const ul = el(step ? "ol" : "ul", "rw-list");
    for (const it of list) {
        const li = el("li", `rw-item${step ? " rw-step" : ""}`);
        li.appendChild(el("span", "rw-mark", it.mark));
        const body = el("div");
        if (it.title) body.appendChild(el("b", null, it.title));
        body.appendChild(el("span", null, it.text));
        li.appendChild(body);
        ul.appendChild(li);
    }
    return ul;
}
function softButton(label, onClick) {
    const b = el("button", "rw-btn is-soft", label);
    b.type = "button";
    b.addEventListener("click", onClick);
    return b;
}

// ── Actions on the tour's pages ─────────────────────────────────────────────
async function loadStarter() {
    try {
        const r = await fetch(api.apiURL("/workflow_templates/radiance/start.json"));
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        await app.loadGraphData(await r.json(), true, true, "Radiance Starter");
        return true;
    } catch (e) {
        toast("Starter workflow", `It could not be opened (${e.message}). Find it under Templates > radiance.`);
        return false;
    }
}
function openLibrary() {
    const url = `${BASE}/workspace_dashboard.html`;
    if (typeof window.showRadianceDashboard === "function") window.showRadianceDashboard(url, "Radiance Workflow Library");
    else window.open(url, "_blank", "noopener");
}

// ── Pages ───────────────────────────────────────────────────────────────────
const PAGES = {
    welcome: {
        kicker: "WELCOME", title: `Radiance ${RADIANCE_VERSION.replace(/\.0$/, "")} is installed`,
        lead: "A minute's tour of what is new and where things are. Every page has one thing to try.",
        art() {
            const f = document.createDocumentFragment();
            f.appendChild(img("card_local.webp", "Radiance 4 card for the SDR to HDR workflow", 316, 158));
            const t = el("div"); t.style.cssText = "display:flex;flex-direction:column;align-items:center;gap:6px";
            const v = el("div", "rw-brand", `RADIANCE ${RADIANCE_VERSION}`); v.style.color = "#aeaeb3";
            const s = el("div", null, "HDR, colour and VFX for ComfyUI"); s.style.cssText = "font-size:13px;color:#8e8e93";
            t.append(v, s); f.appendChild(t);
            return f;
        },
        body() {
            return items([
                { mark: "HDR", title: "RUDRA ACES 2: SDR to HDR with nothing to set up", text: "Any 8-bit image or video becomes a scene-linear ACES 2.0 master. The model downloads on first run (3 MB)." },
                { mark: "VW", title: "A new Viewer: Review, Color and Inspect", text: "Picture first, a Project colour sheet, scopes in nits and stills to compare against." },
                { mark: "44", title: "44 workflows, 25 of them API", text: "Nano Banana, Seedance, Veo, Kling and more, each lifted to HDR with RUDRA." },
                { mark: "F3", title: "FLUX 3 Klein and Prompt tabs", text: "A new image model, and one Prompt node that holds several prompts." },
            ]);
        },
    },
    workflows: {
        kicker: "WORKFLOWS", title: "Start from a workflow",
        lead: "Local workflows run on your GPU. API workflows generate with a ComfyUI partner node (a Comfy account and credits), then do the HDR work on your machine.",
        art() {
            const f = document.createDocumentFragment();
            f.appendChild(img("card_local.webp", "Card for the local SDR to HDR workflow", 300, 150, "transform:rotate(-2deg)"));
            f.appendChild(img("card_api.webp", "Card for the Seedance API workflow", 300, 150, "transform:rotate(1.5deg);margin-top:-18px;box-shadow:0 16px 40px rgba(0,0,0,.55)"));
            const chips = el("div"); chips.style.cssText = "display:flex;gap:8px;margin-top:8px";
            chips.append(el("span", "rw-chip", "14 LOCAL"), el("span", "rw-chip", "25 API"));
            f.appendChild(chips);
            return f;
        },
        body(close) {
            const f = document.createDocumentFragment();
            f.appendChild(items([
                { mark: "T", title: "Templates, in ComfyUI’s sidebar", text: "Every Radiance workflow is listed under radiance. Click one to open it." },
                { mark: "L", title: "The Workflow Library", text: "Cards with covers and tags, your own saved versions and a history for each. Open it from the Project Manager node." },
            ]));
            const a = el("div", "rw-actions");
            a.append(
                softButton("Load the Starter", async () => { if (await loadStarter()) close(); }),
                softButton("Open the Workflow Library", () => { close(); openLibrary(); }),
            );
            f.appendChild(a);
            return f;
        },
    },
    viewer: {
        kicker: "THE VIEWER", title: "Three ways to look at a frame",
        lead: "Add a Viewer node after any image. Switch workspace at the top left; the picture and your grade stay as they are.",
        art() {
            const f = document.createDocumentFragment();
            f.appendChild(img("viewer_inspect.webp", "The Viewer in Inspect: waveform, parade, vectorscope and QC", 324, 216));
            const seg = el("div", "rw-seg"); seg.setAttribute("aria-hidden", "true");
            seg.append(el("span", null, "Review"), el("span", null, "Color"), el("span", "on", "Inspect"));
            f.appendChild(seg);
            return f;
        },
        body() {
            const f = document.createDocumentFragment();
            f.appendChild(items([
                { mark: "R", title: "Review", text: "The picture and one bar: play, compare A and B with a wipe, grab stills to compare against, notes and a report." },
                { mark: "C", title: "Color", text: "A grade that opens on a parade and the wheels. Input and display are set once, in Color > Project Colour…" },
                { mark: "I", title: "Inspect", text: "Waveform, parade and vectorscope in nits on an HDR view, a probe under the pointer, and QC over every frame." },
            ]));
            const k = el("div", "rw-keys");
            k.append(el("kbd", null, "H"), el("span", null, "every key"), el("kbd", null, "F"), el("span", null, "fit"),
                el("kbd", null, "Shift F"), el("span", null, "full screen"));
            f.appendChild(k);
            return f;
        },
    },
    backdrops: {
        kicker: "BACKDROPS", title: "Keep a graph readable",
        lead: "A backdrop is a titled frame behind a group of nodes. Radiance names and colours it from what is inside, in ComfyUI's own group colours.",
        art() {
            const f = document.createDocumentFragment();
            const frame = el("div", "rw-back-frame");
            frame.setAttribute("role", "img");
            frame.setAttribute("aria-label", "Three nodes inside an amber backdrop titled HDR Processing");
            const head = el("div");
            head.append(el("span", null, "◎ HDR Processing"), svg('<svg width="12" height="14" viewBox="0 0 12 14" aria-hidden="true"><rect x="1" y="6" width="10" height="7" rx="1.5" fill="none" stroke="#f0b94f" stroke-width="1.4"/><path d="M3.5 6V4a2.5 2.5 0 015 0v2" fill="none" stroke="#409cff" stroke-width="1.4"/></svg>'));
            const area = el("div"); area.style.cssText = "flex:1;position:relative";
            [["Read", 18, 26, 96], ["SDR → HDR", 116, 14, 140], ["Write", 214, 40, 96]].forEach(([t, x, y, h]) => {
                const n = el("div", "rw-node"); n.style.cssText = `left:${x}px;top:${y}px;width:84px;height:${h}px`;
                n.appendChild(el("div", null, t)); area.appendChild(n);
            });
            frame.append(head, area);
            f.appendChild(frame);
            const chips = el("div"); chips.style.cssText = "display:flex;gap:6px;flex-wrap:wrap;justify-content:center;max-width:320px";
            [["Generation", "#3f789e"], ["HDR", "#b58b2a"], ["Colour", "#8AA"], ["VFX", "#b06634"], ["Optional", "#a1309b"]].forEach(([t, c]) => {
                const d = el("span", "rw-dot"); const i = el("i"); i.style.background = c; d.append(i, document.createTextNode(t)); chips.appendChild(d);
            });
            f.appendChild(chips);
            return f;
        },
        body(close) {
            const f = document.createDocumentFragment();
            f.appendChild(items([
                { mark: "1", text: "Select the nodes that belong together." },
                { mark: "2", text: "Right-click the canvas and choose ◎ Create Smart Backdrop. With nothing selected it is an empty one at the pointer." },
                { mark: "3", text: "Select it and press the lock in the bar above it, so it stays put while you move the nodes inside." },
            ], true));
            const a = el("div", "rw-actions");
            a.appendChild(softButton("Show me on the canvas", () => {
                close();
                toast("Backdrops", "Select some nodes, then right-click the canvas and choose ◎ Create Smart Backdrop.");
            }));
            f.appendChild(a);
            return f;
        },
    },
    gizmos: {
        kicker: "GIZMOS", title: "Turn a setup into one node",
        lead: "A gizmo packs a group of nodes into a single node of your own, like a Nuke gizmo, with only the inputs and knobs you choose.",
        art() {
            const wrap = el("div"); wrap.style.cssText = "display:flex;flex-direction:column;align-items:center;gap:12px";
            wrap.setAttribute("role", "img");
            wrap.setAttribute("aria-label", "Four selected nodes become one gizmo node with two inputs, one output and two knobs");
            const row = el("div"); row.style.cssText = "display:flex;gap:8px";
            for (let i = 0; i < 4; i++) { const n = el("div"); n.style.cssText = "width:62px;height:70px;border-radius:6px;background:#26262a;border:1.5px solid #0a84ff"; row.appendChild(n); }
            const arrow = svg('<svg width="20" height="28" viewBox="0 0 20 28" aria-hidden="true"><path d="M10 2v22M3 17l7 7 7-7" fill="none" stroke="#aeaeb3" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/></svg>');
            const node = el("div"); node.style.cssText = "width:220px;border-radius:8px;background:#26262a;border:1px solid #48484a;overflow:hidden";
            const t = el("div", null, "◎ Chrome Aberration"); t.style.cssText = "padding:8px 12px;background:#303034;font-size:13px;font-weight:600";
            const b = el("div"); b.style.cssText = "padding:10px 12px;display:flex;flex-direction:column;gap:7px;font-size:12px;color:#bcbcc1";
            const line = (l, r) => { const d = el("div"); d.style.cssText = "display:flex;justify-content:space-between"; d.append(el("span", null, l), el("span", null, r || "")); return d; };
            const knob = (l, v) => { const d = line(l, v); d.style.cssText += ";padding:4px 8px;border-radius:5px;background:#1c1c1e"; return d; };
            b.append(line("● image", "image ●"), line("● mask"), knob("strength", "0.60"), knob("spread", "2.0"));
            node.append(t, b);
            wrap.append(row, arrow, node);
            return wrap;
        },
        body() {
            return items([
                { mark: "1", text: "Select the nodes, then right-click the canvas and choose ◎ Create Radiance Gizmo…" },
                { mark: "2", text: "Name it, and tick the inputs, outputs and knobs the node should show." },
                { mark: "3", text: "Restart ComfyUI. It is in the node menu under Radiance > ◎ Gizmos, ready to share as a .gizmo file." },
            ], true);
        },
    },
    learn: {
        kicker: "LEARN MORE", title: "Guides and help",
        lead: "The documentation, every node and its inputs, and the release notes are on our website.",
        art() {
            const f = document.createDocumentFragment();
            f.appendChild(svg('<svg width="96" height="96" viewBox="0 0 100 100" aria-hidden="true"><rect width="100" height="100" rx="22.5" fill="#1C1C1E"/><path d="M52.2 28.4A24 24 0 1 0 71.6 47.8" fill="none" stroke="#F5F5F7" stroke-width="14"/><rect x="58" y="28" width="14" height="14" rx="3" fill="#0A84FF"/></svg>'));
            const t = el("div"); t.style.cssText = "display:flex;flex-direction:column;align-items:center;gap:6px";
            const n = el("div", "rw-brand", "FXTD STUDIOS"); n.style.fontSize = "14px";
            const d = el("div", null, "fxtdstudios.com"); d.style.cssText = "font-size:13px;color:#8e8e93";
            t.append(n, d); f.appendChild(t);
            return f;
        },
        body() {
            const f = document.createDocumentFragment();
            const box = el("div", "rw-links");
            [["Radiance on fxtdstudios.com", "Guides, the node reference and what’s new", LINKS.radiance],
             ["Report a problem", "GitHub issues, with your log and workflow", LINKS.issues],
             ["FXTD Studios", "Our other tools and contact", LINKS.studio]].forEach(([title, text, href]) => {
                const a = el("a", "rw-link"); a.href = href; a.target = "_blank"; a.rel = "noopener noreferrer";
                const s = el("span"); s.append(el("b", null, title), el("span", null, text));
                a.append(s, svg(OUT)); box.appendChild(a);
            });
            f.appendChild(box);
            f.appendChild(el("p", "rw-note", "Open this tour again from the \u25CE Radiance menu at the top, or Help > Radiance Tour."));
            return f;
        },
    },
};

/** A ComfyUI (or other) modal is up: its dialog, a PrimeVue mask, or the body made inert by one. */
function otherDialogOpen() {
    const open = [...document.querySelectorAll('[role="dialog"][data-state="open"], [role="dialog"][aria-modal="true"], .p-dialog-mask')]
        .filter((d) => !d.closest(".rw-back") && d.getClientRects().length);
    return open.length > 0 || getComputedStyle(document.body).pointerEvents === "none";
}

// ── The dialog ──────────────────────────────────────────────────────────────
let current = null;   // { back, close } while open

function openWelcome(view = "tour", { page = TOUR_PAGES[0], from = "" } = {}) {
    if (current) current.close({ record: false });
    ensureCSS();
    const back = el("div", "rw-back");
    back.dataset.radianceWelcome = view;
    const card = el("div", `rw-card${view === "update" ? " is-update" : ""}`);
    card.setAttribute("role", "dialog");
    card.setAttribute("aria-modal", "true");
    card.setAttribute("aria-labelledby", "rw-title");
    card.tabIndex = -1;
    back.appendChild(card);
    const before = document.activeElement;
    const state = { page, noShow: !getSetting(SHOW_ID, true) };

    const close = ({ record = true } = {}) => {
        if (!back.isConnected) return;
        document.removeEventListener("keydown", onKey, true);
        back.remove();
        current = null;
        if (record) setSetting(LAST_ID, RADIANCE_VERSION);
        try { before?.focus?.(); } catch { /* gone */ }
    };

    const foot = (buttons) => {
        const f = el("div", "rw-foot");
        const label = el("label");
        const box = el("input"); box.type = "checkbox"; box.checked = state.noShow;
        box.addEventListener("change", () => { state.noShow = box.checked; setSetting(SHOW_ID, !box.checked); });
        label.append(box, document.createTextNode("Don't show at startup"));
        f.appendChild(label);
        buttons.forEach((b) => f.appendChild(b));
        return f;
    };
    const button = (label, cls, onClick) => { const b = el("button", `rw-btn ${cls}`, label); b.type = "button"; b.addEventListener("click", onClick); return b; };
    const header = (count) => {
        const h = el("div", "rw-head");
        const comfy = el("span", "rw-brand", "COMFYUI"); comfy.style.color = "#409cff";
        h.append(svg(RING), el("span", "rw-brand", "RADIANCE"), comfy);
        if (count) h.appendChild(el("span", "rw-count", count));
        const x = el("button", "rw-close"); x.type = "button"; x.setAttribute("aria-label", "Close"); x.appendChild(svg(CLOSE));
        if (!count) x.style.marginLeft = "auto";
        x.addEventListener("click", () => close());
        h.appendChild(x);
        return h;
    };

    const renderTour = (focusPrimary = false) => {
        card.className = "rw-card";
        card.replaceChildren();
        const i = TOUR_PAGES.indexOf(state.page);
        const p = PAGES[state.page];
        card.dataset.page = state.page;
        const art = el("div", "rw-art"); art.appendChild(p.art());
        const main = el("div", "rw-main");
        main.appendChild(header(`${i + 1} / ${TOUR_PAGES.length}`));
        main.appendChild(el("div", "rw-kicker", p.kicker));
        const h1 = el("h1", "rw-title", p.title); h1.id = "rw-title"; main.appendChild(h1);
        main.appendChild(el("p", "rw-lead", p.lead));
        main.appendChild(p.body(close));
        const dots = el("div", "rw-dots"); dots.setAttribute("aria-hidden", "true");
        TOUR_PAGES.forEach((_, k) => dots.appendChild(el("i", k === i ? "on" : "")));
        const last = i === TOUR_PAGES.length - 1;
        main.appendChild(foot([
            dots,
            i === 0 ? button("Skip tour", "", () => close()) : button("Back", "", () => go(-1)),
            button(last ? "Get started" : "Next", "is-primary", () => (last ? close() : go(1))),
        ]));
        card.append(art, main);
        if (focusPrimary) card.querySelector(".rw-btn.is-primary")?.focus({ preventScroll: true });
        else card.focus({ preventScroll: true });
    };
    const go = (d) => { state.page = stepPage(state.page, d); renderTour(true); };

    const renderUpdate = () => {
        card.replaceChildren();
        card.dataset.page = "update";
        card.appendChild(header(""));
        const v = el("div", "rw-ver");
        if (from) v.append(el("span", null, from), svg('<svg width="16" height="10" viewBox="0 0 16 10" aria-hidden="true"><path d="M1 5h13M10 1l4 4-4 4" fill="none" stroke="#aeaeb3" stroke-width="1.4" stroke-linecap="round" stroke-linejoin="round"/></svg>'));
        v.appendChild(el("b", null, RADIANCE_VERSION));
        card.appendChild(v);
        const h1 = el("h1", "rw-title", "What's new in this update"); h1.id = "rw-title"; h1.style.fontSize = "28px"; h1.style.marginTop = "10px";
        card.appendChild(h1);
        const notes = notesFor(RADIANCE_VERSION);
        if (notes.length) {
            const ul = el("ul", "rw-bul");
            notes.forEach((t) => ul.appendChild(el("li", null, t)));
            card.appendChild(ul);
        }
        const p = el("p", "rw-note"); p.style.marginTop = "16px";
        const a = el("a", "rw-a", "Full release notes"); a.href = LINKS.radiance; a.target = "_blank"; a.rel = "noopener noreferrer";
        p.appendChild(a); card.appendChild(p);
        const spacer = el("div"); spacer.style.flex = "1"; card.appendChild(spacer);
        const f = foot([
            button("Take the tour", "", () => { state.page = TOUR_PAGES[0]; back.dataset.radianceWelcome = "tour"; renderTour(); }),
            button("Got it", "is-primary", () => close()),
        ]);
        f.children[1].style.marginLeft = "auto";
        card.appendChild(f);
        card.focus({ preventScroll: true });
    };

    function onKey(e) {
        if (!back.isConnected) return;
        if (e.key === "Escape") { e.preventDefault(); e.stopPropagation(); close(); return; }
        const inTour = back.dataset.radianceWelcome === "tour";
        const typing = /^(INPUT|TEXTAREA|SELECT)$/.test(document.activeElement?.tagName || "") && document.activeElement.type !== "checkbox";
        if (inTour && !typing && (e.key === "ArrowRight" || e.key === "ArrowLeft")) {
            e.preventDefault(); e.stopPropagation(); go(e.key === "ArrowRight" ? 1 : -1); return;
        }
        if (e.key === "Tab") {   // keep focus in the dialog
            const f = [...card.querySelectorAll('button, a[href], input')].filter((x) => !x.disabled);
            if (!f.length) return;
            const first = f[0], lastEl = f[f.length - 1];
            if (e.shiftKey && document.activeElement === first) { e.preventDefault(); lastEl.focus(); }
            else if (!e.shiftKey && document.activeElement === lastEl) { e.preventDefault(); first.focus(); }
            else if (!card.contains(document.activeElement)) { e.preventDefault(); first.focus(); }
        }
        e.stopPropagation();   // the graph's shortcuts wait while the tour is open
    }
    document.addEventListener("keydown", onKey, true);
    back.addEventListener("pointerdown", (e) => { if (e.target === back) close(); });

    document.body.appendChild(back);
    if (view === "update") renderUpdate(); else renderTour();
    current = { back, close };
    return current;
}

window.radianceWelcome = { open: (view, opts) => openWelcome(view, opts), close: () => current?.close() };

app.registerExtension({
    name: "Radiance.Welcome",
    settings: [
        {
            id: SHOW_ID,
            category: ["Radiance", "Welcome", "Show at startup"],
            name: "Show the Radiance tour at startup (first install, and What's new after an update)",
            type: "boolean",
            defaultValue: true,
        },
        { id: LAST_ID, name: "Radiance version the welcome last showed", type: "hidden", defaultValue: "" },
    ],
    commands: [
        { id: "Radiance.Welcome.Open", label: "Radiance Tour", icon: "pi pi-compass", function: () => openWelcome("tour") },
    ],
    menuCommands: [{ path: ["Help"], commands: ["Radiance.Welcome.Open"] }],
    async setup() {
        // After the graph and the settings have loaded, and out of the way of
        // the first paint. ComfyUI opens its own template browser on a first
        // run, a modal that takes the page's pointer: wait until no other
        // dialog is open rather than open underneath it.
        let tries = 0;
        const tryShow = () => {
            try {
                if (otherDialogOpen() && tries++ < 600) { setTimeout(tryShow, 1000); return; }
                const lastVersion = String(getSetting(LAST_ID, "") || "");
                const view = startupView({ lastVersion, current: RADIANCE_VERSION, showAtStartup: getSetting(SHOW_ID, true) !== false });
                if (view) openWelcome(view, { from: lastVersion });
            } catch (e) {
                console.warn("[Radiance] Welcome tour:", e);
            }
        };
        setTimeout(tryShow, 1500);
    },
});
