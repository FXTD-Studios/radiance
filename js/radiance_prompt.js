import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import { fitNodeSize, isInputLive, liveSourceNode, loaderModelType, modelMetaSourceNode, setWidgetVisible } from "./radiance_widget_utils.js";

// ALBABIT-FIX: only known post-execution (resolved_arch depends on the real
// CLIP/model_meta), same convention as radiance_vae_widgets.js's
// overexposure markers (engine.py's "ui" channel via onExecuted).
const WEAK_NEG_MARKER = " ⚠ no CFG, ignored";

// Returns whether the label changed.
function _setLabelMarker(widget, marker) {
    if (!widget) return false;
    if (widget._radOrigLabel === undefined && !marker) return false;
    if (widget._radOrigLabel === undefined) widget._radOrigLabel = widget.label ?? widget.name;
    const wanted = marker ? widget._radOrigLabel + marker : widget._radOrigLabel;
    if (widget.label === wanted) return false;
    widget.label = wanted;
    return true;
}

// ALBABIT-FIX: live read of the connected Loader's preset/model_type. No
// source (unconnected or bypassed) means false, not null, same as
// isSigmaOverrideActive() in radiance_sampler.js. null is reserved for a
// connected, live Loader in Auto-Detect, resolved server-side only.
function _liveMiniMaxState(node, sourceNode = modelMetaSourceNode(node)) {
    if (!sourceNode) return false;
    const presetW = sourceNode.widgets?.find(w => w.name === "preset");
    if (!presetW) return false;
    if (String(presetW.value).startsWith("MiniMax H3")) return true;
    if (presetW.value !== "Custom") return false;
    const modelTypeW = sourceNode.widgets?.find(w => w.name === "model_type");
    if (!modelTypeW) return false;
    if (modelTypeW.value === "minimax") return true;
    if (modelTypeW.value === "Auto-Detect") return null;
    return false;
}

// ALBABIT-FIX: ComfyUI skips the uncond pass at cfg 1, so the negative is unused
// when every sampler it feeds runs at cfg 1. Still read there: *_cfg_pp samplers,
// LTX-AV's audio_cfg, a Self-Attention Guidance patch. Any consumer this cannot
// read (a node in between, cfg on an input) keeps the field editable.
const CFG_ONE_SAMPLERS = { RadianceSamplerPro: "sampler", KSampler: "sampler_name", KSamplerAdvanced: "sampler_name" };

function _modelChainHasSag(graph, sampler) {
    let node = sampler;
    for (let depth = 0; node && depth < 32; depth++) {
        if (node.comfyClass === "SelfAttentionGuidance") return true;
        const link = graph.links[node.inputs?.find(i => i.name === "model")?.link];
        node = link ? graph.getNodeById(link.origin_id) : null;
    }
    return false;
}

function _samplerIgnoresNegative(graph, sampler) {
    const samplerField = CFG_ONE_SAMPLERS[sampler.comfyClass];
    if (!samplerField || sampler.inputs?.find(i => i.name === "cfg")?.link != null) return false;
    const value = name => sampler.widgets?.find(w => w.name === name)?.value;
    const audioCfg = sampler.widgets?.find(w => w.name === "audio_cfg");
    return Math.abs(Number(value("cfg")) - 1) < 1e-9  // comfy's math.isclose(cfg, 1.0)
        && !String(value(samplerField)).includes("cfg_pp")
        && !(audioCfg && !audioCfg.hidden && audioCfg.value > 0 && audioCfg.value !== 1)
        && !_modelChainHasSag(graph, sampler);
}

// ALBABIT-FIX: node.graph, not app.graph: it is the graph holding this node's
// links, a subgraph in the Comfy-Org templates.
function _negativeUnusedDownstream(node) {
    const graph = node.graph;
    const output = name => node.outputs?.find(o => o.name === name);
    if (!graph || output("negative_text")?.links?.length) return false;
    const samplers = (output("negative")?.links ?? [])
        .map(id => graph.links[id])
        .map(link => link && graph.getNodeById(link.target_id))
        .filter(n => n && n.mode !== 2 && n.mode !== 4);
    return samplers.length > 0 && samplers.every(n => _samplerIgnoresNegative(graph, n));
}

// ALBABIT-FIX: same disabled + inputEl + opacity/pointerEvents combination
// already proven in radiance_sampler.js's updateSigmaLocks(). Skips
// reassignment when already correct, this runs from a 250ms poll and would
// otherwise interrupt in-progress typing (same bug class, same fix).
function _applyNegPromptLock(node, locked) {
    const negW = node.widgets?.find(w => w.name === "negative_prompt");
    if (!negW || negW.disabled === locked) return;
    negW.disabled = locked;
    if (negW.inputEl) {
        negW.inputEl.disabled = locked;
        negW.inputEl.style.opacity = locked ? "0.4" : "1.0";
        negW.inputEl.style.pointerEvents = locked ? "none" : "auto";
    }
    node.setDirtyCanvas(true, true);
}

// ALBABIT-FIX: negative_strength (a combo, no reliable disabled/inputEl
// path the way a text widget has) hides entirely instead, same mechanism
// as every preset-driven widget in radiance_loader.js/radiance_resolution.js.
function _applyNegStrengthLock(node, hidden) {
    const strengthW = node.widgets?.find(w => w.name === "negative_strength");
    if (!strengthW || strengthW.hidden === hidden) return;
    setWidgetVisible(strengthW, !hidden, node, { fallbackType: "combo" });
    fitNodeSize(node);
}

// ALBABIT-FIX: resolution only sizes Qwen-Image 2.1's and Flux.3 Klein's reference images, so it
// stays hidden until one arrives from a node that runs, and under the models
// that size them on their own.
function _applyResolutionVisibility(node, source = modelMetaSourceNode(node)) {
    const resW = node.widgets?.find(w => w.name === "resolution");
    const show = !!node.inputs?.some(i => i.name?.startsWith("images.") && isInputLive(node, i))
        && !["flux2", "flux2-klein", "qwen_image"].includes(loaderModelType(source));
    if (!resW || !resW.hidden === show) return;
    setWidgetVisible(resW, show, node, { fallbackType: "number" });
    fitNodeSize(node);
}

// 4.0.0: the prompt upsampler's switch shows once an upsampler is connected
// (upsampler_clip), and stays while on so it can be turned off.
// ALBABIT-FIX: only under a family that has an upsampler, or a Loader left to
// Auto-Detect (the Loader picks its file once the switch is on). The seed and
// its "control after generate" widget only matter, and show, while it is on.
const UPSAMPLER_FAMILIES = ["flux3", "qwen_image21"];

function _applyUpsamplerVisibility(node, source = modelMetaSourceNode(node)) {
    const input = node.inputs?.find(i => i.name === "upsampler_clip");
    const onW = node.widgets?.find(w => w.name === "upsample_prompt");
    if (!input || !onW) return;
    const on = onW.value === true;
    const arch = loaderModelType(source);
    const showSwitch = on || (isInputLive(node, input) && (arch === null || UPSAMPLER_FAMILIES.includes(arch)));
    const seedW = node.widgets.find(w => w.name === "upsampler_seed");
    const targets = [[onW, "toggle", showSwitch], [seedW, "number", on],
        ...(seedW?.linkedWidgets || []).map(w => [w, "combo", on])];
    let changed = false;
    for (const [w, type, show] of targets) {
        if (!w || !w.hidden === show) continue;
        setWidgetVisible(w, show, node, { fallbackType: type });
        changed = true;
    }
    if (changed) fitNodeSize(node);
}

// ALBABIT-FIX: ComfyUI runs "control after generate" on hidden widgets too: a
// hidden "randomize" gave every run a new seed, which re-ran the Prompt and
// everything after it. The control now acts only while the seed is shown.
function _holdHiddenSeed(node) {
    const seedW = node.widgets?.find(w => w.name === "upsampler_seed");
    for (const w of seedW?.linkedWidgets || []) {
        for (const hook of ["beforeQueued", "afterQueued"]) {
            const run = w[hook];
            if (run) w[hook] = function () { if (!seedW.hidden) return run.apply(this, arguments); };
        }
    }
}

// 4.0.0: the FLUX 3 layout input shows under a FLUX 3 Loader, or while it
// holds a table (so another family's run can report it and it can be cleared).
function _applyLayoutVisibility(node, source = modelMetaSourceNode(node)) {
    const w = node.widgets?.find(x => x.name === "layout");
    if (!w) return;
    const show = loaderModelType(source) === "flux3" || !!String(w.value || "").trim();
    if (!w.hidden === show) return;
    setWidgetVisible(w, show, node, { fallbackType: "customtext" });
    fitNodeSize(node);
}

// ALBABIT-FIX: action is read by video models only (sampler_utils.py's
// VIDEO_MODEL_TYPES). Under an image model it is dimmed and marked, its text kept
// and still editable; with no Loader, or one in Auto-Detect, the last run decides.
const VIDEO_FAMILIES = ["wan", "wan_ti2v", "ltxv", "ltxav", "hunyuan_video", "cosmos", "cogvideox", "mochi",
    "minimax", "hunyuan_video_15", "kandinsky5"];
const IMAGE_MODEL_MARKER = " ⚠ image model, ignored";

function _applyActionState(node, source = modelMetaSourceNode(node)) {
    const w = node.widgets?.find(x => x.name === "action");
    if (!w) return;
    const arch = loaderModelType(source);
    w._radIgnored = (arch === null ? node._radVideo : VIDEO_FAMILIES.includes(arch)) === false;
    _setLabelMarker(w, w._radIgnored ? IMAGE_MODEL_MARKER : null);
}

// ALBABIT-FIX: text_to_video off (image-to-video) leaves base_prompt and the menus out:
// base_prompt is dimmed and marked, its text kept, the menus hide as negative_strength does.
// Wired, the frontend draws the switch's row. An image model ignores the switch (action's
// verdict): it hides, unwired.
const IMAGE_TO_VIDEO_MARKER = " ⚠ image-to-video, ignored";

// The switch as the run reads it: unwired, the value given (the shown tab's or another's);
// wired, a Boolean node that runs, else the last run's value.
function _textToVideo(node, own) {
    const input = node.inputs?.find(i => i.name === "text_to_video");
    if (input?.link == null) return own;
    return liveSourceNode(node, input)?.widgets?.find(x => x.name === "value")?.value ?? node._radTextToVideo;
}

function _applyBasePromptState(node) {
    const w = node.widgets?.find(x => x.name === "base_prompt");
    const onW = node.widgets?.find(x => x.name === "text_to_video");
    if (!w || !onW) return;
    const wired = node.inputs?.find(i => i.name === "text_to_video")?.link != null;
    const imageModel = node.widgets.find(x => x.name === "action")?._radIgnored === true;
    const off = !imageModel && _textToVideo(node, onW.value) === false;
    w._radIgnored = off;
    _setLabelMarker(w, off ? IMAGE_TO_VIDEO_MARKER : null);
    const targets = ["style_preset", ...TRACKED_FIELDS].map(name => [node.widgets.find(x => x.name === name), "combo", !off]);
    if (!wired) targets.push([onW, "toggle", !imageModel]);
    let changed = false;
    for (const [target, type, show] of targets) {
        if (!target || !target.hidden === show) continue;
        setWidgetVisible(target, show, node, { fallbackType: type });
        changed = true;
    }
    if (changed) fitNodeSize(node);
}

// ALBABIT-FIX: Qwen-Image Edit's native encoder takes 3 images. Autogrow adds a
// slot only below comfyDynamic.autogrow max; past a lowered max, empty slots go
// and a connected one stays for the run to report.
// FLUX 3 Image Klein reads up to 10 reference images (BFL's reference code).
const REFERENCE_LIMITS = { qwen_image: 3, flux3: 10 };

function _applyReferenceLimit(node, source = modelMetaSourceNode(node)) {
    const grow = node.comfyDynamic?.autogrow?.images;
    if (!grow) return;
    const max = REFERENCE_LIMITS[loaderModelType(source)] ?? grow.names.length;
    if (grow.max === max) return;
    const raised = max > grow.max;
    grow.max = max;
    const slots = node.inputs.filter(i => i.name.startsWith("images."));
    if (raised) {
        // The frontend adds the next slot when the last one connects: replay it.
        const last = slots.at(-1);
        const link = last?.link != null ? node.graph?.links?.[last.link] : null;
        if (link) node.onConnectionsChange?.(LiteGraph.INPUT, node.inputs.indexOf(last), true, link, last);
    } else {
        for (const slot of slots.slice(max).reverse()) {
            if (slot.link == null) node.removeInput(node.inputs.indexOf(slot));
        }
    }
    fitNodeSize(node);
}

// ALBABIT-FIX: the text boxes shared the node's height equally, so a two-line
// negative took as much room as a long prompt. Each box now fits its text, long
// ones split what is left, and the first box not sized by hand takes the rest.
// The bar under a box sizes it by hand (blue); a double-click undoes it (grey).
const TEXT_MIN = 64;                            // the frontend's textarea minimum
const TEXT_HEIGHTS = "radiance_text_heights";   // node.properties: {widget name: px}
const TEXT_BAR_TIP = "Height of the box above.\n• Drag: set it by hand (blue bar).\n• Double-click: back to automatic (grey bar).";

// Each box's height set by hand, or null. The first box left automatic fills
// the free space; the last box does when all of them were set by hand.
function _textPlan(names, heights) {
    const manual = names.map(n => heights?.[n] ?? null);
    const fill = manual.includes(null) ? manual.indexOf(null) : names.length - 1;
    return manual.map((height, i) => ({ height: i === fill ? null : height, fill: i === fill }));
}

// The boxes set by hand keep their height; the others get up to their text,
// sharing equally what is left when it runs short; the fill box also takes
// whatever remains after that.
function _textShares(plans, needs, total) {
    const shares = plans.map(p => p.height);
    let left = total - shares.reduce((sum, h) => sum + (h ?? 0), 0);
    const open = plans.map((p, i) => i).filter(i => shares[i] == null).sort((a, b) => needs[a] - needs[b]);
    open.forEach((i, k) => {
        shares[i] = Math.max(TEXT_MIN, Math.min(needs[i], Math.floor(left / (open.length - k))));
        left -= shares[i];
    });
    shares[plans.findIndex(p => p.fill)] += Math.max(0, left);
    return shares;
}

function _textHeight(el) {
    if (el._radText !== el.value || el._radWidth !== el.clientWidth) {
        const height = el.style.height;
        el.style.height = "0px";
        el._radNeed = Math.max(TEXT_MIN, el.scrollHeight);
        el.style.height = height;
        el._radText = el.value;
        el._radWidth = el.clientWidth;
    }
    return el._radNeed;
}

function _addStyle(id, css) {
    if (document.getElementById(id)) return;
    const style = document.createElement("style");
    style.id = id;
    style.textContent = css;
    document.head.append(style);
}

function _makeTextBar(node) {
    _addStyle("radiance-text-bar-style", `
        .radiance-text-bar { position: absolute; right: 0; bottom: -7px; height: 10px; z-index: 20; cursor: ns-resize;
            display: flex; align-items: center; justify-content: center; }
        .radiance-text-bar > span { width: 44px; height: 4px; border-radius: 2px; background: #8a8a94;
            box-shadow: 0 0 0 2px rgba(20, 20, 25, 0.9); }
        .radiance-text-bar:hover > span, .radiance-text-bar.radiance-manual > span { background: #39aaff; }`);
    const bar = document.createElement("div");
    bar.className = "radiance-text-bar";
    bar.title = TEXT_BAR_TIP;
    bar.append(document.createElement("span"));
    const setHeight = (height) => {
        const heights = (node.properties[TEXT_HEIGHTS] ??= {});
        const name = bar._radEntries[bar._radIndex].widget.name;
        if (height == null) delete heights[name];
        else heights[name] = height;
        if (!Object.keys(heights).length) delete node.properties[TEXT_HEIGHTS];
        _layoutTextBoxes(node);
        node.setDirtyCanvas(true, true);
    };
    // The window's mouseup, after these, records the change for undo and saving.
    bar.addEventListener("mousedown", e => { e.preventDefault(); e.stopPropagation(); });
    bar.addEventListener("mouseup", e => { if (e.detail === 2) setHeight(null); });
    bar.addEventListener("pointerdown", e => {
        if (e.button !== 0) return;
        e.stopPropagation();
        bar.setPointerCapture(e.pointerId);
        const entries = bar._radEntries, i = bar._radIndex;
        const start = entries[i].el.offsetHeight, y = e.clientY, scale = app.canvas.ds.scale;
        // What the other automatic boxes can give up.
        const room = entries.reduce((sum, b, j) => j === i || b.plan.height != null ? sum
            : sum + Math.max(0, b.el.offsetHeight - TEXT_MIN), 0);
        const move = ev => setHeight(Math.round(Math.min(Math.max(start + (ev.clientY - y) / scale, TEXT_MIN), start + room)));
        bar.addEventListener("pointermove", move);
        bar.addEventListener("lostpointercapture", () => bar.removeEventListener("pointermove", move), { once: true });
    });
    return bar;
}

// Typing resizes the boxes at once, not at the next poll. A box the model
// ignores is dimmed. Every box but the last has a bar under it, in its host.
function _attachTextBox(node, entries, i, host, left) {
    const { el, widget } = entries[i];
    if (!el._radWatched) {
        el._radWatched = true;
        el.addEventListener("input", () => _layoutTextBoxes(node));
    }
    if (widget._radIgnored !== undefined) el.style.opacity = widget._radIgnored ? "0.5" : "";
    // Nodes 2.0 (1.55) leaves out the placeholder the node gives a text box.
    if (el !== widget.element) el.placeholder ||= node.constructor.nodeData?.input?.optional?.[widget.name]?.[1]?.placeholder ?? "";
    let bar = host.querySelector(":scope > .radiance-text-bar");
    if (i === entries.length - 1) return bar?.remove();
    if (!bar) {
        bar = _makeTextBar(node);
        host.append(bar);
    }
    bar._radEntries = entries;
    bar._radIndex = i;
    bar.style.left = `${left}px`;
    bar.classList.toggle("radiance-manual", entries[i].plan.height != null);
}

// Nodes 2.0 gives each text box an "auto" grid row, shared equally. Each box
// that is not the fill one gets its share as the row's maximum, the fill one
// "1fr": the rest, and the boxes still shrink when the node does.
function _layoutVueBoxes(node, grid, widgets, plans) {
    const rows = [...grid.children];
    const areas = [...grid.querySelectorAll("textarea")];
    const base = grid.style.gridTemplateRows === grid._radRows ? grid._radBase : grid.style.gridTemplateRows;
    const tracks = base.split(" ");
    // A frontend that lays the node out another way keeps its own layout.
    if (areas.length !== widgets.length || tracks.length !== rows.length) return;
    const entries = areas.map((el, i) => ({ widget: widgets[i], plan: plans[i], el, row: rows.find(r => r.contains(el)) }));
    const textRows = new Set(entries.map(b => b.row));
    const gap = parseFloat(getComputedStyle(grid).rowGap) || 0;
    const total = rows.reduce((sum, r) => textRows.has(r) ? sum : sum - r.offsetHeight, grid.clientHeight)
        - gap * (rows.length - 1);
    const shares = _textShares(plans, entries.map(b => _textHeight(b.el)), total);
    entries.forEach((b, i) => {
        b.row.style.position = "relative";
        tracks[rows.indexOf(b.row)] = b.plan.fill ? `minmax(${TEXT_MIN}px, 1fr)` : `minmax(${TEXT_MIN}px, ${shares[i]}px)`;
        _attachTextBox(node, entries, i, b.row, b.el.parentElement.offsetLeft);
    });
    const value = tracks.join(" ");
    grid._radBase = base;
    if (grid._radValue !== value || grid.style.gridTemplateRows !== grid._radRows) {
        grid.style.gridTemplateRows = value;
        grid._radValue = value;
        grid._radRows = grid.style.gridTemplateRows;
    }
    // Each re-render of the node writes the frontend's rows back (a wheel in a
    // box does it): put ours back before the frame is painted.
    if (!grid._radObserver) {
        grid._radObserver = new MutationObserver(() => {
            if (grid.style.gridTemplateRows !== grid._radRows) _layoutTextBoxes(node);
        });
        grid._radObserver.observe(grid, { attributes: true, attributeFilter: ["style"] });
    }
}

// The canvas shares its free space equally between the text boxes, each up to
// its getMaxHeight: every box gets its share as that maximum, so the shares
// still shrink with the node. A slot is its textarea plus a margin each side.
function _layoutCanvasBoxes(node, widgets, plans) {
    const entries = widgets.map((widget, i) => ({ widget, plan: plans[i], el: widget.element }))
        .filter(b => b.el?.parentElement);
    if (node.freeWidgetSpace == null || entries.length !== widgets.length) return;
    const margins = entries.map(b => 2 * (b.widget.margin ?? 0));
    const total = margins.reduce((sum, m) => sum - m, node.freeWidgetSpace);
    const shares = _textShares(plans, entries.map(b => _textHeight(b.el)), total);
    let changed = false;
    entries.forEach((b, i) => {
        const max = shares[i] + margins[i];
        changed ||= b.widget._radMax !== max;
        b.widget._radMax = max;
        b.widget.options.getMaxHeight = () => max;
        _attachTextBox(node, entries, i, b.el.parentElement, 0);
    });
    if (changed) node.setDirtyCanvas(true, true);
}

function _layoutTextBoxes(node) {
    const linked = new Set((node.inputs ?? []).filter(i => i.widget && i.link != null).map(i => i.widget.name));
    const widgets = (node.widgets ?? []).filter(w => w.type === "customtext" && !w.hidden && !linked.has(w.name));
    if (!widgets.length) return;
    const plans = _textPlan(widgets.map(w => w.name), node.properties?.[TEXT_HEIGHTS]);
    if (!LiteGraph.vueNodesMode) return _layoutCanvasBoxes(node, widgets, plans);
    const grid = node.graph === app.canvas?.graph && document.querySelector(`[data-node-id="${node.id}"] .lg-node-widgets`);
    if (grid) _layoutVueBoxes(node, grid, widgets, plans);
}

// ALBABIT-FIX: tabs keep several prompts in one node. A tab holds every widget
// value: the shown tab's are in the widgets, the others' in node.properties. A new
// tab starts from the node's defaults; "next tab" moves on after each queued run.
const TABS = "radiance_prompt_tabs";    // node.properties: {active, chain, tabs: [{name, values}]}
const TAB_TIP = "A prompt tab: its own text, action, negative and settings.\n• Click: show it.\n• Drag: move it.\n• Double-click: rename it.\n• Right-click: rename, duplicate or delete it, or run every tab that has a prompt.";
const TAB_ADD_TIP = "New tab, from the node's defaults.";
const TAB_CHAIN_TIP = "After each run:\n• fixed: stay on this tab.\n• next tab: show the next tab that has a prompt. A batch count of 3 runs three tabs.";

function _readTabs(node) {
    return node.properties[TABS] ?? { active: 0, chain: false, tabs: [{ name: "Prompt A" }, { name: "Prompt B" }] };
}

function _editTabs(node) {
    return (node.properties[TABS] = _readTabs(node));
}

function _tabValues(node) {
    return Object.fromEntries(node.widgets.filter(w => w.serialize !== false).map(w => [w.name, w.value]));
}

// ALBABIT-FIX: what the encoder reads. A video model also runs on the action alone, and on
// it alone with text_to_video off; an image model reads base_prompt only (action's verdict).
function _hasPrompt(node, values) {
    const video = node.widgets.find(w => w.name === "action")?._radIgnored !== true;
    const base = !(video && _textToVideo(node, values?.text_to_video) === false) && values?.base_prompt?.trim();
    return !!(base || (video && values?.action?.trim()));
}

// Shows tab i over the widgets' values: save those first to keep them.
function _showTab(node, state, i) {
    const values = { ...node._radFreshTab, ...state.tabs[i].values };
    for (const w of node.widgets) if (w.name in values) w.value = values[w.name];
    node._radPreset = values.style_preset;
    delete state.tabs[i].values;
    state.active = i;
    _refreshLiveState(node);
    node.setDirtyCanvas(true, true);
}

function _selectTab(node, i) {
    const state = _editTabs(node);
    if (i === state.active) return;
    state.tabs[state.active].values = _tabValues(node);
    _showTab(node, state, i);
}

function _insertTab(node, at, tab) {
    const state = _editTabs(node);
    state.tabs.splice(at, 0, tab);
    if (state.active >= at) state.active++;
    _selectTab(node, at);
}

function _newTab(node) {
    const state = _readTabs(node);
    const names = new Set(state.tabs.map(t => t.name));
    const name = n => `Prompt ${n < 26 ? String.fromCharCode(65 + n) : n + 1}`;
    let n = state.tabs.length;
    while (names.has(name(n))) n++;
    _insertTab(node, state.tabs.length, { name: name(n) });
}

function _duplicateTab(node, i) {
    const state = _readTabs(node);
    const values = i === state.active ? _tabValues(node) : { ...state.tabs[i].values };
    _insertTab(node, i + 1, { name: `${state.tabs[i].name} copy`, values });
}

function _deleteTab(node, i) {
    const state = _editTabs(node);
    if (state.tabs.length < 2) return;
    if (i === state.active) _showTab(node, state, i ? i - 1 : 1);
    state.tabs.splice(i, 1);
    if (state.active > i) state.active--;
    _renderTabs(node);
}

function _renameTab(node, el, i) {
    const input = document.createElement("input");
    input.value = _readTabs(node).tabs[i].name;
    el.replaceChildren(input);
    input.focus();
    input.select();
    const finish = save => {
        if (!input.isConnected) return;
        const name = input.value.trim();
        if (save && name) _editTabs(node).tabs[i].name = name;
        el.textContent = _readTabs(node).tabs[i].name;
        _renderTabs(node);
    };
    input.addEventListener("keydown", e => { if (e.key === "Enter" || e.key === "Escape") finish(e.key === "Enter"); });
    input.addEventListener("blur", () => finish(true));
}

// "next tab": after a run is queued, the next tab that has a prompt.
function _chainTab(node) {
    const state = _readTabs(node);
    if (!state.chain || node._radRunningTabs) return;
    for (let k = 1; k < state.tabs.length; k++) {
        const i = (state.active + k) % state.tabs.length;
        if (_hasPrompt(node, state.tabs[i].values)) return _selectTab(node, i);
    }
}

function _promptedTabs(node) {
    const state = _readTabs(node);
    const shown = _hasPrompt(node, _tabValues(node));
    return state.tabs.map((tab, i) => i).filter(i => i === state.active ? shown : _hasPrompt(node, state.tabs[i].values));
}

// One run per tab that has a prompt, in order, then back to the tab shown
// before; "next tab" sits it out. A run that fails to build, or that the server
// refuses, stops the rest.
// While the frontend is queueing another run, queuePrompt adds this one behind it and
// returns at once: the run is built later, from the tab shown then. Wait for the queue.
async function _queueIdle() {
    while (app.processingQueue) await new Promise(resolve => setTimeout(resolve, 50));
}

async function _runAllTabs(node) {
    await _queueIdle();
    const start = _readTabs(node).active;
    node._radRunningTabs = true;
    try {
        for (const i of _promptedTabs(node)) {
            _selectTab(node, i);
            if (!await app.queuePrompt(0)) {
                await _queueIdle();
                break;
            }
        }
    } finally {
        node._radRunningTabs = false;
        _selectTab(node, start);
    }
}

function _moveTab(node, from, to) {
    const state = _editTabs(node), shown = state.tabs[state.active];
    state.tabs.splice(to, 0, ...state.tabs.splice(from, 1));
    state.active = state.tabs.indexOf(shown);
}

// A dragged tab follows the pointer; a neighbour it passes the middle of steps
// aside. The new order is saved on release. Near an edge, the row scrolls.
function _dragTab(node, el, down) {
    const list = el.parentElement, from = [...list.children].indexOf(el);
    const grab = down.clientX - el.getBoundingClientRect().left;
    const middle = t => { const r = t.getBoundingClientRect(); return r.left + r.width / 2; };
    let dragging = false;
    const move = e => {
        if (!dragging && Math.abs(e.clientX - down.clientX) < 4) return;
        dragging = true;
        el.classList.add("radiance-dragging");
        const box = list.getBoundingClientRect();
        if (e.clientX < box.left + 16) list.scrollLeft -= 8;
        else if (e.clientX > box.right - 16) list.scrollLeft += 8;
        el.style.transform = "";
        const width = el.getBoundingClientRect().width, center = e.clientX - grab + width / 2;
        // Move the neighbour, not the tab: moving it would release the pointer.
        if (el.nextSibling && center > middle(el.nextSibling)) list.insertBefore(el.nextSibling, el);
        else if (el.previousSibling && center < middle(el.previousSibling)) el.after(el.previousSibling);
        const scale = width / el.offsetWidth;
        el.style.transform = `translateX(${(e.clientX - grab - el.getBoundingClientRect().left) / scale}px)`;
    };
    el.setPointerCapture(down.pointerId);
    el.addEventListener("pointermove", move);
    el.addEventListener("pointerup", () => {
        el.removeEventListener("pointermove", move);
        if (!dragging) return;
        el.classList.remove("radiance-dragging");
        el.style.transform = "";
        _moveTab(node, from, [...list.children].indexOf(el));
        list._radNames = null;
        _renderTabs(node);
    }, { once: true });
}

const TAB_PEEK = 32;    // px of the next tab left in view, so the hidden tabs show

function _revealTab(list, el) {
    list.scrollLeft = Math.min(Math.max(list.scrollLeft, el.offsetLeft + el.offsetWidth + TAB_PEEK - list.clientWidth),
        el.offsetLeft - TAB_PEEK);
}

// An arrow brings in the next tab cut off on its side.
function _scrollTabs(list, dir) {
    const tabs = [...list.children], start = list.scrollLeft, end = start + list.clientWidth;
    const next = dir > 0 ? tabs.find(t => t.offsetLeft + t.offsetWidth > end + 1) : tabs.findLast(t => t.offsetLeft < start - 1);
    if (next) _revealTab(list, next);
}

// The arrows show when the tabs do not fit (counting the room they take) and dim
// at either end; a side with hidden tabs fades out.
function _updateTabScroll(strip) {
    const [left, list, right] = strip.children;
    const arrows = strip.classList.contains("radiance-overflow") ? left.offsetWidth + right.offsetWidth + 6 : 0;
    const over = list.scrollWidth > list.clientWidth + arrows;
    strip.classList.toggle("radiance-overflow", over);
    const before = over && list.scrollLeft > 0, after = over && list.scrollLeft + list.clientWidth < list.scrollWidth - 1;
    left.classList.toggle("radiance-off", !before);
    right.classList.toggle("radiance-off", !after);
    list.style.maskImage = before || after
        ? `linear-gradient(to right, ${before ? "transparent" : "#000"}, #000 16px, #000 calc(100% - 16px), ${after ? "transparent" : "#000"})` : "";
}

function _tabButton(className, text, tip, onPress) {
    const el = document.createElement("div");
    el.className = `radiance-tab-button ${className}`;
    el.textContent = text;
    el.title = tip;
    el.addEventListener("mousedown", e => e.stopPropagation());
    el.addEventListener("pointerdown", e => {
        e.stopPropagation();
        if (e.button === 0) onPress(e);
    });
    return el;
}

// Rebuilt when the names change, so a double-click lands on the tab it started on.
function _renderTabs(node) {
    const strip = node.widgets?.find(w => w.name === "prompt_tabs")?.element.firstChild;
    if (!strip || strip.querySelector("input, .radiance-dragging")) return;
    const list = strip.children[1];
    const state = _readTabs(node);
    const names = JSON.stringify(state.tabs.map(t => t.name));
    if (list._radNames !== names) {
        list._radNames = names;
        list._radShown = null;
        list.replaceChildren(...state.tabs.map((tab, i) => {
            const el = _tabButton("", tab.name, `${tab.name}\n${TAB_TIP}`, e => {
                _selectTab(node, i);
                _dragTab(node, el, e);
            });
            el.addEventListener("dblclick", e => {
                e.stopPropagation();
                _renameTab(node, el, i);
            });
            el.addEventListener("contextmenu", e => {
                e.preventDefault();
                e.stopPropagation();
                const runs = _promptedTabs(node).length;
                new LiteGraph.ContextMenu([
                    { content: "Rename", callback: () => _renameTab(node, el, i) },
                    { content: "Duplicate", callback: () => _duplicateTab(node, i) },
                    { content: "Delete", disabled: _readTabs(node).tabs.length < 2, callback: () => _deleteTab(node, i) },
                    null,
                    { content: `Run all tabs (${runs})`, disabled: !runs, callback: () => _runAllTabs(node) },
                ], { event: e });
            });
            return el;
        }));
    }
    state.tabs.forEach((tab, i) => {
        list.children[i].classList.toggle("radiance-active", i === state.active);
        list.children[i].classList.toggle("radiance-empty", i !== state.active && !_hasPrompt(node, tab.values));
    });
    strip.classList.toggle("radiance-chain", !!state.chain);
    strip.lastChild.textContent = state.chain ? "⟳ next tab" : "⟳ fixed";
    _updateTabScroll(strip);
    // Once laid out, the shown tab is scrolled into view each time it changes.
    if (list._radShown !== state.active && list.clientWidth) {
        list._radShown = state.active;
        _revealTab(list, list.children[state.active]);
    }
}

// The tab row, above the text boxes. Never saved: the tabs are in node.properties.
// It reads the node's defaults before a graph sets any value.
function _addTabsWidget(node) {
    _addStyle("radiance-prompt-tabs-style", `
        .radiance-prompt-tabs { display: flex; align-items: center; gap: 3px; height: 28px; overflow: hidden; font-size: 12px; }
        .radiance-tab-list { position: relative; display: flex; align-items: center; gap: 3px; height: 100%; min-width: 0; overflow: hidden; }
        .radiance-tab-button { flex: none; height: 22px; line-height: 22px; padding: 0 10px; box-sizing: border-box; border-radius: 6px;
            white-space: nowrap; cursor: pointer; user-select: none; color: var(--node-component-slot-text, #a0a0a0); }
        .radiance-tab-list > .radiance-tab-button { max-width: 140px; overflow: hidden; text-overflow: ellipsis; }
        .radiance-tab-list > .radiance-active { background: var(--component-node-widget-background, #313235);
            color: var(--component-node-foreground, #fff); box-shadow: inset 0 -2px 0 #39aaff; }
        .radiance-tab-list > .radiance-dragging { position: relative; z-index: 1; cursor: grabbing; outline: 1px solid #39aaff;
            box-shadow: inset 0 -2px 0 #39aaff, 0 3px 10px #000a; }
        .radiance-tab-button:has(input) { outline: 1px solid #39aaff; }
        .radiance-chain .radiance-empty { color: color-mix(in srgb, var(--node-component-slot-text, #a0a0a0) 45%, transparent); }
        .radiance-tab-arrow { width: 18px; padding: 0; text-align: center; font-size: 16px; }
        .radiance-prompt-tabs:not(.radiance-overflow) > .radiance-tab-arrow { display: none; }
        .radiance-tab-arrow.radiance-off { opacity: 0.3; cursor: default; }
        .radiance-tab-add { width: 22px; padding: 0; text-align: center; line-height: 20px; font-size: 15px;
            border: 1px dashed var(--border-color, #4e4e4e); }
        .radiance-tab-chain { margin-left: auto; padding: 0 8px; line-height: 20px; font-size: 11px; border: 1px solid var(--border-color, #4e4e4e); }
        .radiance-chain > .radiance-tab-chain { color: #39aaff; background: #39aaff26; border-color: #39aaff88; }
        .radiance-prompt-tabs input { width: 10em; padding: 0; font: inherit; color: inherit; background: none; border: none; outline: none; }`);
    node._radFreshTab = _tabValues(node);
    const strip = document.createElement("div");
    strip.className = "radiance-prompt-tabs";
    const list = document.createElement("div");
    list.className = "radiance-tab-list";
    list.addEventListener("scroll", () => _updateTabScroll(strip));
    strip.append(_tabButton("radiance-tab-arrow", "‹", "", () => _scrollTabs(list, -1)), list,
        _tabButton("radiance-tab-arrow", "›", "", () => _scrollTabs(list, 1)),
        _tabButton("radiance-tab-add", "+", TAB_ADD_TIP, () => _newTab(node)),
        _tabButton("radiance-tab-chain", "", TAB_CHAIN_TIP, () => {
            const edited = _editTabs(node);
            edited.chain = !edited.chain;
            _renderTabs(node);
        }));
    const el = document.createElement("div");
    el.append(strip);
    const widget = node.addDOMWidget("prompt_tabs", "radiance_tabs", el,
        { serialize: false, getMinHeight: () => 28, getMaxHeight: () => 28 });
    widget.serialize = false;
    // ALBABIT-FIX: Nodes 2.0 stretches a row with a layout size like a text box: each time
    // it drew its own rows, the tab row grew to ~160px and the sockets below were measured
    // there, so a link to a widget ended rows below it. computeSize keeps its 28px.
    const { computeLayoutSize, computeSize } = widget;
    Object.defineProperties(widget, {
        computeLayoutSize: { get: () => (LiteGraph.vueNodesMode ? undefined : computeLayoutSize) },
        computeSize: { get: () => (LiteGraph.vueNodesMode ? (width) => [width, 28] : computeSize) },
    });
    // The frontend runs each widget's afterQueued in turn, this one first: wait for
    // the others, so a seed control steps the tab that ran, not the next one.
    widget.afterQueued = () => queueMicrotask(() => _chainTab(node));
    node.widgets.splice(node.widgets.indexOf(widget), 1);
    node.widgets.unshift(widget);
    _renderTabs(node);
}

// ALBABIT-FIX: shared by the poll loop, onConfigure and onExecuted below. A
// Loader in Auto-Detect (liveState null) falls back to the last run's verdict.
// The model_meta source is resolved once per call and shared by the helpers.
function _refreshLiveState(node) {
    updatePresetDivergenceMarkers(node);
    const source = modelMetaSourceNode(node);
    _applyResolutionVisibility(node, source);
    _applyReferenceLimit(node, source);
    _applyUpsamplerVisibility(node, source);
    _applyLayoutVisibility(node, source);
    const liveState = _liveMiniMaxState(node, source);
    const cfgOne = _negativeUnusedDownstream(node);
    const locked = liveState === null ? cfgOne || !!node._radWeakNeg : liveState || cfgOne;
    _applyNegPromptLock(node, locked);
    _applyNegStrengthLock(node, locked);
    const negW = node.widgets?.find(w => w.name === "negative_prompt");
    _setLabelMarker(negW, cfgOne || node._radWeakNeg ? WEAK_NEG_MARKER : null);
    _applyActionState(node, source);
    _applyBasePromptState(node);
    _layoutTextBoxes(node);
    _renderTabs(node);
}

// ALBABIT-FIX: apply_style_preset() overwrote these 7 widgets on every run. They
// are now filled once on selection and a later edit gets a "✎" marker; preset
// fields with no widget here are still applied by Python.
//
// 3.5.0: the preset table comes from Python (nodes/generate/prompt.py,
// /radiance/prompt/presets) instead of a copy kept here by hand. If the
// request fails, picking a preset leaves the widgets alone and Python applies
// the whole preset at run time (it does so whenever all 7 are still "None").
let PRESET_CONFIGS = {};
const presetsReady = api.fetchApi("/radiance/prompt/presets")
    .then(r => (r.ok ? r.json() : {}))
    .then(data => { PRESET_CONFIGS = data || {}; })
    .catch(err => console.warn("[Radiance Cinematic Encoder] presets not loaded:", err));

const PRESET_MARKER = " ✎";
const TRACKED_FIELDS = [
    "framing", "camera_type", "lens_focal", "aperture_dof",
    "lighting", "style_aesthetic", "color_grading",
];

function getPresetConfig(name) {
    return PRESET_CONFIGS[name] || null;
}

async function applyPreset(node, presetName) {
    await presetsReady;
    const config = getPresetConfig(presetName);
    if (!config) return;

    const widgets = node.widgets;
    if (!widgets) return;

    for (const widget of widgets) {
        if (config[widget.name] !== undefined) {
            widget.value = config[widget.name];
        }
    }

    // Widgets now match the preset again — clear any "✎" markers.
    updatePresetDivergenceMarkers(node);
    node.setDirtyCanvas(true);
}

// Every tracked dataset (FRAMING/CAMERAS/LENSES/APERTURES/LIGHTING/STYLES/
// COLOR_GRADING) has "None" as its first, valid entry — switching to
// "None (Custom)" blanks all 7 style widgets back to it, same as selecting
// a named preset overwrites them with that preset's values.
function resetToCustomDefaults(node) {
    const widgets = node.widgets;
    if (!widgets) return;
    for (const widget of widgets) {
        if (TRACKED_FIELDS.includes(widget.name)) {
            widget.value = "None";
        }
    }
    updatePresetDivergenceMarkers(node);
    node.setDirtyCanvas(true);
}

function updatePresetDivergenceMarkers(node) {
    if (!node.widgets) return;
    const presetW = node.widgets.find(w => w.name === "style_preset");
    const presetVal = presetW ? presetW.value : "None (Custom)";
    const config = presetVal === "None (Custom)" ? null : getPresetConfig(presetVal);

    let changed = false;
    for (const w of node.widgets) {
        if (!w || !w.name || !TRACKED_FIELDS.includes(w.name)) continue;
        let marked = false;
        if (config && config[w.name] !== undefined) {
            marked = String(w.value) !== String(config[w.name]);
        }
        changed = _setLabelMarker(w, marked ? PRESET_MARKER : null) || changed;
    }
    if (changed) node.setDirtyCanvas(true, true);
}

// ALBABIT-FIX: action sits under base_prompt. Graphs saved before it hold
// positional widgets_values in this order; declaring it makes the frontend
// restore this node by name.
const PROMPT_SAVED_WIDGET_ORDER = [
    "base_prompt", "style_preset", "framing", "camera_type", "lens_focal", "aperture_dof", "lighting",
    "style_aesthetic", "color_grading", "negative_strength", "negative_prompt", "negative_mode",
    "resolution", "upsample_prompt", "upsampler_seed", "control_after_generate", "layout",
];

app.registerExtension({
    name: "FXTD.RadianceCinematicPromptEncoder",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== "RadianceCinematicPromptEncoder") return;
        nodeData.fallbackWidgetsValuesNames = PROMPT_SAVED_WIDGET_ORDER;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            if (onNodeCreated) onNodeCreated.apply(this, arguments);
            _holdHiddenSeed(this);
            _addTabsWidget(this);

            const self = this;
            const presetWidget = this.widgets?.find(w => w.name === "style_preset");
            if (!presetWidget) return;

            // ALBABIT-FIX: _radPreset is also set when a graph or a tab shows a preset:
            // picking the preset last picked elsewhere did nothing after either.
            this._radPreset = presetWidget.value;

            const originalCallback = presetWidget.callback;
            presetWidget.callback = (value) => {
                if (originalCallback) originalCallback.call(presetWidget, value);
                if (window.app && window.app.configuringGraph) return;
                if (value !== this._radPreset) {
                    if (value === "None (Custom)") {
                        resetToCustomDefaults(this);
                    } else {
                        applyPreset(this, value);
                    }
                }
                this._radPreset = value;
            };

            // Poll for manual widget edits, undo/redo — same state-based
            // approach as js/radiance_sampler.js's marker refresh. Also
            // covers the connected Loader's own preset/model_type changing
            // live, which _liveMiniMaxState reads directly.
            this._presetMarkerInterval = setInterval(() => _refreshLiveState(self), 250);
            const origOnRemoved = this.onRemoved;
            this.onRemoved = function () {
                if (self._presetMarkerInterval) {
                    clearInterval(self._presetMarkerInterval);
                    self._presetMarkerInterval = null;
                }
                if (origOnRemoved) origOnRemoved.apply(this, arguments);
            };
        };

        // Flag pre-existing divergence in a loaded workflow (e.g. a widget
        // edited manually before saving) — mirrors js/radiance_sampler.js.
        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function (info) {
            if (onConfigure) onConfigure.apply(this, arguments);
            this._radPreset = this.widgets?.find(w => w.name === "style_preset")?.value;
            const self = this;
            setTimeout(() => _refreshLiveState(self), 150);
            setTimeout(() => {
                _refreshLiveState(self);
                // ALBABIT-FIX: a graph saved before the tab row keeps a height without it,
                // and Nodes 2.0 grew the node to fit the row: the text boxes give the room.
                if (info.size && self.size[1] > info.size[1]) self.setSize([self.size[0], info.size[1]]);
            }, 600);
        };

        // ALBABIT-FIX: the last real run's arch verdicts, read by _refreshLiveState
        // for the negative's marker and Auto-Detect lock, and for action and text_to_video.
        const onExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            if (onExecuted) onExecuted.apply(this, arguments);
            this._radWeakNeg = !!message?.weak_neg_arch?.[0];
            this._radVideo = message?.video_arch?.[0];
            this._radTextToVideo = message?.text_to_video?.[0];
            _refreshLiveState(this);
            this.setDirtyCanvas?.(true, true);
        };
    }
});

console.log("[Radiance Cinematic Prompt Encoder] Extension loaded");
