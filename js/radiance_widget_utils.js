// ◎ Radiance — shared LiteGraph widget helpers.
//
// These lived in six copies: radiance_io.js, radiance_loader.js,
// radiance_resolution.js, radiance_sampler.js, radiance_upscale.js and
// radiance_vae_widgets.js.
//
// _forceWidgetReinsert was byte-identical in all six. setWidgetVisible had
// drifted into four variants that differed on exactly two axes:
//
//   1. The fallback widget type when restoring a widget whose original type was
//      never recorded — "number", "combo", "INT" and "text" respectively. That
//      is a per-module detail, so it is now an option rather than a fork.
//
//   2. Whether the hidden state also suppressed 'draw' and the DOM nodes
//      ('inputEl', 'element'). radiance_loader.js and radiance_upscale.js did;
//      radiance_io.js and radiance_resolution.js did not, so DOM-backed widgets
//      in those two modules kept painting after being "hidden". Every guard is
//      conditional, so applying it everywhere is a strict superset of what the
//      four variants did.
//
// The implementation below is the union of all four. Nothing that worked before
// stops working; two of the four call sites gain the DOM handling they were
// missing.

/**
 * Remove and re-insert a widget at the same index.
 *
 * ALBABIT-FIX (preserved from the originals): always force a remove+reinsert,
 * even when type/hidden did not change on this call. Once a widget's Vue
 * component has been (re)mounted it stops reacting to later type/hidden changes
 * via a no-op splice(0,0) alone — it keeps rendering its previous state until
 * reinserted. Reinserting unconditionally guarantees every widget's component
 * reflects its current state regardless of how many times it toggled before.
 */
export function forceWidgetReinsert(widget, node) {
    if (!node?.widgets) return;
    const idx = node.widgets.indexOf(widget);
    if (idx === -1) return;
    node.widgets.splice(idx, 1);
    node.widgets.splice(idx, 0, widget);
}

/** Find a widget by name, or null. */
export function getWidget(node, name) {
    return node?.widgets?.find(w => w.name === name) ?? null;
}

// LiteGraph.isValidConnection without the LiteGraph global: equal types, a
// wildcard, or a shared entry in comma-separated type lists.
function typesMatch(a, b) {
    if (a === b || a === "*" || b === "*" || !a || !b) return true;
    const list = t => String(t).toLowerCase().split(",");
    return list(a).some(t => list(b).includes(t));
}

// The input a bypassed node forwards to output 'slot', as ComfyUI's
// ExecutableNodeDTO._getBypassSlotIndex picks it: the input at the same index
// if its type fits, else the first exact type match, else the first that fits.
function bypassInput(node, slot, type) {
    const inputs = node.inputs ?? [];
    const outputType = node.outputs?.[slot]?.type ?? type;
    if (type === "*" || type === "") return inputs[slot] ?? inputs[0];
    const same = inputs[slot];
    if (same && typesMatch(same.type, outputType) && typesMatch(same.type, type)) return same;
    return inputs.find(i => i.type === type)
        ?? inputs.find(i => typesMatch(i.type, outputType) && typesMatch(i.type, type));
}

// ALBABIT-FIX: the node that actually feeds 'input', or null. A muted node
// (mode 2) sends nothing; a bypassed one (mode 4) forwards one of its inputs,
// chosen as ComfyUI does (bypassInput). node.graph, so it works inside subgraphs.
export function liveSourceNode(node, input, depth = 0) {
    const graph = node?.graph;
    const link = input?.link != null ? graph?.links?.[input.link] : null;
    const origin = link && graph.getNodeById(link.origin_id);
    if (!origin || origin.mode === 2 || depth > 32) return null;
    if (origin.mode !== 4) return origin;
    return liveSourceNode(origin, bypassInput(origin, link.origin_slot, link.type), depth + 1);
}

// ALBABIT-FIX: the Loader (or other node) that feeds 'model_meta', or null.
// Shared by radiance_prompt.js and radiance_sampler.js.
export function modelMetaSourceNode(node) {
    return liveSourceNode(node, node?.inputs?.find(i => i.name === "model_meta"));
}

/** True when 'input' is fed by a node that runs (see liveSourceNode). */
export function isInputLive(node, input) {
    return liveSourceNode(node, input) !== null;
}

// ALBABIT-FIX: fit a node to its visible widgets plus the height the user
// dragged in. Showing or hiding a widget used to snap it back to its minimum
// height, on every reload too. The added height lives in node.properties, so
// it is saved with the workflow.
export function fitNodeSize(node) {
    if (!node?.computeSize) return;
    trackUserHeight(node);
    const [minWidth, minHeight] = node.computeSize();
    const width = Math.max(node.size[0], minWidth);
    const height = minHeight + (node.properties?.radExtraHeight ?? 0);
    if (node.size[0] === width && node.size[1] === height) return;
    node._radFitting = true;
    node.setSize([width, height]);
    node._radFitting = false;
    node.setDirtyCanvas?.(true, true);
}

// A resize by the user reaches onResize, on the canvas and in the Vue nodes
// alike; the ones fitNodeSize and a graph load make are not the user's.
function trackUserHeight(node) {
    if (node._radTracksHeight) return;
    node._radTracksHeight = true;
    const onResize = node.onResize;
    node.onResize = function (size) {
        const result = onResize?.apply(this, arguments);
        if (!this._radFitting && !globalThis.app?.configuringGraph) {
            (this.properties ??= {}).radExtraHeight = Math.max(0, size[1] - this.computeSize()[1]);
        }
        return result;
    };
}

/**
 * Show or hide a widget, collapsing its row when hidden.
 *
 * @param {object} widget
 * @param {boolean} visible
 * @param {object} node
 * @param {{fallbackType?: string}} [options]
 *        fallbackType is used only when restoring a widget whose original type
 *        was never captured — i.e. one that was already "hidden" the first time
 *        this ran. Pass the type that module's widgets actually are.
 */
export function setWidgetVisible(widget, visible, node, options = {}) {
    if (!widget) return false;

    // MERGE-PORT (beta/main 9a5ac88): remember the prior state so the Vue
    // remount below only happens on a real transition, and so callers can
    // batch layout work behind a boolean.
    const wasHidden = widget.hidden === true || widget.type === "hidden";

    const fallbackType = options.fallbackType ?? "text";

    if (!widget.options) widget.options = {};
    widget.options.hidden = !visible;
    widget.hidden = !visible;

    if (visible) {
        if (widget.type === "hidden") {
            widget.type = widget._origType || fallbackType;

            // Restore the saved computeSize when there was one, otherwise delete
            // the override so LiteGraph's prototype recalculates. A fallback
            // closure here gave wrong heights for toggles and combos.
            if (widget._origComputeSize !== undefined) {
                widget.computeSize = widget._origComputeSize;
            } else {
                delete widget.computeSize;
            }
            delete widget._origComputeSize;

            if (widget._origDraw !== undefined) {
                widget.draw = widget._origDraw;
                delete widget._origDraw;
            } else {
                delete widget.draw;
            }

            if (widget.inputEl) widget.inputEl.style.display = "";
            if (widget.element) widget.element.style.display = "";

            // Nodes 2.0 Vue layout reads computedHeight for the row CSS.
            if (widget._origComputedHeight !== undefined) {
                widget.computedHeight = widget._origComputedHeight;
                delete widget._origComputedHeight;
            } else {
                widget.computedHeight = 32;
            }
        }
    } else {
        if (widget.type !== "hidden") {
            widget._origType = widget.type;
            widget._origComputeSize = widget.computeSize;
            widget._origComputedHeight = widget.computedHeight;

            widget.type = "hidden";
            widget.computeSize = () => [0, -4];
            if (widget.draw) widget._origDraw = widget.draw;
            widget.draw = function () { };

            if (widget.inputEl) widget.inputEl.style.display = "none";
            if (widget.element) widget.element.style.display = "none";

            // 4, not -4: Vue uses computedHeight for CSS and 4px collapses the row.
            widget.computedHeight = 4;
        }
    }

    // MERGE-PORT (beta/main 9a5ac88): reinsert ONLY on an actual
    // hidden/visible transition. The old unconditional reinsert destroyed and
    // recreated every widget's Vue component on each call — running on the
    // 250 ms polls, that interrupted in-progress typing in neighbouring
    // widgets. Returns true when the visibility actually changed so callers
    // can skip node-resize work on no-op calls.
    if (wasHidden !== !visible) {
        forceWidgetReinsert(widget, node);
        return true;
    }
    return false;
}

export default { fitNodeSize, forceWidgetReinsert, getWidget, isInputLive, liveSourceNode, setWidgetVisible };
