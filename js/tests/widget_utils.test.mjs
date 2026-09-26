// setWidgetVisible had four divergent copies before consolidation, and the
// shared version carries two fixes that are easy to undo by accident:
//
//   * it reinserts the widget ONLY on a real visibility transition — the old
//     unconditional reinsert recreated every widget's Vue component on each
//     call, and running on 250 ms polls it interrupted typing in neighbouring
//     widgets;
//   * restoring a widget deletes the computeSize override rather than
//     substituting a closure, because a fallback closure gave wrong row
//     heights for toggles and combos.
//
// LiteGraph widgets are plain objects, so these need fakes rather than a
// browser.
//
// Run: node --test js/tests/
import test from "node:test";
import assert from "node:assert/strict";

import {
    forceWidgetReinsert,
    getWidget,
    setWidgetVisible,
} from "../radiance_widget_utils.js";

const makeWidget = (name, type = "number") => ({
    name,
    type,
    computeSize: () => [100, 20],
    options: {},
});

const makeNode = (...widgets) => ({ widgets: [...widgets] });

// ── getWidget ───────────────────────────────────────────────────────────────

test("getWidget finds by name", () => {
    const w = makeWidget("steps");
    assert.equal(getWidget(makeNode(makeWidget("cfg"), w), "steps"), w);
});

test("getWidget returns null rather than throwing on a missing name", () => {
    assert.equal(getWidget(makeNode(makeWidget("cfg")), "nope"), null);
});

test("getWidget tolerates a node with no widgets, and no node at all", () => {
    assert.equal(getWidget({}, "x"), null);
    assert.equal(getWidget(null, "x"), null);
    assert.equal(getWidget(undefined, "x"), null);
});

// ── forceWidgetReinsert ─────────────────────────────────────────────────────

test("forceWidgetReinsert keeps the widget at its original index", () => {
    const [a, b, c] = [makeWidget("a"), makeWidget("b"), makeWidget("c")];
    const node = makeNode(a, b, c);
    forceWidgetReinsert(b, node);
    assert.deepEqual(node.widgets, [a, b, c]);
});

test("forceWidgetReinsert ignores a widget that is not on the node", () => {
    const node = makeNode(makeWidget("a"));
    const before = [...node.widgets];
    forceWidgetReinsert(makeWidget("orphan"), node);
    assert.deepEqual(node.widgets, before);
});

test("forceWidgetReinsert tolerates a missing node", () => {
    assert.doesNotThrow(() => forceWidgetReinsert(makeWidget("a"), null));
});

// ── setWidgetVisible ────────────────────────────────────────────────────────

test("hiding collapses the row and records the original type", () => {
    const w = makeWidget("steps", "number");
    setWidgetVisible(w, false, makeNode(w));

    assert.equal(w.type, "hidden");
    assert.equal(w.hidden, true);
    assert.equal(w.options.hidden, true);
    assert.equal(w._origType, "number");
    assert.deepEqual(w.computeSize(), [0, -4]);
    assert.equal(w.computedHeight, 4, "Vue reads computedHeight for the row CSS");
});

test("showing restores the original type", () => {
    const w = makeWidget("steps", "combo");
    const node = makeNode(w);
    setWidgetVisible(w, false, node);
    setWidgetVisible(w, true, node);

    assert.equal(w.type, "combo");
    assert.equal(w.hidden, false);
    assert.equal(w.computedHeight, 32);
});

test("restoring deletes the computeSize override instead of faking one", () => {
    // A fallback closure here gave wrong heights for toggles and combos.
    const w = makeWidget("t", "toggle");
    delete w.computeSize;                       // never had one of its own
    const node = makeNode(w);

    setWidgetVisible(w, false, node);
    setWidgetVisible(w, true, node);

    assert.equal(
        Object.prototype.hasOwnProperty.call(w, "computeSize"), false,
        "the override must be removed so LiteGraph's prototype recalculates",
    );
});

test("a widget with its own computeSize gets exactly that back", () => {
    const own = () => [123, 45];
    const w = makeWidget("x");
    w.computeSize = own;
    const node = makeNode(w);

    setWidgetVisible(w, false, node);
    setWidgetVisible(w, true, node);

    assert.equal(w.computeSize, own);
});

test("fallbackType is used only when the original type was never captured", () => {
    const w = { name: "x", type: "hidden", options: {} };   // hidden from birth
    setWidgetVisible(w, true, makeNode(w), { fallbackType: "combo" });
    assert.equal(w.type, "combo");
});

test("fallbackType defaults to text", () => {
    const w = { name: "x", type: "hidden", options: {} };
    setWidgetVisible(w, true, makeNode(w));
    assert.equal(w.type, "text");
});

test("it returns true on a real transition and false on a no-op", () => {
    const w = makeWidget("x");
    const node = makeNode(w);

    assert.equal(setWidgetVisible(w, false, node), true, "first hide is a transition");
    assert.equal(setWidgetVisible(w, false, node), false, "hiding twice changes nothing");
    assert.equal(setWidgetVisible(w, true, node), true, "unhiding is a transition");
    assert.equal(setWidgetVisible(w, true, node), false, "showing twice changes nothing");
});

test("a no-op call does not reinsert the widget", () => {
    // The regression this guards: an unconditional reinsert destroyed and
    // recreated the Vue component on every 250 ms poll, interrupting typing.
    const w = makeWidget("x");
    const node = makeNode(w);
    setWidgetVisible(w, false, node);

    let spliced = 0;
    const realSplice = node.widgets.splice.bind(node.widgets);
    node.widgets.splice = (...args) => { spliced++; return realSplice(...args); };

    setWidgetVisible(w, false, node);
    assert.equal(spliced, 0, "a no-op call reinserted the widget anyway");
});

test("DOM elements are hidden and restored when present", () => {
    const w = makeWidget("x");
    w.inputEl = { style: { display: "" } };
    w.element = { style: { display: "" } };
    const node = makeNode(w);

    setWidgetVisible(w, false, node);
    assert.equal(w.inputEl.style.display, "none");
    assert.equal(w.element.style.display, "none");

    setWidgetVisible(w, true, node);
    assert.equal(w.inputEl.style.display, "");
    assert.equal(w.element.style.display, "");
});

test("a custom draw is stashed and restored", () => {
    const draw = () => {};
    const w = makeWidget("x");
    w.draw = draw;
    const node = makeNode(w);

    setWidgetVisible(w, false, node);
    assert.notEqual(w.draw, draw, "draw should be neutralised while hidden");

    setWidgetVisible(w, true, node);
    assert.equal(w.draw, draw);
});

test("a null widget is refused rather than throwing", () => {
    assert.equal(setWidgetVisible(null, true, makeNode()), false);
});
