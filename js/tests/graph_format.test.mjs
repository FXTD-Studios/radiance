// Opening a saved graph in the format it was stored in.
//
// The Project Manager node stored the API prompt in its .rad files, and the
// library opened every entry with app.loadGraphData, which only takes a UI
// workflow, so those entries never opened (code review P2-2).
//
// Run: node --test js/tests/graph_format.test.mjs
import test from "node:test";
import assert from "node:assert/strict";

import { isApiPromptGraph, openSavedGraph } from "../radiance_graph_format.js";
import defaultExport from "../radiance_graph_format.js";

const UI = { nodes: [{ id: 1, type: "PreviewImage" }], links: [] };
const API = {
    "1": { class_type: "CheckpointLoaderSimple", inputs: { ckpt_name: "a.safetensors" } },
    "2": { class_type: "PreviewImage", inputs: { images: ["1", 0] } },
};

function fakeApp({ withApiLoader = true } = {}) {
    const calls = [];
    const app = { loadGraphData: (...args) => calls.push(["loadGraphData", ...args]) };
    if (withApiLoader) app.loadApiJson = (...args) => calls.push(["loadApiJson", ...args]);
    return { app, calls };
}

test("tells the two formats apart", () => {
    assert.equal(isApiPromptGraph(API), true);
    assert.equal(isApiPromptGraph(UI), false);
    assert.equal(isApiPromptGraph({ nodes: [] }), false);
    assert.equal(isApiPromptGraph({}), false);
    assert.equal(isApiPromptGraph(null), false);
    assert.equal(isApiPromptGraph([API]), false);
    assert.equal(isApiPromptGraph({ "1": { class_type: "X" }, extra: { version: 0.4 } }), false);
});

test("a UI workflow opens and merges through loadGraphData as before", () => {
    const { app, calls } = fakeApp();
    openSavedGraph(app, UI, false, "sh010");
    openSavedGraph(app, UI, true, "sh010");
    assert.deepEqual(calls, [["loadGraphData", UI, false], ["loadGraphData", UI, true]]);
});

test("an API prompt opens through loadApiJson", () => {
    const { app, calls } = fakeApp();
    openSavedGraph(app, API, false, "sh010_ada_v0001.rad");
    assert.deepEqual(calls, [["loadApiJson", API, "sh010_ada_v0001.rad"]]);
});

test("an API prompt cannot be merged and says so", () => {
    const { app, calls } = fakeApp();
    assert.throws(() => openSavedGraph(app, API, true), /no layout to merge/);
    assert.deepEqual(calls, []);
});

test("an older ComfyUI without loadApiJson gets a clear error", () => {
    const { app } = fakeApp({ withApiLoader: false });
    assert.throws(() => openSavedGraph(app, API), /cannot open API-format/);
});

test("the default export carries both helpers", () => {
    assert.equal(defaultExport.isApiPromptGraph, isApiPromptGraph);
    assert.equal(defaultExport.openSavedGraph, openSavedGraph);
});
