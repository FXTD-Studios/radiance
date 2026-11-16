import { test } from "node:test";
import assert from "node:assert/strict";
import * as M from "../studio_model.mjs";

test("screenFromHash maps old dashboard names and ignores cache-busting", () => {
    assert.equal(M.screenFromHash("#shots?t=123"), "shots");
    assert.equal(M.screenFromHash("#library"), "workflows");
    assert.equal(M.screenFromHash("#manager"), "shots");
    assert.equal(M.screenFromHash(""), "home");
    assert.equal(M.screenFromHash("#nope"), "home");
});

test("greeting follows the clock", () => {
    assert.equal(M.greeting(new Date(2026, 0, 1, 9)), "Good morning");
    assert.equal(M.greeting(new Date(2026, 0, 1, 14)), "Good afternoon");
    assert.equal(M.greeting(new Date(2026, 0, 1, 21)), "Good evening");
});

const wf = (filename, kicker, extra = {}) => ({ filename, mtime: 1, size: 10, metadata: { stats: { kicker, title: filename, tags: ["A"] }, ...extra } });

test("workflow cards: official first, Starter leads, groups from kicker and path", () => {
    const cards = M.workflowCards([
        wf("mine/SH010_grade_v001.rad", ""),
        wf("official/API/api_veo3_hdr.rad", "API · Video"),
        wf("official/Local/hdr_delivery.rad", "Delivery"),
        wf("official/Local/start.rad", "Start here"),
        wf("official/Local/sdr_to_hdr.rad", "HDR"),
        wf("official/Local/ltx25_hdr.rad", "Generate · Video"),
    ]);
    assert.equal(cards[0].filename, "official/Local/start.rad");
    assert.equal(cards[1].filename, "official/Local/sdr_to_hdr.rad");
    assert.equal(cards.at(-1).filename, "mine/SH010_grade_v001.rad");
    const by = Object.fromEntries(cards.map((c) => [c.filename, c.groups]));
    assert.deepEqual(by["mine/SH010_grade_v001.rad"], ["mine"]);
    assert.ok(by["official/API/api_veo3_hdr.rad"].includes("api"));
    assert.ok(by["official/API/api_veo3_hdr.rad"].includes("video"));
    assert.ok(by["official/Local/hdr_delivery.rad"].includes("deliver"));
    assert.ok(by["official/Local/sdr_to_hdr.rad"].includes("hdr"));
    assert.ok(by["official/Local/ltx25_hdr.rad"].includes("generate"));
    const counts = M.workflowCounts(cards);
    assert.equal(counts.all, 6);
    assert.equal(counts.mine, 1);
    assert.equal(M.filterWorkflows(cards, "all", "veo3").length, 1);
    assert.equal(M.filterWorkflows(cards, "deliver").length, 1);
});

test("assets filter by type, HDR and text", () => {
    const assets = [
        { id: 1, name: "a.exr", type: "hdri", format: "EXR", path: "/o/a.exr" },
        { id: 2, name: "b.png", type: "image", format: "PNG", path: "/o/b.png" },
        { id: 3, name: "c", type: "sequence", format: "EXR", frames: 24, path: "/o/c_0001.exr" },
        { id: 4, name: "d.mp4", type: "video", format: "MP4", path: "/o/d.mp4" },
    ];
    assert.deepEqual(M.filterAssets(assets, "hdr").map((a) => a.id), [1, 3]);
    assert.deepEqual(M.filterAssets(assets, "image").map((a) => a.id), [1, 2]);
    assert.deepEqual(M.filterAssets(assets, "sequence").map((a) => a.id), [3]);
    assert.deepEqual(M.filterAssets(assets, "all", "mp4").map((a) => a.id), [4]);
    assert.equal(M.assetKind(assets[2]), "EXR sequence · 24 frames");
});

test("shot rows group versions, keep the real status and match outputs and notes", () => {
    const rows = M.shotRows([
        { shot: "SH020", version: "v002", status: "Review", workflow: "SH020_fill_v002", filename: "P/SH020_fill_v002.rad", updated: "today" },
        { shot: "SH020", version: "v003", status: "Saved from Project Manager dashboard", workflow: "SH020_fill_v003", filename: "P/SH020_fill_v003.rad", updated: "now" },
        { shot: "SH010", version: "v004", status: "Approved", workflow: "SH010_grade_v004", filename: "P/SH010.rad", updated: "2 hours ago" },
        { shot: "GENERAL", version: "v001", status: "WIP" },
    ], [{ name: "SH020_fill_v003.exr", path: "/o/x.exr" }, { name: "other.png", path: "/o/y.png" }], ["SH020: check the glass", "General note"]);
    assert.deepEqual(rows.map((r) => r.shot), ["SH010", "SH020"]);
    const sh20 = rows[1];
    assert.equal(sh20.version, "v003");
    assert.equal(sh20.status.key, "Review");
    assert.equal(sh20.status.label, "In review");
    assert.equal(sh20.versions.length, 2);
    assert.equal(sh20.outputs.length, 1);
    assert.deepEqual(sh20.notes, ["SH020: check the glass"]);
    assert.deepEqual(M.shotStats(rows), { total: 2, wip: 0, review: 1, approved: 1, retake: 0 });
    assert.equal(M.filterShots(rows, "Review").length, 1);
    assert.equal(M.filterShots(rows, "all", "sh010").length, 1);
});

test("render rows read ComfyUI history, newest first, with errors", () => {
    const rows = M.renderRows({
        a: { prompt: [1], status: { status_str: "success", messages: [["execution_start", { timestamp: 1000 }], ["execution_success", { timestamp: 4500 }]] },
            outputs: { 9: { images: [{ filename: "R_00001_.png", type: "output", subfolder: "" }] } } },
        b: { prompt: [2], status: { status_str: "error", messages: [["execution_start", { timestamp: 9000 }], ["execution_error", { timestamp: 9100, node_type: "RadianceWrite", exception_message: "disk full\nmore" }]] }, outputs: {} },
    });
    assert.equal(rows[0].id, "b");
    assert.equal(rows[0].failed, true);
    assert.equal(rows[0].error, "RadianceWrite: disk full");
    assert.equal(rows[1].seconds, 3.5);
    assert.equal(M.viewUrl(rows[1].images[0]), "/view?filename=R_00001_.png&type=output&subfolder=");
});

test("needs-you only lists real things", () => {
    const shots = M.shotRows([{ shot: "SH020", version: "v003", status: "Review", workflow: "w" }, { shot: "SH040", version: "v001", status: "Retake", workflow: "w" }]);
    const items = M.needsYou({ shots, renders: [{ failed: true, error: "X: boom" }] });
    assert.deepEqual(items.map((i) => i.tone), ["info", "bad", "bad"]);
    assert.deepEqual(M.needsYou({}), []);
});

test("gpu summary from /system_stats", () => {
    const g = M.gpuSummary({ devices: [{ name: "cuda:0 NVIDIA GeForce RTX 4090 : cudaMallocAsync", vram_total: 24 * 2 ** 30, vram_free: 18 * 2 ** 30 }] });
    assert.equal(g.name, "NVIDIA GeForce RTX 4090");
    assert.equal(g.percent, 25);
    assert.equal(g.label, "6.0 GB / 24 GB");
    assert.equal(M.gpuSummary({}), null);
});
