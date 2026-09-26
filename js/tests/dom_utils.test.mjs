// escapeHtml is a security primitive. It existed in five copies with two
// different null behaviours before being consolidated here, and had no test
// at all — the whole js/ tree had none.
//
// Run: node --test js/tests/
import test from "node:test";
import assert from "node:assert/strict";

import { escapeHtml } from "../radiance_dom_utils.js";
import defaultExport from "../radiance_dom_utils.js";

test("escapes the five characters that matter", () => {
    assert.equal(escapeHtml("&"), "&amp;");
    assert.equal(escapeHtml("<"), "&lt;");
    assert.equal(escapeHtml(">"), "&gt;");
    assert.equal(escapeHtml('"'), "&quot;");
    assert.equal(escapeHtml("'"), "&#39;");
});

test("neutralises a script tag", () => {
    assert.equal(
        escapeHtml("<script>alert(1)</script>"),
        "&lt;script&gt;alert(1)&lt;/script&gt;",
    );
});

test("neutralises an attribute break-out in both quote styles", () => {
    assert.ok(!escapeHtml('" onerror="alert(1)').includes('"'));
    assert.ok(!escapeHtml("' onerror='alert(1)").includes("'"));
});

test("escapes the ampersand first, so entities are not double-decoded", () => {
    // If & were escaped last, "&lt;" would come back as "&lt;" and render as "<".
    assert.equal(escapeHtml("&lt;"), "&amp;lt;");
});

test("null and undefined become empty, not the words", () => {
    // Two of the five original copies rendered the literal text "null" and
    // "undefined" into the UI. This is the behaviour that was kept.
    assert.equal(escapeHtml(null), "");
    assert.equal(escapeHtml(undefined), "");
});

test("falsy values that are not nullish survive", () => {
    assert.equal(escapeHtml(0), "0");
    assert.equal(escapeHtml(false), "false");
    assert.equal(escapeHtml(""), "");
});

test("non-strings are coerced", () => {
    assert.equal(escapeHtml(42), "42");
    assert.equal(escapeHtml(["a", "b"]), "a,b");
});

test("plain text is returned unchanged", () => {
    assert.equal(escapeHtml("shot_01 v3 — final"), "shot_01 v3 — final");
});

test("escaping is idempotent in the sense that output is inert", () => {
    const once = escapeHtml("<img src=x onerror=alert(1)>");
    assert.ok(!once.includes("<"));
    assert.ok(!once.includes(">"));
});

test("the default export carries the same function", () => {
    assert.equal(defaultExport.escapeHtml, escapeHtml);
});
