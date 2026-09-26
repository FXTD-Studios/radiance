/**
 * Every shipped module must parse as an ES module.
 *
 * This exists because `node --check foo.js` does not check what you think it
 * checks. On a `.js` file in a package with no `"type": "module"`, it does not
 * apply ES-module parsing — and `radiance_webgl.js` passed it while being
 * unparseable in a browser. The result was the Viewer node rendering with no UI
 * at all: three inputs, one output, nothing else. Every `node --check` run
 * during the session that introduced it reported success.
 *
 * The specific defect was a **backtick inside a comment inside a GLSL template
 * literal**. The shaders are template strings, so a backtick in a comment ends
 * the string, and everything after it is parsed as JavaScript:
 *
 *     // an HDR colourist actually works to. `v` arrives in the internal
 *                                            ^ ends the shader string
 *
 * Nothing else in the file changed. The module simply stopped existing, so
 * ComfyUI's extension registration never ran, so the node had no widgets.
 *
 * Two assertions below: every file parses as a module, and no comment anywhere
 * carries a backtick. The second is broader than strictly necessary — plenty of
 * comments sit outside template literals — but the cost of the rule is zero and
 * the cost of getting it wrong is the whole front end.
 *
 * Run: node --test js/tests/module_parse.test.mjs
 */
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync, writeFileSync, readdirSync, mkdtempSync, rmSync } from 'node:fs';
import { execFileSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
import { tmpdir } from 'node:os';

const JS = join(dirname(fileURLToPath(import.meta.url)), '..');

/** Every module Radiance ships to the browser. */
function shippedModules() {
    const top = readdirSync(JS)
        .filter((f) => f.endsWith('.js'))
        .map((f) => join(JS, f));
    // The vendored OCIO loader is patched, so it is ours to keep parsing.
    top.push(join(JS, 'vendor', 'ocio', 'index.js'));
    return top;
}

test('every shipped module parses as an ES module', () => {
    // Copied to .mjs first. That is the whole point: the extension decides
    // which grammar Node applies, and `.js` here does not get module parsing.
    const dir = mkdtempSync(join(tmpdir(), 'radiance-parse-'));
    try {
        for (const file of shippedModules()) {
            const copy = join(dir, 'check.mjs');
            writeFileSync(copy, readFileSync(file));
            try {
                execFileSync(process.execPath, ['--check', copy], { stdio: 'pipe' });
            } catch (e) {
                const out = String(e.stderr || e.stdout || e.message)
                    .replace(new RegExp(copy.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'), 'g'), file);
                assert.fail(`${file} does not parse as an ES module:\n${out}`);
            }
        }
    } finally {
        rmSync(dir, { recursive: true, force: true });
    }
});

test('no comment contains a backtick', () => {
    // A backtick in a comment inside a shader template literal terminates the
    // string. The failure is silent at every checkpoint short of loading the
    // page: the file looks fine, greps fine, and `node --check` on the .js
    // passes.
    const offenders = [];
    for (const file of shippedModules()) {
        readFileSync(file, 'utf8').split('\n').forEach((line, i) => {
            const t = line.trim();
            if (t.startsWith('//') && t.includes('`')) {
                offenders.push(`${file}:${i + 1}  ${t.slice(0, 80)}`);
            }
        });
    }
    assert.deepEqual(offenders, [],
        'backticks in comments — use quotes instead:\n' + offenders.join('\n'));
});

test('the viewer\'s entry point still declares the extension imports', () => {
    // A cheap canary for the failure this file is named after: when the module
    // stopped parsing, nothing about the file's *content* looked wrong. If the
    // ComfyUI imports ever vanish, the node loses its UI the same way.
    const viewer = readFileSync(join(JS, 'radiance_viewer.js'), 'utf8');
    assert.match(viewer, /import \{ app \} from "\.\.\/\.\.\/scripts\/app\.js"/);
    assert.match(viewer, /import \{ api \} from "\.\.\/\.\.\/scripts\/api\.js"/);
});
