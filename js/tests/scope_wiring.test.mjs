/**
 * The scope panel's wiring.
 *
 * Source assertions, for the same reason as `probe_wiring.test.mjs`:
 * `radiance_viewer.js` pulls in ComfyUI's `app.js` and cannot be imported in
 * Node. What these hold is the set of connections that make the scopes
 * *unambiguous* — which is the whole of Step 4. A scope that silently reverts to
 * an unlabelled axis is back to being an untrusted instrument even though every
 * pixel it plots is still correct.
 *
 * Run: node --test js/tests/scope_wiring.test.mjs
 */
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const JS = join(dirname(fileURLToPath(import.meta.url)), '..');
const viewer = readFileSync(join(JS, 'radiance_viewer.js'), 'utf8');

function code(src) {
    return src.replace(/\/\*[\s\S]*?\*\//g, '').replace(/(^|\s)\/\/.*$/gm, '$1');
}

function methodBody(src, name) {
    const start = src.indexOf(`\n    ${name}(`);
    assert.notEqual(start, -1, `method ${name} not found`);
    const rest = src.slice(start + 1);
    const end = rest.search(/\n    \}\n/);
    assert.notEqual(end, -1, `could not find the end of ${name}`);
    return rest.slice(0, end);
}

test('the scales come from the shared module', () => {
    assert.match(viewer, /from "\.\/radiance_scope_units\.js"/,
        'the viewer must import the scope units module');
    for (const fn of ['_scopeTicks', '_scopeValue', '_scopeDescribe']) {
        assert.ok(viewer.includes(fn), `${fn} imported but unused — the panel is inventing its own scale`);
    }
});

test('the Reinhard nit ruler is gone', () => {
    // What it did: placed "203 nit", "1k nit" and "10k nit" lines using
    // v/(v+1) as a stand-in for the display transform. The pixels being plotted
    // had come through the actual ACES transform, so the labels sat wherever
    // that unrelated curve put them — "203 nit" landed at exactly half height
    // no matter what the viewer was doing. Nits now come from ST.2084 or HLG,
    // which are defined on the signal actually being measured.
    assert.doesNotMatch(code(viewer), /hdrRuler/,
        'the Reinhard-proxy HDR ruler is back');
    assert.doesNotMatch(code(viewer), /scopeHdrRuler/,
        'the HDR ruler toggle is back');
    assert.doesNotMatch(code(viewer), /const R = v => v \/ \(v \+ 1\)/,
        'a Reinhard display proxy has reappeared in the scopes');
});

test('the waveform, parade and histogram all draw a labelled graticule', () => {
    for (const m of ['_drawScopeParade', '_drawScopeWaveform']) {
        assert.match(code(methodBody(viewer, m)), /_drawScopeGraticule\(/,
            `${m} draws its own guides instead of the shared graticule`);
    }
    assert.match(code(methodBody(viewer, '_drawScopeHistogram')), /_scopeTicks\(/,
        'the histogram axis is not driven by the scale');
});

test('no scope draws an unlabelled tick', () => {
    const grat = code(methodBody(viewer, '_drawScopeGraticule'));
    assert.match(grat, /fillText\(t\.label/, 'the graticule draws lines without their numbers');
    assert.match(grat, /fillText\(unit/, 'the graticule never states its unit');
});

test('the old contradictory caption is gone', () => {
    // The panel used to say "Linear · 0–255" above a graticule labelled
    // 0/25/50/75/100. Two units, one panel, neither stated.
    assert.doesNotMatch(code(viewer), /Linear · 0–255/, 'the contradictory caption is back');
    assert.doesNotMatch(code(methodBody(viewer, '_drawScopeHistogram')), /'255'/,
        'the histogram is labelling an 8-bit axis again');
});

test('the panel states what it is measuring, every time it renders', () => {
    const tab = code(methodBody(viewer, 'renderScopesTab'));
    assert.match(tab, /_scopeDescribe\(this\.scopeScale/,
        'the scopes tab never asks what it is measuring');
    for (const part of ['desc.label', 'desc.levels', 'desc.detail', 'desc.measuredAt']) {
        assert.ok(tab.includes(part), `the caption omits ${part}`);
    }
    assert.ok(tab.includes('desc.warn'),
        'the caption drops the warning for a nit scale on untransformed pixels');
    // The scopes read an 8-bit canvas. Offering a 10-bit scale over that without
    // saying so implies four times the precision the signal carries.
    assert.match(tab, /Sampled at 8 bits/,
        'the caption must declare the sampling depth alongside the scale');
});

test('the measurement point actually changes where pixels are sampled', () => {
    // A toggle that relabels the caption without changing the sample is worse
    // than no toggle: it makes a false claim with a control attached.
    const tab = methodBody(viewer, 'renderScopesTab');
    assert.match(tab, /this\.scopeTransformed && canUseGL/,
        'the sample source must depend on the measurement-point toggle');
    assert.match(tab, /_scopeMeasuredTransformed = this\.scopeTransformed && canUseGL/,
        'the caption must reflect what was actually sampled, not what was asked for');
});

test('the caption reports what was sampled, not what was requested', () => {
    // With no GL canvas there is nothing to measure "after the transforms", so
    // the toggle silently falls back. The caption has to fall back with it.
    assert.match(methodBody(viewer, '_scopeCtx'), /transformed: this\._scopeMeasuredTransformed/,
        '_scopeCtx must read the sampled flag, not this.scopeTransformed');
});

test('log assist reshapes the graticule rather than warning about it', () => {
    // Both the lines and the hover readout go through the same curve, so the
    // labels stay on their traces instead of the panel apologising for them.
    assert.match(code(methodBody(viewer, '_drawScopeGraticule')), /_scopePlotPos\(t\.at, logView\)/,
        'the graticule ignores log assist and its labels will slide off');
    assert.match(methodBody(viewer, '_scopePlotPos'), /_logAssistPos/);
    assert.match(methodBody(viewer, '_scopePlotInv'), /_logAssistInv/);
});

test('every scale control persists', () => {
    // Losing the scale on every reload would make it a novelty rather than a
    // setting.
    for (const key of [
        'radiance_scope_scale', 'radiance_scope_levels',
        'radiance_scope_xform', 'radiance_scope_hlg_peak',
    ]) {
        assert.ok(viewer.includes(`localStorage.setItem('${key}'`), `${key} is never written`);
        assert.ok(viewer.includes(`localStorage.getItem('${key}')`), `${key} is never read back`);
    }
});

test('the measurement-point toggle defaults to on', () => {
    // It is what the scopes did before this change; flipping the default would
    // silently alter every existing user's scopes.
    assert.match(viewer, /this\.scopeTransformed = localStorage\.getItem\('radiance_scope_xform'\) !== '0'/,
        'the default must be "include viewer transforms", matching prior behaviour');
});

test('the default scale is 10-bit code value', () => {
    assert.match(viewer, /radiance_scope_scale'\) \|\| 'cv10'/);
});

test('IRE appears nowhere in the scope panel', () => {
    assert.doesNotMatch(code(methodBody(viewer, 'renderScopesTab')), /\bIRE\b/,
        'IRE is a deliberate non-goal');
    // And nowhere else in the scopes either: the false-colour ramp used to be
    // called an "IRE scale bar" while carrying no values at all.
    for (const m of ['_drawScopeFalseColor', '_drawScopeParade', '_drawScopeWaveform', '_drawScopeHistogram']) {
        assert.doesNotMatch(code(methodBody(viewer, m)), /\bIRE\b/i, `${m} still refers to IRE`);
    }
});

test('the HLG peak selector only appears for the HLG scale', () => {
    // A nominal-peak control next to a PQ scale would imply PQ has one.
    assert.match(methodBody(viewer, 'renderScopesTab'), /this\.scopeScale === 'nits-hlg'/,
        'the HLG peak selector is not gated on the HLG scale');
});

test('the hover readout resolves through the same scale it draws', () => {
    const tab = code(methodBody(viewer, 'renderScopesTab'));
    assert.match(tab, /_scopePlotInv\(p, this\.scopeLogView\)/,
        'the hover readout must invert the plot curve');
    assert.match(tab, /_scopeValue\(norm, this\.scopeScale, this\._scopeCtx\(\)\)/,
        'the hover readout must use the selected scale');
});

test('the hover readout is only offered on scopes that have a scale', () => {
    // A vectorscope has no luminance axis; a value readout on it would be
    // meaningless.
    const tab = methodBody(viewer, 'renderScopesTab');
    assert.match(tab, /scopeMode === 'histogram'\) \? 'x'/);
    assert.match(tab, /const axis = /);
    assert.match(tab, /if \(axis\) \{/, 'the readout must be skipped when there is no axis');
});
