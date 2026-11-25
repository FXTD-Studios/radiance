import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import vm from 'node:vm';

const source = readFileSync(new URL('../radiance_prompt.js', import.meta.url), 'utf8')
    .replace(/import\s+[\s\S]*?from\s+["'][^"']+["'];\s*/g, '');

const context = vm.createContext({
    app: { registerExtension() {} },
    api: { fetchApi: () => new Promise(() => {}) },
    console: { log() {}, warn() {} },
});
vm.runInContext(source + 'globalThis.plan = _textPlan; globalThis.shares = _textShares;', context);

const NAMES = ['base_prompt', 'negative_prompt'];
const plain = (value) => JSON.parse(JSON.stringify(value));

test('the first box left automatic takes the free space', () => {
    assert.deepEqual(plain(context.plan(NAMES, undefined)).map(p => p.fill), [true, false]);
    assert.deepEqual(plain(context.plan(NAMES, { base_prompt: 300 })).map(p => [p.height, p.fill]), [[300, false], [null, true]]);
});

test('when every box was set by hand, the last one still takes the free space', () => {
    const boxes = plain(context.plan(NAMES, { base_prompt: 300, negative_prompt: 200 }));
    assert.deepEqual(boxes.map(p => [p.height, p.fill]), [[300, false], [null, true]]);
});

test('a short negative keeps the height of its text, the prompt takes the rest', () => {
    assert.deepEqual(plain(context.shares(context.plan(NAMES), [688, 76], 894)), [818, 76]);
});

test('long boxes share what is left equally, a shorter one keeps its text', () => {
    assert.deepEqual(plain(context.shares(context.plan(NAMES), [1300, 1100], 894)), [447, 447]);
    assert.deepEqual(plain(context.shares(context.plan(NAMES), [688, 403], 894)), [491, 403]);
});

test('a box set by hand keeps its height, the next automatic one takes the rest', () => {
    assert.deepEqual(plain(context.shares(context.plan(NAMES, { base_prompt: 191 }), [688, 403], 894)), [191, 703]);
});

test('no box goes under the textarea minimum', () => {
    assert.deepEqual(plain(context.shares(context.plan(NAMES), [20, 10], 894)), [830, 64]);
});
