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
vm.runInContext(source + 'globalThis.unused = _negativeUnusedDownstream;', context);

// A Prompt whose "negative" output feeds the given nodes, each built from a
// class and widget values; `model` chains upstream nodes on the model input.
function graphWith(targets, { negativeText = false } = {}) {
    const links = {};
    const nodes = {};
    let nextLink = 1;
    let nextNode = 2;
    const add = (spec) => {
        const id = nextNode++;
        const node = {
            id, comfyClass: spec.cls, mode: spec.mode ?? 0,
            widgets: Object.entries(spec.widgets ?? {}).map(([name, value]) => ({ name, value, hidden: spec.hidden?.includes(name) })),
            inputs: [{ name: 'cfg', link: spec.cfgLinked ? 99 : null }],
        };
        if (spec.model) {
            const upstream = add(spec.model);
            links[nextLink] = { origin_id: upstream.id, target_id: id };
            node.inputs.push({ name: 'model', link: nextLink++ });
        }
        nodes[id] = node;
        return node;
    };
    const negLinks = targets.map(spec => {
        const target = add(spec);
        links[nextLink] = { origin_id: 1, target_id: target.id };
        return nextLink++;
    });
    const graph = { links, getNodeById: id => nodes[id] };
    return {
        graph,
        outputs: [
            { name: 'negative', links: negLinks },
            { name: 'negative_text', links: negativeText ? [42] : [] },
        ],
    };
}

const ks = (cfg, extra = {}) => ({ cls: 'KSampler', widgets: { cfg, sampler_name: 'euler' }, ...extra });

test('every sampler at cfg 1 locks the negative', () => {
    assert.equal(context.unused(graphWith([ks(1)])), true);
    assert.equal(context.unused(graphWith([ks(1), { cls: 'KSamplerAdvanced', widgets: { cfg: 1.0, sampler_name: 'euler' } }])), true);
});

test('cfg 1 within float rounding still locks it, like comfy\'s math.isclose', () => {
    assert.equal(context.unused(graphWith([ks(1.0000000000000002)])), true);
});

test('any cfg other than 1 keeps it editable: below 1 the negative is mixed in', () => {
    assert.equal(context.unused(graphWith([ks(7)])), false);
    assert.equal(context.unused(graphWith([ks(0.8)])), false);
    assert.equal(context.unused(graphWith([ks(0)])), false);
    assert.equal(context.unused(graphWith([ks(1), ks(4.5)])), false);
});

test('a consumer it cannot read keeps it editable', () => {
    assert.equal(context.unused(graphWith([{ cls: 'ControlNetApplyAdvanced' }])), false);
    assert.equal(context.unused(graphWith([ks(1, { cfgLinked: true })])), false);
    assert.equal(context.unused(graphWith([])), false);
    assert.equal(context.unused(graphWith([ks(1, { mode: 4 })])), false);
});

test('cfg_pp samplers, LTX-AV audio_cfg and a SAG patch still read it at cfg 1', () => {
    assert.equal(context.unused(graphWith([{ cls: 'KSampler', widgets: { cfg: 1, sampler_name: 'euler_cfg_pp' } }])), false);
    const radiance = (audioCfg, hidden) => ({ cls: 'RadianceSamplerPro', widgets: { cfg: 1, sampler: 'euler', audio_cfg: audioCfg }, hidden });
    assert.equal(context.unused(graphWith([radiance(3, [])])), false);
    assert.equal(context.unused(graphWith([radiance(3, ['audio_cfg'])])), true);
    assert.equal(context.unused(graphWith([radiance(0, [])])), true);
    assert.equal(context.unused(graphWith([ks(1, { model: { cls: 'SelfAttentionGuidance', model: { cls: 'UNETLoader' } } })])), false);
    assert.equal(context.unused(graphWith([ks(1, { model: { cls: 'LoraLoaderModelOnly', model: { cls: 'UNETLoader' } } })])), true);
});

test('negative_text wired elsewhere keeps it editable', () => {
    assert.equal(context.unused(graphWith([ks(1)], { negativeText: true })), false);
});
