/**
 * The WGSL half of the grade, compiled and run.
 *
 * `js/radiance_grade.js` emits GLSL and WGSL from the same file as its JS
 * functions. `gradeharness.html` proves the GLSL agrees with the JS in a real
 * WebGL2 context; this proves the same thing for the WGSL, in a real WebGPU
 * device, over the identical case matrix.
 *
 * It runs under Deno rather than Node because Deno ships WebGPU and Node does
 * not, and because headless Chromium has no navigator.gpu at all -- checked
 * across every documented flag combination, including --enable-unsafe-webgpu
 * with SwiftShader and with Vulkan. The adapter here comes from Mesa's lavapipe
 * software Vulkan driver, so it needs no GPU and runs on an ordinary CI box.
 *
 * Output is one JSON object on stdout. Driven by grade_wgsl_gpu.test.mjs.
 */
import { WGSL, gradePixel } from '../radiance_grade.js';

// The same pixels and grades gradeharness.html uses. They have to be the same
// list: the value of this file is that both dialects meet the same bar, and a
// separate matrix would let the WGSL pass on easier cases.
const PIXELS = [[0,0,0], [0.18,0.18,0.18], [1,1,1], [4,2,0.5], [-0.3,0.2,0.9], [0.05,0.5,12]];
const GRADES = [
    {},
    { lift: [0.1, 0.05, 0], gain: [1.2, 1, 0.9] },
    { gamma: [2.2, 2.2, 2.2] },
    { gamma: [0, 0, 0] },
    { contrast: 1.6, pivot: 0.18 },
    { contrast: 99, pivot: 0 },
    { contrast: 0, pivot: 0.5 },
    { saturation: 0 },
    { saturation: 2.5 },
    { offset: [0.02, -0.02, 0], lift: [0.2, 0, -0.1], gain: [2, 0.5, 1],
      gamma: [0.5, 1, 2.4], contrast: 3, pivot: 0.3, saturation: 1.8 },
];

// vec4f everywhere rather than vec3f. WGSL aligns vec3f to 16 bytes anyway, so
// the padding is the same; writing it out means the JS side and the shader
// agree about the layout without anyone having to remember the rule.
const COMPUTE = `
${WGSL}

struct Grade {
    offset: vec4f,
    lift: vec4f,
    gain: vec4f,
    gamma: vec4f,
    scalars: vec4f,
}

@group(0) @binding(0) var<storage, read> inputs: array<vec4f>;
@group(0) @binding(1) var<storage, read_write> outputs: array<vec4f>;
@group(0) @binding(2) var<uniform> g: Grade;

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) id: vec3u) {
    // The same order gradePixel() applies, including the two conditionals.
    // If this and the JS ever diverge the comparison fails, which is the point.
    var c = radGradeOrder(inputs[id.x].xyz, g.offset.xyz, g.lift.xyz, g.gain.xyz, g.gamma.xyz);
    if (g.scalars.x != 1.0) { c = radContrast(c, g.scalars.x, g.scalars.y); }
    if (g.scalars.z != 1.0) { c = radSaturation(c, g.scalars.z); }
    outputs[id.x] = vec4f(c, 1.0);
}
`;

const out = { ok: false, cases: [], compiled: false };

try {
    if (!navigator.gpu) {
        out.error = 'no WebGPU adapter: navigator.gpu is undefined (is --unstable-webgpu set?)';
        console.log(JSON.stringify(out));
        Deno.exit(0);
    }
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) { out.error = 'no WebGPU adapter'; console.log(JSON.stringify(out)); Deno.exit(0); }
    const device = await adapter.requestDevice();
    out.adapter = adapter.info ? { ...adapter.info } : {};

    device.pushErrorScope('validation');
    const module = device.createShaderModule({ code: COMPUTE, label: 'radiance-grade-wgsl' });
    const info = await module.getCompilationInfo();
    const fatal = info.messages.filter((m) => m.type === 'error');
    out.compileMessages = info.messages.map((m) => ({ type: m.type, message: m.message }));
    if (fatal.length) {
        out.error = 'the emitted WGSL did not compile: ' + fatal.map((m) => m.message).join('\n');
        console.log(JSON.stringify(out));
        Deno.exit(0);
    }
    out.compiled = true;

    const pipeline = device.createComputePipeline({
        layout: 'auto',
        compute: { module, entryPoint: 'main' },
    });
    const validation = await device.popErrorScope();
    if (validation) {
        out.error = 'pipeline creation failed: ' + validation.message;
        console.log(JSON.stringify(out));
        Deno.exit(0);
    }

    const N = PIXELS.length;
    const inData = new Float32Array(N * 4);
    PIXELS.forEach((p, i) => { inData.set([p[0], p[1], p[2], 1], i * 4); });

    const inBuf = device.createBuffer({ size: inData.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(inBuf, 0, inData);
    const outBuf = device.createBuffer({ size: N * 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    const readBuf = device.createBuffer({ size: N * 16, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const uniBuf = device.createBuffer({ size: 80, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });

    const bind = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
            { binding: 0, resource: { buffer: inBuf } },
            { binding: 1, resource: { buffer: outBuf } },
            { binding: 2, resource: { buffer: uniBuf } },
        ],
    });

    for (const g of GRADES) {
        const o = { offset: [0,0,0], lift: [0,0,0], gain: [1,1,1], gamma: [1,1,1],
                    contrast: 1, pivot: 0.18, saturation: 1, ...g };
        const u = new Float32Array(20);
        u.set([...o.offset, 0], 0);
        u.set([...o.lift, 0], 4);
        u.set([...o.gain, 0], 8);
        u.set([...o.gamma, 0], 12);
        u.set([o.contrast, o.pivot, o.saturation, 0], 16);
        device.queue.writeBuffer(uniBuf, 0, u);

        const enc = device.createCommandEncoder();
        const pass = enc.beginComputePass();
        pass.setPipeline(pipeline);
        pass.setBindGroup(0, bind);
        pass.dispatchWorkgroups(N);
        pass.end();
        enc.copyBufferToBuffer(outBuf, 0, readBuf, 0, N * 16);
        device.queue.submit([enc.finish()]);

        await readBuf.mapAsync(GPUMapMode.READ);
        const got = new Float32Array(readBuf.getMappedRange().slice(0));
        readBuf.unmap();

        PIXELS.forEach((px, i) => {
            const gpu = [got[i * 4], got[i * 4 + 1], got[i * 4 + 2]];
            const cpu = gradePixel(px, o);
            // Same rule as the GLSL harness: fp32 tops out at 3.4e38, and gamma
            // 0 floors to 0.01, so an over-range pixel raised to the power of
            // 100 leaves the format behind. That is the format, not the formula.
            const representable = cpu.every((v) => Math.abs(v) < 3.4e38);
            const delta = representable
                ? Math.max(...gpu.map((v, k) => Math.abs(v - cpu[k]) / Math.max(1, Math.abs(cpu[k]))))
                : null;
            out.cases.push({ grade: g, in: px, gpu, cpu, delta, representable });
        });
    }
    out.ok = true;
} catch (e) {
    out.error = String(e && e.message ? e.message : e);
}
console.log(JSON.stringify(out));
