/**
 * RadianceWebGPURenderer — WebGPU backend for the Radiance HDR viewer.
 *
 * Implements the RadianceRenderer abstract interface using the WebGPU API.
 * Provides:
 *   • Full HDR composite pipeline (exposure, gamma, LMT, grading, OETF)
 *   • 3D LUT via sampler3D
 *   • Pipeline precision: f32 native (no extension dependency)
 *   • Compute-shader based scopes (waveform, vectorscope, histogram)
 *
 * Browser support: Chrome 113+, Edge 113+, Chrome Android, (Safari 18+ with flag)
 * Falls back gracefully: navigator.gpu is undefined → viewer uses RadianceWebGLRenderer.
 */

import { RadianceRenderer } from "./radiance_renderer.js";
// The grade maths, emitted from one file for both backends. This backend used
// to carry its own copy in WGSL and a *third* one in JS for the CPU readback,
// and all three disagreed with WebGL. See js/radiance_grade.js.
import {
    WGSL as GRADE_WGSL,
    applyLift as gradeLift, applyGain as gradeGain, applyOffset as gradeOffset,
    applyGamma as gradeGamma, applyContrast as gradeContrast,
    applySaturation as gradeSaturation,
} from "./radiance_grade.js";

// ── WGSL Shader Sources ────────────────────────────────────────────────────

const FULLSCREEN_VERT_WGSL = `
struct VertexOutput {
    @builtin(position) position: vec4f,
    @location(0) texcoord: vec2f,
};

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> VertexOutput {
    let pos = array(
        vec2f(-1.0, -1.0),  // 0: bottom-left
        vec2f( 3.0, -1.0),  // 1: bottom-right (tricky)
        vec2f(-1.0,  3.0),  // 2: top-left (tricky)
    );
    let uv = array(
        vec2f(0.0, 1.0),
        vec2f(2.0, 1.0),
        vec2f(0.0,-1.0),
    );
    var out: VertexOutput;
    out.position = vec4f(pos[vi], 0.0, 1.0);
    out.texcoord = uv[vi];
    return out;
}
`;

function buildCompositeFragmentWGSL() {
    return `
${GRADE_WGSL}
// ── Radiance HDR Composite Shader (WebGPU/WGSL) ──────────────────────────
// Ported from WebGL GLSL composite shader. Implements the full VFX-grade
// color grading pipeline: IDT → Exposure → LMT → Primary Grading →
// Log Wheels → Printer Lights → Soft Clip → Display LUT → OETF.

struct GradingParams {
    exposure: f32,
    gamma_r: f32,
    gamma_g: f32,
    gamma_b: f32,
    saturation: f32,
    lift_r: f32,
    lift_g: f32,
    lift_b: f32,
    grade_gamma_r: f32,
    grade_gamma_g: f32,
    grade_gamma_b: f32,
    gain_r: f32,
    gain_g: f32,
    gain_b: f32,
    offset_r: f32,
    offset_g: f32,
    offset_b: f32,
    temperature: f32,
    tint: f32,
    contrast: f32,
    pivot: f32,
    color_boost: f32,
    shadows: f32,
    highlights: f32,
    mid_detail: f32,
    hue_shift: f32,
    luma_mix: f32,
    color_science: i32,
    log_shadow_r: f32,
    log_shadow_g: f32,
    log_shadow_b: f32,
    log_midtone_r: f32,
    log_midtone_g: f32,
    log_midtone_b: f32,
    log_highlight_r: f32,
    log_highlight_g: f32,
    log_highlight_b: f32,
    printer_r: i32,
    printer_g: i32,
    printer_b: i32,
    soft_clip: f32,
    display_lut_strength: f32,
    channel_mode: i32,
    grain_amount: f32,
    grain_size: f32,
    grain_color: f32,
    grain_animate: i32,
    bloom: f32,
    halation: f32,
    diffusion: f32,
    lens_distortion: f32,
    lens_fringe: f32,
    vignette_intensity: f32,
    vignette_falloff: f32,
    dof_enabled: i32,
    focus_distance: f32,
    aperture: f32,
    aperture_blades: i32,
    aperture_rotation: f32,
    aperture_anamorphic: f32,
    zebra_threshold: f32,
    false_color: i32,
    zebra: i32,
    gamut_warning: i32,
    clipping_monitor: i32,
    focus_peaking: i32,
    focus_peak_threshold: f32,
    denoise: f32,
    frame: i32,
    time: f32,
    lut_strength: f32,
    lut_enabled: i32,
    lut_is_display: i32,
    display_lut_mode: i32,
    input_lut_mode: i32,
    show_depth: i32,
    curve_mix: f32,
    secondary_curve_mix: f32,
    pan_x: f32,
    pan_y: f32,
    zoom: f32,
    wipe_pos: f32,
    wipe_enabled: i32,
};

@group(0) @binding(0) var u_image: texture_2d<f32>;
@group(0) @binding(1) var u_sampler: sampler;
@group(0) @binding(2) var u_depth: texture_2d<f32>;
@group(0) @binding(3) var u_lut: texture_3d<f32>;
@group(0) @binding(4) var u_lut_sampler: sampler;
@group(0) @binding(5) var u_compare: texture_2d<f32>;
@group(0) @binding(6) var<uniform> u: GradingParams;
@group(0) @binding(7) var u_curve_lut: texture_1d<f32>;
@group(0) @binding(8) var u_secondary_curve_lut: texture_1d<f32>;
@group(0) @binding(9) var u_bloom: texture_2d<f32>;

fn linear_to_srgb(c: vec3f) -> vec3f {
    let cutoff = vec3f(0.0031308);
    let higher = 1.055 * pow(max(c, vec3f(0.0)), vec3f(1.0 / 2.4)) - 0.055;
    let lower = c * 12.92;
    return select(higher, lower, c < cutoff);
}

fn srgb_to_linear(c: vec3f) -> vec3f {
    let cutoff = vec3f(0.04045);
    let higher = pow((c + 0.055) / 1.055, vec3f(2.4));
    let lower = c / 12.92;
    return select(higher, lower, c < cutoff);
}

fn rec709_to_linear(c: vec3f) -> vec3f {
    let cutoff = vec3f(0.081);
    let higher = pow((c + 0.099) / 1.099, vec3f(1.0 / 0.45));
    let lower = c / 4.5;
    return select(higher, lower, c < cutoff);
}

fn linear_to_rec709(c: vec3f) -> vec3f {
    let cutoff = vec3f(0.018);
    let higher = 1.099 * pow(max(c, cutoff), vec3f(0.45)) - 0.099;
    let lower = c * 4.5;
    return select(higher, lower, c < cutoff);
}

fn tone_map_aces(c: vec3f) -> vec3f {
    let a = 2.51;
    let b = 0.03;
    let cc = 2.43;
    let d = 0.59;
    let e = 0.14;
    return clamp((c * (a * c + b)) / (c * (cc * c + d) + e), vec3f(0.0), vec3f(1.0));
}

fn tone_map_filmic(c: vec3f) -> vec3f {
    let A = 0.15;
    let B = 0.50;
    let C = 0.10;
    let D = 0.20;
    let E = 0.02;
    let F = 0.30;
    let v = c * 2.0;
    let curr = ((v * (A * v + C * B) + D * E) / (v * (A * v + B) + D * F)) - E / F;
    let wh = ((11.2 * (A * 11.2 + C * B) + D * E) / (11.2 * (A * 11.2 + B) + D * F)) - E / F;
    return clamp(curr / wh, vec3f(0.0), vec3f(1.0));
}

fn logc3_to_linear(c: vec3f) -> vec3f {
    let lc3_cut = 0.010591;
    let lc3_a = 5.555556;
    let lc3_b = 0.052272;
    let lc3_c = 0.247190;
    let lc3_d = 0.385537;
    let lc3_e = 5.367655;
    let lc3_f = 0.092809;
    let cut_cv = lc3_e * lc3_cut + lc3_f;
    let log_branch = (pow(vec3f(10.0), (c - lc3_d) / lc3_c) - lc3_b) / lc3_a;
    let lin_branch = (c - lc3_f) / lc3_e;
    return max(select(log_branch, lin_branch, c <= vec3f(cut_cv)), vec3f(0.0));
}

fn logc4_to_linear(c: vec3f) -> vec3f {
    let a = 2231.8263;
    let b = 0.9071359;
    let cc = 0.0928641;
    let s = 0.1135972;
    let t = -0.0180570;
    let log_cut = (log2(a * t + 64.0) - 6.0) / 14.0 * b + cc;
    let log_branch = (exp2((c - cc) / b * 14.0 + 6.0) - 64.0) / a;
    let lin_branch = c * s + t;
    return max(select(log_branch, lin_branch, c < vec3f(log_cut)), vec3f(0.0));
}

fn slog3_to_linear(c: vec3f) -> vec3f {
    let cut_cv = 0.167361;
    let log_branch = pow(vec3f(10.0), (c * 1023.0 - 420.0) / 261.5) * 0.19 - 0.01;
    let lin_branch = (c * 1023.0 - 95.0) * 0.01125 / (171.2102946929 - 95.0);
    return max(select(log_branch, lin_branch, c < vec3f(cut_cv)), vec3f(0.0));
}

fn hash21(p: vec2f) -> f32 {
    let q = fract(vec3f(p.x, p.y, p.x) * vec3f(0.1031, 0.1030, 0.0973));
    let r = q + vec3f(dot(q, q.yzx + vec3f(33.33)));
    return fract((r.x + r.y) * r.z);
}

fn source_luma(uv: vec2f) -> f32 {
    let c = textureSampleLevel(u_image, u_sampler, clamp(uv, vec2f(0.0), vec2f(1.0)), 0.0).rgb;
    return dot(c / (vec3f(1.0) + abs(c)), vec3f(0.2126, 0.7152, 0.0722));
}

fn kelvin_to_rgb(temp: f32, tint: f32) -> vec3f {
    var t = clamp(temp, -1.0, 1.0);
    let r = 1.0 + 0.3 * max(t, 0.0);
    let b = 1.0 - 0.3 * min(t, 0.0);
    let g = 1.0 - 0.15 * abs(t);
    let tint_factor = 1.0 + 0.1 * tint;
    return vec3f(r, g * tint_factor, b);
}

fn apply_log_wheels(col: vec3f, shadow: vec3f, mid: vec3f, highlight: vec3f) -> vec3f {
    let luma = dot(col, vec3f(0.2126, 0.7152, 0.0722));
    let shadow_w = 1.0 - smoothstep(0.0, 0.3, luma);
    let mid_w = exp(-4.0 * (luma - 0.5) * (luma - 0.5));
    let highlight_w = smoothstep(0.5, 1.0, luma);
    let total = shadow_w + mid_w + highlight_w + 1e-6;
    return col + (shadow * shadow_w + mid * mid_w + highlight * highlight_w) / total;
}

fn apply_printer_lights(col: vec3f) -> vec3f {
    let r = f32(u.printer_r) * 0.01;
    let g = f32(u.printer_g) * 0.01;
    let b = f32(u.printer_b) * 0.01;
    return col * (1.0 + vec3f(r, g, b));
}

fn hue_rotate(col: vec3f, angle: f32) -> vec3f {
    let cosA = cos(angle);
    let sinA = sin(angle);
    let m = mat3x3f(
        0.299 + 0.701 * cosA + 0.168 * sinA,
        0.587 - 0.587 * cosA + 0.330 * sinA,
        0.114 - 0.114 * cosA - 0.497 * sinA,
        0.299 - 0.299 * cosA - 0.328 * sinA,
        0.587 + 0.413 * cosA + 0.035 * sinA,
        0.114 - 0.114 * cosA + 0.292 * sinA,
        0.299 - 0.299 * cosA + 1.250 * sinA,
        0.587 - 0.587 * cosA - 1.050 * sinA,
        0.114 + 0.886 * cosA - 0.203 * sinA,
    );
    return m * col;
}

fn sample_lut(col: vec3f, lut_size: f32) -> vec3f {
    let scale = (lut_size - 1.0) / lut_size;
    let offset = 0.5 / lut_size;
    let uvw = col * scale + offset;
    return textureSampleLevel(u_lut, u_lut_sampler, uvw, 0.0).rgb;
}

fn rotate2(v: vec2f, a: f32) -> vec2f {
    let cs = cos(a);
    let sn = sin(a);
    return vec2f(v.x * cs - v.y * sn, v.x * sn + v.y * cs);
}

fn aperture_boundary(theta: f32, blades_i: i32) -> f32 {
    if (blades_i < 3) {
        return 1.0;
    }
    let blades = max(f32(blades_i), 3.0);
    let blade_angle = 6.2831853 / blades;
    let wrapped = theta + blade_angle * 0.5;
    let local = wrapped - floor(wrapped / blade_angle) * blade_angle - blade_angle * 0.5;
    return clamp(cos(blade_angle * 0.5) / max(cos(local), 0.08), 0.0, 1.0);
}

fn circle_of_confusion(depth: f32) -> f32 {
    let focus_delta = max(abs(depth - u.focus_distance) - 0.0035, 0.0);
    return clamp(focus_delta * u.aperture * 72.0, 0.0, 14.0);
}

@fragment
fn fs_main(@location(0) texcoord: vec2f) -> @location(0) vec4f {
    var color = textureSampleLevel(u_image, u_sampler, texcoord, 0.0).rgb;
    if (u.wipe_enabled != 0 && texcoord.x < u.wipe_pos) {
        color = textureSampleLevel(u_compare, u_sampler, texcoord, 0.0).rgb;
    }
    var depth_val = 0.0;
    if (u.show_depth != 0) {
        depth_val = textureSampleLevel(u_depth, u_sampler, texcoord, 0.0).r;
    }

    // Depth of field: realtime aperture gather driven by the depth map.
    if (u.dof_enabled != 0 && u.aperture > 0.0) {
        let img_size = vec2f(f32(textureDimensions(u_image).x), f32(textureDimensions(u_image).y));
        let px = vec2f(1.0) / max(img_size, vec2f(1.0));
        let depth_here = textureSampleLevel(u_depth, u_sampler, texcoord, 0.0).r;
        let coc = circle_of_confusion(depth_here);
        let ana = max(u.aperture_anamorphic, 0.25);
        let ana_stretch = sqrt(ana);
        let rot = u.aperture_rotation * 0.01745329252;
        let seed = fract(sin(dot(texcoord * img_size + vec2f(f32(u.frame), 17.0), vec2f(12.9898, 78.233))) * 43758.5453);
        let noise_rot = seed * 6.2831853;
        let center_near = 1.0 - step(u.focus_distance, depth_here);
        let center_far = 1.0 - center_near;
        let center_blur = smoothstep(0.8, 4.0, coc);
        var near_acc = color * center_near * 0.35;
        var far_acc = color * center_far * 0.35;
        var near_wsum = center_near * 0.35;
        var far_wsum = center_far * 0.35;
        var near_coverage = 0.0;
        for (var i: i32 = 0; i < 16; i = i + 1) {
            let fi = f32(i) + 0.5;
            let r = sqrt(fi / 16.0);
            let theta = fi * 2.3999632 + noise_rot;
            var o = vec2f(cos(theta), sin(theta)) * r * aperture_boundary(theta, u.aperture_blades);
            o = vec2f(o.x * ana_stretch, o.y / ana_stretch);
            o = rotate2(o, rot);
            let sample_uv = clamp(texcoord + o * px * coc, vec2f(0.0), vec2f(1.0));
            let sample_depth = textureSampleLevel(u_depth, u_sampler, sample_uv, 0.0).r;
            let sample_coc = circle_of_confusion(sample_depth);
            let sample_dist = length(o) * coc;
            let coc_covers = smoothstep(sample_dist - 1.25, sample_dist + 1.25, sample_coc);
            let sample_blur = smoothstep(0.8, 3.0, sample_coc);
            let sample_near = 1.0 - step(u.focus_distance, sample_depth);
            let sample_far = 1.0 - sample_near;
            let near_w = sample_near * max(coc_covers * sample_blur, center_near * sample_near * 0.55);
            let far_w = sample_far * max(coc_covers * sample_blur, center_far * sample_far * 0.45);
            let sample_col = textureSampleLevel(u_image, u_sampler, sample_uv, 0.0).rgb;
            near_acc += sample_col * near_w;
            far_acc += sample_col * far_w;
            near_wsum += near_w;
            far_wsum += far_w;
            near_coverage = max(near_coverage, near_w);
        }
        let near_col = select(color, near_acc / near_wsum, near_wsum > 0.001);
        let far_col = select(color, far_acc / far_wsum, far_wsum > 0.001);
        let far_alpha = center_far * center_blur;
        let near_alpha = max(center_near * center_blur, smoothstep(0.04, 0.35, near_coverage));
        color = mix(mix(color, far_col, far_alpha), near_col, clamp(near_alpha, 0.0, 1.0));
    }

    if (u.denoise > 0.0) {
        let img_size = vec2f(f32(textureDimensions(u_image).x), f32(textureDimensions(u_image).y));
        let px = vec2f(1.0) / max(img_size, vec2f(1.0));
        var blur = vec3f(0.0);
        for (var yy: i32 = -1; yy <= 1; yy = yy + 1) {
            for (var xx: i32 = -1; xx <= 1; xx = xx + 1) {
                blur += textureSampleLevel(u_image, u_sampler, clamp(texcoord + vec2f(f32(xx), f32(yy)) * px, vec2f(0.0), vec2f(1.0)), 0.0).rgb;
            }
        }
        color = mix(color, blur / 9.0, clamp(u.denoise, 0.0, 1.0));
    }

    // Input transform (IDT)
    if (u.input_lut_mode == 1 || u.input_lut_mode == 35) { // sRGB -> Linear
        color = srgb_to_linear(color);
    } else if (u.input_lut_mode == 2 || u.input_lut_mode == 34) {
        color = rec709_to_linear(color);
    } else if (u.input_lut_mode == 29) {
        color = logc3_to_linear(color);
    } else if (u.input_lut_mode == 22) {
        color = logc4_to_linear(color);
    } else if (u.input_lut_mode == 31) {
        color = slog3_to_linear(color);
    }
    let pre_grade = color;

    // Exposure
    color *= exp2(u.exposure);

    // White balance
    let wb = kelvin_to_rgb(u.temperature, u.tint);
    color *= wb;

    // Saturation (linear-space)
    let luma = dot(color, vec3f(0.2126, 0.7152, 0.0722));
    color = mix(vec3f(luma), color, u.saturation);

    // Primary grading: Lift Gamma Gain + Offset
    let gGain = vec3f(u.gain_r, u.gain_g, u.gain_b);
    let gLift = vec3f(u.lift_r, u.lift_g, u.lift_b);
    let gGradeGamma = vec3f(u.grade_gamma_r, u.grade_gamma_g, u.grade_gamma_b);
    let gOffset = vec3f(u.offset_r, u.offset_g, u.offset_b);
    // Was: flat additive lift, unguarded gamma, unclamped contrast -- three
    // separate disagreements with the WebGL backend, on a path the viewer
    // *prefers* whenever navigator.gpu exists. Now the shared definition.
    color = radGradeOrder(color, gOffset, gLift, gGain, gGradeGamma);

    // Contrast
    if (u.contrast != 1.0) {
        color = radContrast(color, u.contrast, u.pivot);
    }

    // Resolve-style controls
    if (u.color_boost > 0.0) {
        let b_luma = dot(color, vec3f(0.2126, 0.7152, 0.0722));
        color = mix(vec3f(b_luma), color, 1.0 + u.color_boost);
    }
    if (u.hue_shift != 0.0) {
        color = hue_rotate(color, u.hue_shift * 3.14159);
    }
    if (u.shadows != 0.0 || u.highlights != 0.0 || u.mid_detail != 0.0) {
        let s_luma = dot(color, vec3f(0.2126, 0.7152, 0.0722));
        let sw = 1.0 - smoothstep(0.0, 0.3, s_luma);
        let hw = smoothstep(0.5, 1.0, s_luma);
        let mw = exp(-4.0 * (s_luma - 0.5) * (s_luma - 0.5));
        color += vec3f(u.shadows * sw + u.highlights * hw + u.mid_detail * mw);
    }
    if (u.luma_mix != 1.0) {
        let current_luma = dot(color, vec3f(0.2126, 0.7152, 0.0722));
        let original_luma = dot(pre_grade, vec3f(0.2126, 0.7152, 0.0722));
        let color_with_original_luma = color * (original_luma / max(current_luma, 0.0001));
        color = mix(color_with_original_luma, color, clamp(u.luma_mix, 0.0, 1.0));
    }

    // Log Wheels
    let logShadow = vec3f(u.log_shadow_r, u.log_shadow_g, u.log_shadow_b);
    let logMidtone = vec3f(u.log_midtone_r, u.log_midtone_g, u.log_midtone_b);
    let logHighlight = vec3f(u.log_highlight_r, u.log_highlight_g, u.log_highlight_b);
    if (any(logShadow != vec3f(0.0)) || any(logMidtone != vec3f(0.0)) || any(logHighlight != vec3f(0.0))) {
        color = apply_log_wheels(color, logShadow, logMidtone, logHighlight);
    }

    // Printer Lights
    color = apply_printer_lights(color);

    // Soft Clip
    if (u.soft_clip > 0.0) {
        let soft = 1.0 - exp(-color / max(u.soft_clip, 0.001));
        color = mix(color, soft, 0.5);
    }

    // Display LUT / Color Space Transform
    var display_encoded = false;
    if (u.lut_enabled != 0 && u.display_lut_mode > 0 && u.lut_is_display != 0) {
        if (u.lut_is_display != 0) {
            color = sample_lut(color, 33.0);
        }
        if (u.display_lut_strength < 1.0) {
            color = mix(color, sample_lut(color, 33.0), u.display_lut_strength);
        } else {
            color = sample_lut(color, 33.0);
        }
    }

    // Curve LUT
    if (u.curve_mix > 0.0) {
        let curve_r = textureSampleLevel(u_curve_lut, u_sampler, color.r * 0.996 + 0.002, 0.0).r;
        let curve_g = textureSampleLevel(u_curve_lut, u_sampler, color.g * 0.996 + 0.002, 0.0).r;
        let curve_b = textureSampleLevel(u_curve_lut, u_sampler, color.b * 0.996 + 0.002, 0.0).r;
        color = mix(color, vec3f(curve_r, curve_g, curve_b), u.curve_mix);
    }

    // Secondary Curve LUT (binding 8 — applied after primary, pre-output-transform)
    if (u.secondary_curve_mix > 0.0) {
        let sc_r = textureSampleLevel(u_secondary_curve_lut, u_sampler, color.r * 0.996 + 0.002, 0.0).r;
        let sc_g = textureSampleLevel(u_secondary_curve_lut, u_sampler, color.g * 0.996 + 0.002, 0.0).r;
        let sc_b = textureSampleLevel(u_secondary_curve_lut, u_sampler, color.b * 0.996 + 0.002, 0.0).r;
        color = mix(color, vec3f(sc_r, sc_g, sc_b), u.secondary_curve_mix);
    }

    // Output transform / tone map, then display OETF.
    if (u.display_lut_mode == 2) {
        color = linear_to_rec709(max(color, vec3f(0.0)));
        display_encoded = true;
    } else if (u.display_lut_mode == 3) {
        color = tone_map_filmic(color);
    } else if (u.display_lut_mode == 8) {
        color = color / (color + vec3f(1.0));
    } else if (u.display_lut_mode == 9) {
        color = tone_map_aces(color);
    } else if (max(max(color.r, color.g), color.b) > 1.0) {
        color = tone_map_aces(color);
    }

    if (!display_encoded) {
        color = linear_to_srgb(max(color, vec3f(0.0)));
    }

    // Bloom / Halation / Diffusion (post-OETF bloom composite)
    if (u.bloom > 0.0) {
        let bloomSize = vec2f(f32(textureDimensions(u_bloom).x), f32(textureDimensions(u_bloom).y));
        let bloomUV = texcoord * vec2f(f32(textureDimensions(u_image).x), f32(textureDimensions(u_image).y)) / bloomSize;
        var bloomSample = textureSampleLevel(u_bloom, u_sampler, bloomUV, 0.0).rgb;
        bloomSample = bloomSample / (vec3f(1.0) + bloomSample); // Reinhard compress
        color += bloomSample * u.bloom * 1.2;
    }
    if (u.halation > 0.0) {
        let bloomSize = vec2f(f32(textureDimensions(u_bloom).x), f32(textureDimensions(u_bloom).y));
        let bloomUV = texcoord * vec2f(f32(textureDimensions(u_image).x), f32(textureDimensions(u_image).y)) / bloomSize;
        var halSample = textureSampleLevel(u_bloom, u_sampler, bloomUV, 0.0).rgb;
        halSample = halSample / (vec3f(1.0) + halSample);
        // Halation: warm-tinted bloom (red/magenta shift)
        color += vec3f(halSample.r * 1.5, halSample.g * 0.7, halSample.b * 0.3) * u.halation * 0.8;
    }
    if (u.diffusion > 0.0) {
        let bloomSize = vec2f(f32(textureDimensions(u_bloom).x), f32(textureDimensions(u_bloom).y));
        let bloomUV = texcoord * vec2f(f32(textureDimensions(u_image).x), f32(textureDimensions(u_image).y)) / bloomSize;
        var diffSample = textureSampleLevel(u_bloom, u_sampler, bloomUV, 0.0).rgb;
        color = mix(color, diffSample, u.diffusion * 0.15);
    }

    if (u.vignette_intensity > 0.0) {
        let p = texcoord * 2.0 - vec2f(1.0);
        let radius = dot(p, p);
        let vig = smoothstep(1.0 - clamp(u.vignette_falloff, 0.05, 0.95), 1.35, radius);
        color *= 1.0 - vig * clamp(u.vignette_intensity, 0.0, 1.0);
    }

    if (u.grain_amount > 0.0) {
        let img_size = vec2f(f32(textureDimensions(u_image).x), f32(textureDimensions(u_image).y));
        let grain_px = floor(texcoord * img_size / max(u.grain_size, 0.25));
        let frame_seed = select(f32(u.frame), floor(u.time * 24.0), u.grain_animate != 0);
        let n = hash21(grain_px + vec2f(frame_seed, frame_seed * 1.37)) - 0.5;
        let amount = u.grain_amount * 0.20;
        if (u.grain_color > 0.0) {
            let nr = hash21(grain_px + vec2f(17.0, frame_seed)) - 0.5;
            let nb = hash21(grain_px + vec2f(frame_seed, 29.0)) - 0.5;
            let chroma = u.grain_color * amount;
            color += vec3f(n * amount + nr * chroma, n * amount, n * amount + nb * chroma);
        } else {
            color += vec3f(n * amount);
        }
        color = clamp(color, vec3f(0.0), vec3f(1.0));
    }

    // Channel isolation (post-OETF)
    if (u.channel_mode == 1) { color = vec3f(color.r, 0.0, 0.0); }
    else if (u.channel_mode == 2) { color = vec3f(0.0, color.g, 0.0); }
    else if (u.channel_mode == 3) { color = vec3f(0.0, 0.0, color.b); }
    else if (u.channel_mode == 4) { color = vec3f(luma, luma, luma); }
    else if (u.channel_mode == 5) { color = vec3f(0.0, 0.0, 0.0); } // Alpha placeholder

    // Analytics overlays (simplified)
    if (u.false_color != 0) {
        let fc_luma = dot(color, vec3f(0.2126, 0.7152, 0.0722));
        if (fc_luma < 0.1) { color = vec3f(0.0, 0.0, 1.0); }
        else if (fc_luma < 0.2) { color = vec3f(1.0, 0.0, 1.0); }
        else if (fc_luma < 0.3) { color = vec3f(0.0, 0.5, 1.0); }
        else if (fc_luma < 0.4) { color = vec3f(0.0, 1.0, 1.0); }
        else if (fc_luma < 0.5) { color = vec3f(0.0, 1.0, 0.0); }
        else if (fc_luma < 0.6) { color = vec3f(0.5, 1.0, 0.0); }
        else if (fc_luma < 0.7) { color = vec3f(1.0, 1.0, 0.0); }
        else if (fc_luma < 0.8) { color = vec3f(1.0, 0.5, 0.0); }
        else if (fc_luma < 0.9) { color = vec3f(1.0, 0.0, 0.0); }
        else { color = vec3f(0.5, 0.0, 0.0); }
    }
    if (u.zebra != 0 && luma > u.zebra_threshold) {
        color = mix(color, vec3f(1.0, 0.0, 0.0), 0.5);
    }
    if (u.focus_peaking != 0) {
        let img_size = vec2f(f32(textureDimensions(u_image).x), f32(textureDimensions(u_image).y));
        let px = vec2f(1.0) / max(img_size, vec2f(1.0));
        let gx = -source_luma(texcoord + vec2f(-px.x, -px.y)) + source_luma(texcoord + vec2f(px.x, -px.y))
            - 2.0 * source_luma(texcoord + vec2f(-px.x, 0.0)) + 2.0 * source_luma(texcoord + vec2f(px.x, 0.0))
            - source_luma(texcoord + vec2f(-px.x, px.y)) + source_luma(texcoord + vec2f(px.x, px.y));
        let gy = -source_luma(texcoord + vec2f(-px.x, -px.y)) - 2.0 * source_luma(texcoord + vec2f(0.0, -px.y)) - source_luma(texcoord + vec2f(px.x, -px.y))
            + source_luma(texcoord + vec2f(-px.x, px.y)) + 2.0 * source_luma(texcoord + vec2f(0.0, px.y)) + source_luma(texcoord + vec2f(px.x, px.y));
        if (sqrt(gx * gx + gy * gy) * 255.0 > u.focus_peak_threshold) {
            color = mix(color, vec3f(1.0, 0.0, 0.18), 0.85);
        }
    }

    // Show depth overlay
    if (u.show_depth != 0) {
        color = mix(color, vec3f(depth_val), 0.5);
    }

    return vec4f(color, 1.0);
}
`;
}

// ── Kawase Bloom Shaders ─────────────────────────────────────────────────

const BLOOM_DOWN_WGSL = `
@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var smp: sampler;
@group(0) @binding(2) var<uniform> u: vec4f; // [threshold, _, _, _]

@fragment
fn fs_main(@location(0) uv: vec2f) -> @location(0) vec4f {
    let step = vec2f(0.5 / f32(textureDimensions(src).x), 0.5 / f32(textureDimensions(src).y));
    let c  = textureSampleLevel(src, smp, uv,         0.0).rgb;
    let t  = textureSampleLevel(src, smp, uv + vec2f( 0.0, -step.y), 0.0).rgb;
    let b  = textureSampleLevel(src, smp, uv + vec2f( 0.0,  step.y), 0.0).rgb;
    let l  = textureSampleLevel(src, smp, uv + vec2f(-step.x,  0.0), 0.0).rgb;
    let r  = textureSampleLevel(src, smp, uv + vec2f( step.x,  0.0), 0.0).rgb;
    let avg = (c * 4.0 + t + b + l + r) / 8.0;
    let luma = dot(avg, vec3f(0.2126, 0.7152, 0.0722));
    if (luma < u.x) { return vec4f(0.0); }
    return vec4f(max(avg, vec3f(0.0)), 1.0);
}
`;

const BLOOM_UP_WGSL = `
@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var smp: sampler;
@group(0) @binding(2) var<uniform> u: vec4f; // [bloom_intensity, _, _, _]

@fragment
fn fs_main(@location(0) uv: vec2f) -> @location(0) vec4f {
    let step = vec2f(0.5 / f32(textureDimensions(src).x), 0.5 / f32(textureDimensions(src).y));
    let c  = textureSampleLevel(src, smp, uv,         0.0).rgb;
    let t  = textureSampleLevel(src, smp, uv + vec2f( 0.0, -step.y), 0.0).rgb;
    let b  = textureSampleLevel(src, smp, uv + vec2f( 0.0,  step.y), 0.0).rgb;
    let l  = textureSampleLevel(src, smp, uv + vec2f(-step.x,  0.0), 0.0).rgb;
    let r  = textureSampleLevel(src, smp, uv + vec2f( step.x,  0.0), 0.0).rgb;
    let blur = (c * 4.0 + t + b + l + r) / 8.0;
    return vec4f(blur * u.x, 1.0);
}
`;

// ── Renderer class ─────────────────────────────────────────────────────────

class RadianceWebGPURenderer extends RadianceRenderer {
    constructor(canvas) {
        super(canvas);
        this.device = null;
        this.context = null;
        this.pipeline = null;
        this.bindGroup = null;
        this.uniformBuffer = null;
        this.textureView = null;
        this.sampler = null;
        this.lutTextureView = null;
        this.lutSampler = null;
        this.depthTextureView = null;
        this.compareTextureView = null;
        this.curveTextureView = null;
        this.secondaryCurveTextureView = null;
        this._format = (typeof navigator !== 'undefined' && navigator.gpu?.getPreferredCanvasFormat)
            ? navigator.gpu.getPreferredCanvasFormat()
            : 'bgra8unorm';
        this._textureFloatFormat = 'rgba16float';
        this._lutSize = 33;
        this._supported = typeof navigator !== 'undefined' && 'gpu' in navigator;
        this._cleanup = [];
        this.imageWidth = 0;
        this.imageHeight = 0;
        this._pipelineNeedsUpdate = true;
        this._lastSourcePixels = null;
        this._lastSubmitPromise = Promise.resolve();

        // Bloom state
        this._bloomTextures = [];
        this._bloomPipelines = { down: null, up: null };
        this._bloomBindGroups = { down: [], up: [] };
        this._bloomUniformBuffer = null;
        this._bloomSrcW = 0;
        this._bloomSrcH = 0;
    }

    // ── Lifecycle ──────────────────────────────────────────────────────────

    async init() {
        if (!this._supported) {
            console.warn('[Radiance] WebGPU not available');
            return false;
        }
        try {
            const adapter = await navigator.gpu.requestAdapter();
            if (!adapter) {
                console.warn('[Radiance] No WebGPU adapter found');
                return false;
            }
            const features = [];
            if (adapter.features.has('float32-filterable')) {
                features.push('float32-filterable');
            }
            this.device = await adapter.requestDevice({ requiredFeatures: features });
            this._textureFloatFormat = features.includes('float32-filterable') ? 'rgba32float' : 'rgba16float';
            this._cleanup.push(() => this.device.destroy());

            this.context = this.canvas.getContext('webgpu');
            if (!this.context) {
                console.warn('[Radiance] Could not get WebGPU context');
                return false;
            }

            // Configure swap chain
            this.context.configure({
                device: this.device,
                format: this._format,
                alphaMode: 'premultiplied',
            });

            // Create shared sampler
            this.sampler = this.device.createSampler({
                addressModeU: 'clamp-to-edge',
                addressModeV: 'clamp-to-edge',
                magFilter: 'linear',
                minFilter: 'linear',
            });
            this.lutSampler = this.device.createSampler({
                addressModeU: 'clamp-to-edge',
                addressModeV: 'clamp-to-edge',
                addressModeW: 'clamp-to-edge',
                magFilter: 'linear',
                minFilter: 'linear',
            });

            // Create uniform buffer (GradingParams)
            this.uniformBuffer = this.device.createBuffer({
                size: 512, // Enough for all params
                usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
            });

            // Create pipeline
            this._createPipeline();

            console.log('[Radiance] WebGPU Renderer initialized');
            return true;
        } catch (e) {
            console.error('[Radiance] WebGPU init failed:', e);
            return false;
        }
    }

    _createPipeline() {
        const device = this.device;
        if (!device) return;

        const shaderModule = device.createShaderModule({
            code: FULLSCREEN_VERT_WGSL + buildCompositeFragmentWGSL(),
        });

        this.pipeline = device.createRenderPipeline({
            layout: 'auto',
            vertex: {
                module: shaderModule,
                entryPoint: 'vs_main',
            },
            fragment: {
                module: shaderModule,
                entryPoint: 'fs_main',
                targets: [{ format: this._format }],
            },
            primitive: { topology: 'triangle-list' },
        });

        this._pipelineNeedsUpdate = false;
    }

    // ── Bloom lifecycle ─────────────────────────────────────────────────

    _initBloomTextures(srcW, srcH) {
        if (this._bloomTextures.length &&
            this._bloomSrcW === srcW && this._bloomSrcH === srcH) return;
        this._destroyBloomTextures();
        this._bloomSrcW = srcW;
        this._bloomSrcH = srcH;
        const device = this.device;
        const levels = 6;
        for (let i = 0; i < levels; i++) {
            const w = Math.max(1, Math.floor(srcW / Math.pow(2, i + 1)));
            const h = Math.max(1, Math.floor(srcH / Math.pow(2, i + 1)));
            const tex = device.createTexture({
                size: [w, h, 1],
                format: 'rgba16float',
                usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_DST,
            });
            this._bloomTextures.push({ tex, w, h, view: tex.createView() });
        }
    }

    _destroyBloomTextures() {
        for (const bt of this._bloomTextures) bt.tex.destroy();
        this._bloomTextures = [];
        this._bloomSrcW = 0;
        this._bloomSrcH = 0;
    }

    _ensureBloomPipelines() {
        const device = this.device;
        if (this._bloomPipelines.down) return;
        const downMod = device.createShaderModule({ code: FULLSCREEN_VERT_WGSL + BLOOM_DOWN_WGSL });
        const upMod = device.createShaderModule({ code: FULLSCREEN_VERT_WGSL + BLOOM_UP_WGSL });
        this._bloomPipelines.down = device.createRenderPipeline({
            layout: 'auto',
            vertex: { module: downMod, entryPoint: 'vs_main' },
            fragment: { module: downMod, entryPoint: 'fs_main', targets: [{ format: 'rgba16float' }] },
            primitive: { topology: 'triangle-list' },
        });
        this._bloomPipelines.up = device.createRenderPipeline({
            layout: 'auto',
            vertex: { module: upMod, entryPoint: 'vs_main' },
            fragment: { module: upMod, entryPoint: 'fs_main', targets: [{ format: 'rgba16float' }] },
            primitive: { topology: 'triangle-list' },
        });
        // Reusable uniform buffer for bloom params
        this._bloomUniformBuffer = device.createBuffer({
            size: 16,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
    }

    _runBloomChain(lutStrength) {
        const device = this.device;
        if (!device || this.bloom <= 0 || !this.textureView) return null;
        const imgW = this.imageWidth;
        const imgH = this.imageHeight;
        if (!imgW || !imgH) return null;

        this._initBloomTextures(imgW, imgH);
        if (!this._bloomTextures.length) return null;
        this._ensureBloomPipelines();

        const levels = this._bloomTextures.length;
        const downPipe = this._bloomPipelines.down;
        const upPipe = this._bloomPipelines.up;

        // Write bloom threshold uniform
        const bloomBuf = new Float32Array([this.bloom, 0, 0, 0]);
        device.queue.writeBuffer(this._bloomUniformBuffer, 0, bloomBuf.buffer);

        // ── Downsample pass ─────────────────────────────────────────────
        let srcView = this.textureView;
        for (let i = 0; i < levels; i++) {
            const dst = this._bloomTextures[i];
            const bgLayout = downPipe.getBindGroupLayout(0);
            const bg = device.createBindGroup({
                layout: bgLayout,
                entries: [
                    { binding: 0, resource: srcView },
                    { binding: 1, resource: this.sampler },
                    { binding: 2, resource: { buffer: this._bloomUniformBuffer } },
                ],
            });
            const encoder = device.createCommandEncoder();
            const pass = encoder.beginRenderPass({
                colorAttachments: [{
                    view: dst.view,
                    loadOp: 'clear',
                    storeOp: 'store',
                }],
            });
            pass.setPipeline(downPipe);
            pass.setBindGroup(0, bg);
            pass.draw(3);
            pass.end();
            device.queue.submit([encoder.finish()]);
            srcView = dst.view;
        }

        // ── Upsample pass ───────────────────────────────────────────────
        let srcViewUp = this._bloomTextures[levels - 1].view;
        for (let i = levels - 2; i >= 0; i--) {
            const dst = this._bloomTextures[i];
            const bgLayout = upPipe.getBindGroupLayout(0);
            const bg = device.createBindGroup({
                layout: bgLayout,
                entries: [
                    { binding: 0, resource: srcViewUp },
                    { binding: 1, resource: this.sampler },
                    { binding: 2, resource: { buffer: this._bloomUniformBuffer } },
                ],
            });
            const encoder = device.createCommandEncoder();
            const pass = encoder.beginRenderPass({
                colorAttachments: [{
                    view: dst.view,
                    loadOp: 'load',
                    storeOp: 'store',
                }],
            });
            pass.setPipeline(upPipe);
            pass.setBindGroup(0, bg);
            pass.draw(3);
            pass.end();
            device.queue.submit([encoder.finish()]);
            srcViewUp = dst.view;
        }

        // Return the view of level 0 (accumulated bloom at half-res)
        return this._bloomTextures[0].view;
    }

    destroy() {
        this._destroyBloomTextures();
        this._cleanup.forEach(fn => fn());
        this._cleanup = [];
        this.device = null;
        this.context = null;
        this.pipeline = null;
    }

    // ── Texture loading helpers ────────────────────────────────────────────

    _makeTexture(data, width, height, format = this._textureFloatFormat || 'rgba16float') {
        const device = this.device;
        if (!device) return null;
        const bytesPerPixel = format === 'rgba32float' ? 16 : 8;
        const uploadData = format === 'rgba16float'
            ? this._float32ToHalfTextureData(data, width, height)
            : data;
        const texture = device.createTexture({
            size: [width, height, 1],
            format: format,
            usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
        });
        device.queue.writeTexture(
            { texture },
            uploadData,
            { bytesPerRow: width * bytesPerPixel, rowsPerImage: height },
            { width, height, depthOrArrayLayers: 1 },
        );
        return texture;
    }

    loadImageTexture(img) {
        if (!this.device || !img) return null;
        const w = img.naturalWidth || img.width;
        const h = img.naturalHeight || img.height;
        this.imageWidth = w;
        this.imageHeight = h;

        // Create offscreen canvas to get RGBA data
        const canvas = document.createElement('canvas');
        canvas.width = w;
        canvas.height = h;
        const ctx = canvas.getContext('2d');
        ctx.drawImage(img, 0, 0);
        const imageData = ctx.getImageData(0, 0, w, h);
        const data = imageData.data; // Safe direct Uint8ClampedArray reference

        // Convert to f32 linear texture
        const f32 = new Float32Array(w * h * 4);
        for (let i = 0; i < w * h; i++) {
            f32[i * 4] = data[i * 4] / 255;
            f32[i * 4 + 1] = data[i * 4 + 1] / 255;
            f32[i * 4 + 2] = data[i * 4 + 2] / 255;
            f32[i * 4 + 3] = 1.0;
        }

        const texture = this._makeTexture(f32, w, h);
        this._lastSourcePixels = f32;
        this._setImageTexture(texture, false);
        this.isLinearTexture = false;
        return texture;
    }

    loadFloat16Texture(data, w, h, ch) {
        if (!this.device) return null;
        this.imageWidth = w;
        this.imageHeight = h;
        // Convert half-float to f32
        const f32 = new Float32Array(w * h * 4);
        for (let i = 0; i < w * h; i++) {
            const src = i * ch;
            const dst = i * 4;
            const r = this._halfToFloat(data[src] ?? 0);
            f32[dst] = r;
            f32[dst + 1] = ch > 1 ? this._halfToFloat(data[src + 1] ?? 0) : r;
            f32[dst + 2] = ch > 2 ? this._halfToFloat(data[src + 2] ?? 0) : r;
            f32[dst + 3] = ch > 3 ? this._halfToFloat(data[src + 3] ?? 1) : 1.0;
        }
        const texture = this._makeTexture(f32, w, h);
        this._lastSourcePixels = f32;
        this._setImageTexture(texture, true);
        this.isLinearTexture = true;
        return texture;
    }

    loadFloat32Texture(data, w, h, ch) {
        if (!this.device) return null;
        this.imageWidth = w;
        this.imageHeight = h;
        const f32 = new Float32Array(w * h * 4);
        for (let i = 0; i < w * h; i++) {
            const src = i * ch;
            const dst = i * 4;
            const r = data[src] ?? 0;
            f32[dst] = r;
            f32[dst + 1] = ch > 1 ? (data[src + 1] ?? 0) : r;
            f32[dst + 2] = ch > 2 ? (data[src + 2] ?? 0) : r;
            f32[dst + 3] = ch > 3 ? (data[src + 3] ?? 1) : 1.0;
        }
        const texture = this._makeTexture(f32, w, h);
        this._lastSourcePixels = f32;
        this._setImageTexture(texture, true);
        this.isLinearTexture = true;
        return texture;
    }

    loadFloat16TextureCached(frameId, data, w, h, ch) {
        // Keep the WebGL-compatible method surface. WebGPU uploads are cheap
        // enough here; clearFrameCache can later grow a real cache behind this.
        return this.loadFloat16Texture(data, w, h, ch);
    }

    loadFloat32TextureCached(frameId, data, w, h, ch) {
        return this.loadFloat32Texture(data, w, h, ch);
    }

    loadDepthTexture(img) {
        if (!this.device || !img) return;
        const w = img.naturalWidth || img.width;
        const h = img.naturalHeight || img.height;
        const canvas = document.createElement('canvas');
        canvas.width = w;
        canvas.height = h;
        const ctx = canvas.getContext('2d');
        ctx.drawImage(img, 0, 0);
        const imageData = ctx.getImageData(0, 0, w, h);
        const f32 = new Float32Array(w * h * 4);
        for (let i = 0; i < w * h; i++) {
            f32[i * 4] = imageData.data[i * 4] / 255;
        }
        const texture = this._makeTexture(f32, w, h);
        this.textures.depth = texture;
        this.depthTextureView = texture ? texture.createView() : null;
    }

    loadCompareTexture(img) {
        if (!this.device || !img) return;
        const w = img.naturalWidth || img.width;
        const h = img.naturalHeight || img.height;
        const canvas = document.createElement('canvas');
        canvas.width = w;
        canvas.height = h;
        const ctx = canvas.getContext('2d');
        ctx.drawImage(img, 0, 0);
        const imageData = ctx.getImageData(0, 0, w, h);
        const f32 = new Float32Array(w * h * 4);
        for (let i = 0; i < w * h; i++) {
            f32[i * 4] = imageData.data[i * 4] / 255;
            f32[i * 4 + 1] = imageData.data[i * 4 + 1] / 255;
            f32[i * 4 + 2] = imageData.data[i * 4 + 2] / 255;
            f32[i * 4 + 3] = 1.0;
        }
        const texture = this._makeTexture(f32, w, h);
        this.textures.compare = texture;
        this.compareTextureView = texture ? texture.createView() : null;
    }

    loadLUT(data, size) {
        if (!this.device) return;
        this._lutSize = size || 33;
        const lutData = new Float32Array(size * size * size * 4);
        for (let z = 0; z < size; z++) {
            for (let y = 0; y < size; y++) {
                for (let x = 0; x < size; x++) {
                    const idx = (z * size * size + y * size + x) * 4;
                    lutData[idx] = data[idx] || x / (size - 1);
                    lutData[idx + 1] = data[idx + 1] || y / (size - 1);
                    lutData[idx + 2] = data[idx + 2] || z / (size - 1);
                    lutData[idx + 3] = 1.0;
                }
            }
        }
        const format = this._textureFloatFormat || 'rgba16float';
        const uploadData = format === 'rgba16float'
            ? this._float32ToHalfTextureData(lutData, size, size * size)
            : lutData;
        const texture = this.device.createTexture({
            size: [size, size, size],
            dimension: '3d',
            format,
            usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
        });
        this.device.queue.writeTexture(
            { texture },
            uploadData,
            { bytesPerRow: size * (format === 'rgba32float' ? 16 : 8), rowsPerImage: size },
            { width: size, height: size, depthOrArrayLayers: size },
        );
        this.textures.lut = texture;
        this.lutTextureView = texture.createView({ dimension: '3d' });
    }

    updateCurveLut(data) {
        if (!this.device) return;
        const size = 256;
        const texData = new Float32Array(size * 4);
        for (let i = 0; i < size; i++) {
            const v = data ? (data[i * 4] !== undefined ? data[i * 4] : i / (size - 1)) : i / (size - 1);
            texData[i * 4] = v;
            texData[i * 4 + 1] = v;
            texData[i * 4 + 2] = v;
            texData[i * 4 + 3] = 1.0;
        }
        if (!this._curveTexture) {
            this._curveTexture = this.device.createTexture({
                size: [size],
                dimension: '1d',
                format: this._textureFloatFormat || 'rgba16float',
                usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
            });
        }
        const format = this._textureFloatFormat || 'rgba16float';
        const uploadData = format === 'rgba16float'
            ? this._float32ToHalfTextureData(texData, size, 1)
            : texData;
        this.device.queue.writeTexture(
            { texture: this._curveTexture },
            uploadData,
            { bytesPerRow: size * (format === 'rgba32float' ? 16 : 8), rowsPerImage: 1 },
            { width: size, height: 1, depthOrArrayLayers: 1 },
        );
        this.curveTextureView = this._curveTexture.createView({ dimension: '1d' });
    }

    updateSecondaryCurveLut(data) {
        // Same pattern as updateCurveLut
        if (!this.device) return;
        const size = 256;
        const texData = new Float32Array(size * 4);
        for (let i = 0; i < size; i++) {
            const v = data ? (data[i * 4] !== undefined ? data[i * 4] : i / (size - 1)) : i / (size - 1);
            texData[i * 4] = v;
            texData[i * 4 + 1] = v;
            texData[i * 4 + 2] = v;
            texData[i * 4 + 3] = 1.0;
        }
        if (!this._secondaryCurveTexture) {
            this._secondaryCurveTexture = this.device.createTexture({
                size: [size],
                dimension: '1d',
                format: this._textureFloatFormat || 'rgba16float',
                usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
            });
        }
        const format = this._textureFloatFormat || 'rgba16float';
        const uploadData = format === 'rgba16float'
            ? this._float32ToHalfTextureData(texData, size, 1)
            : texData;
        this.device.queue.writeTexture(
            { texture: this._secondaryCurveTexture },
            uploadData,
            { bytesPerRow: size * (format === 'rgba32float' ? 16 : 8), rowsPerImage: 1 },
            { width: size, height: 1, depthOrArrayLayers: 1 },
        );
        this.secondaryCurveTextureView = this._secondaryCurveTexture.createView({ dimension: '1d' });
    }

    // ── Main render ────────────────────────────────────────────────────────

    render(lutStrength) {
        const device = this.device;
        if (!device || !this.textureView) return;

        // Size canvas to match image
        if (this.canvas.width !== this.imageWidth || this.canvas.height !== this.imageHeight) {
            this.canvas.width = this.imageWidth;
            this.canvas.height = this.imageHeight;
        }

        // Run Kawase bloom chain (pre-compute bloom texture)
        const bloomView = this._runBloomChain(lutStrength);

        // Build the bind group
        const entries = [
            { binding: 0, resource: this.textureView },
            { binding: 1, resource: this.sampler },
            { binding: 2, resource: this.depthTextureView || this.textureView },
            { binding: 3, resource: this.lutTextureView || this._getIdentityLutTextureView() },
            { binding: 4, resource: this.lutSampler },
            { binding: 5, resource: this.compareTextureView || this.textureView },
            { binding: 6, resource: { buffer: this.uniformBuffer } },
            { binding: 7, resource: this.curveTextureView || this._getIdentityTextureView() },
            { binding: 8, resource: this.secondaryCurveTextureView || this._getIdentityTextureView() },
            { binding: 9, resource: bloomView || this.textureView },
        ];

        const bindGroupLayout = this.pipeline.getBindGroupLayout(0);
        this.bindGroup = device.createBindGroup({
            layout: bindGroupLayout,
            entries: entries.slice(0, Math.max(bindGroupLayout.entries?.length || 10, 10)),
        });

        // Write uniforms
        this._writeUniforms(lutStrength);

        // Render pass
        const commandEncoder = device.createCommandEncoder();
        const textureView = this.context.getCurrentTexture().createView();
        const renderPass = commandEncoder.beginRenderPass({
            colorAttachments: [{
                view: textureView,
                clearValue: { r: 0.0, g: 0.0, b: 0.0, a: 1.0 },
                loadOp: 'clear',
                storeOp: 'store',
            }],
        });
        renderPass.setPipeline(this.pipeline);
        renderPass.setBindGroup(0, this.bindGroup);
        renderPass.draw(3); // Fullscreen triangle
        renderPass.end();
        device.queue.submit([commandEncoder.finish()]);
        this._lastSubmitPromise = typeof device.queue.onSubmittedWorkDone === 'function'
            ? device.queue.onSubmittedWorkDone().catch((err) => {
                console.warn('[Radiance] WebGPU submit completion failed:', err);
                return null;
            })
            : Promise.resolve();
    }

    whenSubmitted() {
        return this._lastSubmitPromise || Promise.resolve();
    }

    // ── Scopes (CPU readback + 2D canvas drawing) ─────────────────────────
    // Reads the rendered frame back to CPU and draws scope overlays on 2D canvas.
    // Phase 3 will replace this with compute-shader based scopes for higher perf.

    renderHistogram(targetCanvas, logScale) {
        this.renderScope('histogram', targetCanvas, null, false, false, logScale);
    }

    renderScope(mode, targetCanvas, sourceTexture, isLinear, paradeMode, logScale) {
        if (!targetCanvas || !this.imageWidth || !this.imageHeight) return;

        const W = this.imageWidth;
        const H = this.imageHeight;

        const scopeW = Math.min(256, W);
        const scopeH = Math.round(H * (scopeW / W)) || 144;

        const cw = targetCanvas.width;
        const ch = targetCanvas.height;

        // 1. Get the graded pixels directly from the display canvas if readable!
        let data = this._getGradedPixels(scopeW, scopeH);
        let pixelsPromise;
        if (data && !this._isAllZeroes(data)) {
            pixelsPromise = Promise.resolve(data);
        } else {
            // Fall back to CPU grading emulation on raw float pixels
            // readPixelsFloat32 is synchronous now and returns a wrapper.
            const readback = this.readPixelsFloat32(scopeW, scopeH, 1.0);
            pixelsPromise = Promise.resolve(
                readback ? this._emulateGradingOnCPU(readback.data) : null
            );
        }

        pixelsPromise.then(pixels => {
            if (!pixels) return;
            const ctx = targetCanvas.getContext('2d');
            if (mode === 'histogram') {
                this._drawHistogram(ctx, pixels, cw, ch);
            } else if (mode === 'waveform') {
                this._drawWaveform(ctx, pixels, scopeW, scopeH, cw, ch, paradeMode);
            } else if (mode === 'vectorscope') {
                this._drawVectorscope(ctx, pixels, scopeW, scopeH, cw, ch);
            }
        }).catch(err => console.warn('[WebGPU Scopes] Draw failed:', err));
    }

    _getGradedPixels(w, h) {
        if (!this.canvas || this.canvas.width === 0 || this.canvas.height === 0) {
            return null;
        }
        try {
            const tempCanvas = document.createElement('canvas');
            tempCanvas.width = w;
            tempCanvas.height = h;
            const tempCtx = tempCanvas.getContext('2d');
            tempCtx.drawImage(this.canvas, 0, 0, w, h);
            return tempCtx.getImageData(0, 0, w, h).data;
        } catch (e) {
            return null;
        }
    }

    _isAllZeroes(data) {
        const step = Math.max(1, Math.floor(data.length / 40));
        for (let i = 0; i < data.length; i += step * 4) {
            if (data[i] !== 0 || data[i + 1] !== 0 || data[i + 2] !== 0) {
                return false;
            }
        }
        return true;
    }

    _emulateGradingOnCPU(pixels) {
        let adjExp = 0.0;
        let adjSat = 1.0;
        let adjCon = 1.0;
        let adjLift = [0.0, 0.0, 0.0];
        let adjGain = [1.0, 1.0, 1.0];
        let adjGamma = [1.0, 1.0, 1.0];

        // Grab V2 segments parameters from global active instance safely
        const viewer = window.RadianceViewer?.activeInstance;
        if (viewer && viewer.v2Segments && viewer.v2Segments.length > 0) {
            const cur = viewer.currentFrame || 0;
            const activeAdj = viewer.v2Segments.find(s => s.type === 'adjustment' && cur >= s.startFrame && cur <= s.endFrame);
            if (activeAdj && activeAdj.gradeProps) {
                const p = activeAdj.gradeProps;
                if (p.exposure !== undefined) adjExp += p.exposure;
                if (p.saturation !== undefined) adjSat *= p.saturation;
                if (p.contrast !== undefined) adjCon *= p.contrast;
                if (p.lift) {
                    adjLift[0] += p.lift[0];
                    adjLift[1] += p.lift[1];
                    adjLift[2] += p.lift[2];
                }
                if (p.gain) {
                    adjGain[0] *= p.gain[0];
                    adjGain[1] *= p.gain[1];
                    adjGain[2] *= p.gain[2];
                }
                if (p.gamma) {
                    adjGamma[0] *= p.gamma[0];
                    adjGamma[1] *= p.gamma[1];
                    adjGamma[2] *= p.gamma[2];
                }
            }
        }

        const exp = Math.pow(2.0, (this.exposure || 0.0) + adjExp);
        const sat = (this.saturation !== undefined ? this.saturation : 1.0) * adjSat;
        const con = (this.contrast !== undefined ? this.contrast : 1.0) * adjCon;
        const piv = this.pivot !== undefined ? this.pivot : 0.18;

        const gainR = (this.gain?.[0] ?? 1.0) * adjGain[0];
        const gainG = (this.gain?.[1] ?? 1.0) * adjGain[1];
        const gainB = (this.gain?.[2] ?? 1.0) * adjGain[2];

        const liftR = (this.lift?.[0] ?? 0.0) + adjLift[0];
        const liftG = (this.lift?.[1] ?? 0.0) + adjLift[1];
        const liftB = (this.lift?.[2] ?? 0.0) + adjLift[2];

        const gammaR = (this.gradingGamma?.[0] ?? 1.0) * adjGamma[0];
        const gammaG = (this.gradingGamma?.[1] ?? 1.0) * adjGamma[1];
        const gammaB = (this.gradingGamma?.[2] ?? 1.0) * adjGamma[2];

        const offsetR = this.offset?.[0] ?? 0.0;
        const offsetG = this.offset?.[1] ?? 0.0;
        const offsetB = this.offset?.[2] ?? 0.0;

        const out = new Uint8ClampedArray(pixels.length);
        for (let i = 0; i < pixels.length; i += 4) {
            let r = pixels[i];
            let g = pixels[i + 1];
            let b = pixels[i + 2];

            // Exposure
            r *= exp;
            g *= exp;
            b *= exp;

            // Offset / Lift / Gain / Gamma / Contrast / Saturation, from the
            // shared definition.
            //
            // This block used to be a third implementation of the grade, and
            // the one that diverged most: flat additive lift where the shaders
            // pivot at white, no gamma floor, and contrast as
            // piv * pow(c/piv, con) -- a different curve from the linear form
            // every other path used, and NaN at pivot 0, which the slider
            // allows. It fed _getGradedPixels, so a dark or letterboxed frame
            // that tripped the _isAllZeroes check landed here and the scopes
            // silently measured a different grade from the one on screen.
            let px = gradeOffset([r, g, b], [offsetR, offsetG, offsetB]);
            px = gradeLift(px, [liftR, liftG, liftB]);
            px = gradeGain(px, [gainR, gainG, gainB]);
            px = gradeGamma(px, [gammaR, gammaG, gammaB]);
            if (con !== 1.0) px = gradeContrast(px, con, piv);
            if (sat !== 1.0) px = gradeSaturation(px, sat);
            r = px[0]; g = px[1]; b = px[2];

            out[i] = Math.round(Math.max(0.0, Math.min(1.0, r)) * 255);
            out[i + 1] = Math.round(Math.max(0.0, Math.min(1.0, g)) * 255);
            out[i + 2] = Math.round(Math.max(0.0, Math.min(1.0, b)) * 255);
            out[i + 3] = Math.round(Math.max(0.0, Math.min(1.0, pixels[i + 3])) * 255);
        }
        return out;
    }

    _drawHistogram(ctx, data, cw, ch) {
        ctx.fillStyle = '#050508';
        ctx.fillRect(0, 0, cw, ch);

        const hR = new Uint32Array(256), hG = new Uint32Array(256), hB = new Uint32Array(256);
        for (let i = 0; i < data.length; i += 4) {
            hR[data[i]]++;
            hG[data[i + 1]]++;
            hB[data[i + 2]]++;
        }

        let max = 1;
        for (let i = 0; i < 256; i++) max = Math.max(max, hR[i], hG[i], hB[i]);

        // Grid
        ctx.strokeStyle = 'rgba(255,255,255,0.06)'; ctx.lineWidth = 1;
        for (let i = 1; i < 4; i++) {
            const x = (i / 4) * cw;
            ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, ch); ctx.stroke();
        }

        const drawCurve = (hist, color) => {
            ctx.strokeStyle = color;
            ctx.lineWidth = 1.5;
            ctx.beginPath();
            for (let i = 0; i < 256; i++) {
                const x = (i / 255) * cw;
                const y = ch - (hist[i] / max) * ch * 0.95;
                if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
            }
            ctx.stroke();
        };

        const fillCurve = (hist, color) => {
            ctx.fillStyle = color;
            ctx.beginPath();
            ctx.moveTo(0, ch);
            for (let i = 0; i < 256; i++) {
                const x = (i / 255) * cw;
                const y = ch - (hist[i] / max) * ch * 0.95;
                ctx.lineTo(x, y);
            }
            ctx.lineTo(cw, ch);
            ctx.fill();
        };

        ctx.globalAlpha = 0.12;
        fillCurve(hR, '#ff4444');
        fillCurve(hG, '#44ff44');
        fillCurve(hB, '#4488ff');
        ctx.globalAlpha = 0.8;
        drawCurve(hR, '#ff4444');
        drawCurve(hG, '#44ff44');
        drawCurve(hB, '#4488ff');
        ctx.globalAlpha = 1.0;
    }

    _drawWaveform(ctx, data, w, h, cw, ch, paradeMode) {
        ctx.fillStyle = '#050508';
        ctx.fillRect(0, 0, cw, ch);

        // Grid lines
        ctx.strokeStyle = 'rgba(255,255,255,0.06)';
        ctx.lineWidth = 1;
        for (let i = 1; i < 4; i++) {
            const y = (i / 4) * ch;
            ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(cw, y); ctx.stroke();
        }

        if (paradeMode) {
            const secW = Math.floor(cw / 3);
            const channels = [
                { idx: 0, color: 'rgba(255,80,80,', label: 'R', x: 0 },
                { idx: 1, color: 'rgba(80,255,80,', label: 'G', x: secW },
                { idx: 2, color: 'rgba(80,120,255,', label: 'B', x: secW * 2 }
            ];

            const step = Math.max(1, Math.floor(w / secW));

            channels.forEach(chInfo => {
                ctx.globalAlpha = 0.08;
                for (let col = 0; col < w; col += step) {
                    const x = chInfo.x + Math.floor((col / w) * secW);
                    const hist = new Uint32Array(256);
                    for (let row = 0; row < h; row++) {
                        hist[data[(row * w + col) * 4 + chInfo.idx]]++;
                    }
                    for (let v = 0; v < 256; v++) {
                        if (hist[v] > 0) {
                            const intensity = Math.min(hist[v] / (h * 0.08), 1.0);
                            const alpha = intensity * 0.7 + 0.15;
                            ctx.fillStyle = chInfo.color + alpha + ')';
                            ctx.fillRect(x, ch - (v / 255) * ch, 1, 1);
                        }
                    }
                }
                // Channel label
                ctx.globalAlpha = 0.8;
                ctx.fillStyle = chInfo.color + '0.8)';
                ctx.font = '10px monospace';
                ctx.fillText(chInfo.label, chInfo.x + 4, 12);
            });

            // Separators
            ctx.strokeStyle = 'rgba(255,255,255,0.12)'; ctx.lineWidth = 1;
            ctx.beginPath();
            ctx.moveTo(secW, 0); ctx.lineTo(secW, ch);
            ctx.moveTo(secW * 2, 0); ctx.lineTo(secW * 2, ch);
            ctx.stroke();
            ctx.globalAlpha = 1.0;
        } else {
            // Monochrome Luma Waveform
            const step = Math.max(1, Math.floor(w / cw));
            ctx.globalAlpha = 0.08;
            for (let col = 0; col < w; col += step) {
                const x = Math.floor((col / w) * cw);
                for (let row = 0; row < h; row += 2) {
                    const idx = (row * w + col) * 4;
                    const luma = data[idx] * 0.2126 + data[idx + 1] * 0.7152 + data[idx + 2] * 0.0722;
                    const y = ch - (luma / 255) * ch;
                    const bright = Math.floor(40 + luma * 0.6);
                    ctx.fillStyle = `rgb(${bright}, ${Math.floor(bright * 1.4)}, ${bright})`;
                    ctx.fillRect(x, y, 1, 1);
                }
            }
            ctx.globalAlpha = 1.0;
            ctx.fillStyle = 'rgba(80,255,80,0.8)';
            ctx.font = '10px monospace';
            ctx.fillText('LUMA', 4, 12);
        }
    }

    _drawVectorscope(ctx, data, w, h, cw, ch) {
        ctx.fillStyle = '#050508';
        ctx.fillRect(0, 0, cw, ch);

        const cx = cw / 2;
        const cy = ch / 2;
        const rad = Math.min(cx, cy) - 8;

        // Graticule rings
        ctx.strokeStyle = 'rgba(255,255,255,0.06)'; ctx.lineWidth = 1;
        [0.25, 0.5, 0.75, 1.0].forEach(r => {
            ctx.beginPath(); ctx.arc(cx, cy, rad * r, 0, Math.PI * 2); ctx.stroke();
        });

        // Crosshair
        ctx.beginPath(); ctx.moveTo(cx, cy - rad); ctx.lineTo(cx, cy + rad); ctx.stroke();
        ctx.beginPath(); ctx.moveTo(cx - rad, cy); ctx.lineTo(cx + rad, cy); ctx.stroke();

        // Rec.709 color targets
        const targets = [
            { a: 103, c: '#f33', l: 'R' },
            { a: 167, c: '#ff0', l: 'Yl' },
            { a: 241, c: '#0f0', l: 'G' },
            { a: 283, c: '#0ff', l: 'Cy' },
            { a: 347, c: '#33f', l: 'B' },
            { a: 61, c: '#f0f', l: 'Mg' },
        ];
        targets.forEach(t => {
            const ang = (t.a - 90) * Math.PI / 180;
            const tx = cx + Math.cos(ang) * rad * 0.75;
            const ty = cy + Math.sin(ang) * rad * 0.75;
            ctx.fillStyle = t.c;
            ctx.beginPath(); ctx.arc(tx, ty, 3, 0, Math.PI * 2); ctx.fill();
        });

        // Skin Tone Indicator (I-Line)
        ctx.strokeStyle = 'rgba(255, 140, 100, 0.3)';
        ctx.lineWidth = 1.5;
        ctx.setLineDash([3, 3]);
        const iLineAng = (123 - 90) * Math.PI / 180;
        ctx.beginPath();
        ctx.moveTo(cx, cy);
        ctx.lineTo(cx + Math.cos(iLineAng) * rad * 0.9, cy + Math.sin(iLineAng) * rad * 0.9);
        ctx.stroke();
        ctx.setLineDash([]);

        // Plot pixels in gorgeous premium Ocean Cyan with heat map density
        const accW = 128;
        const accH = 128;
        const acc = new Uint32Array(accW * accH);

        const step = Math.max(1, Math.floor(data.length / 4 / 15000));
        for (let i = 0; i < data.length; i += 4 * step) {
            const r = data[i] / 255;
            const g = data[i + 1] / 255;
            const b = data[i + 2] / 255;
            const yVal = r * 0.2126 + g * 0.7152 + b * 0.0722;
            const u = (b - yVal) * 0.492;
            const v = (r - yVal) * 0.877;

            const ux = Math.min(accW - 1, Math.max(0, Math.floor((u * 0.5 + 0.5) * accW)));
            const vy = Math.min(accH - 1, Math.max(0, Math.floor((v * 0.5 + 0.5) * accH)));
            acc[vy * accW + ux]++;
        }

        let maxAcc = 1;
        for (let i = 0; i < acc.length; i++) maxAcc = Math.max(maxAcc, acc[i]);

        // Draw heat map
        ctx.globalAlpha = 0.85;
        for (let ay = 0; ay < accH; ay++) {
            for (let ax = 0; ax < accW; ax++) {
                const val = acc[ay * accW + ax];
                if (val === 0) continue;
                const intensity = Math.min(1.0, val / (maxAcc * 0.1));
                const u = (ax / accW - 0.5) * 2;
                const v = (ay / accH - 0.5) * 2;

                const px = cx + u * rad * 1.1;
                const py = cy + v * rad * 1.1; // note + instead of - because vy is 0..accH matching top..bottom

                ctx.fillStyle = `rgba(0, 189, 255, ${intensity * 0.95})`; // Gorgeous Ocean Cyan
                ctx.fillRect(px, py, 1.5, 1.5);
            }
        }
        ctx.globalAlpha = 1.0;
    }

    // ── Pixel readback for export ──────────────────────────────────────────

    /**
     * Read back float pixels.
     *
     * Contract matches the WebGL renderer's deliberately: a **synchronous**
     * `{ data, width, height, graded }` or `null`. It used to return a
     * `Promise<Float32Array>` -- a different type, a different shape, and no
     * indication that the pixels were ungraded -- while WebGL returned a
     * synchronous wrapper around a fully graded composite.
     *
     * The two consumers held contradictory expectations of the same method.
     * `_renderScopeFromPixels` used `.then()`, so it worked; the 32-bit EXR
     * export at `radiance_viewer.js` read `result.data` straight off the
     * Promise, got `undefined`, and reported "[Export] EXR encoding failed" --
     * an error blaming the encoder for a backend contract mismatch. The
     * `!this.useWebGL` guard did not catch it because `_tryWebGPUUpgrade` sets
     * `useWebGL = true`.
     *
     * `graded` is part of the contract, not decoration: this backend returns
     * the ungraded scene-linear source, so anything writing a "graded" file
     * has to know that and say so rather than silently export the wrong image.
     */
    readPixelsFloat32(w, h, lutStrength) {
        if (!this._lastSourcePixels || !this.imageWidth || !this.imageHeight) return null;

        // CPU-side nearest resample of the uploaded scene-linear source. This is
        // intentionally conservative: it keeps WebGPU scopes/export from relying
        // on optional float32 render-target readback support that is not present
        // on every browser/driver combination.
        const src = this._lastSourcePixels;
        const srcW = this.imageWidth;
        const srcH = this.imageHeight;
        const out = new Float32Array(w * h * 4);
        for (let y = 0; y < h; y++) {
            const sy = Math.min(srcH - 1, Math.floor((y / Math.max(1, h - 1)) * (srcH - 1)));
            for (let x = 0; x < w; x++) {
                const sx = Math.min(srcW - 1, Math.floor((x / Math.max(1, w - 1)) * (srcW - 1)));
                const si = (sy * srcW + sx) * 4;
                const di = (y * w + x) * 4;
                out[di] = src[si];
                out[di + 1] = src[si + 1];
                out[di + 2] = src[si + 2];
                out[di + 3] = src[si + 3];
            }
        }
        return { data: out, width: w, height: h, graded: false };
    }

    // ── Reference shelf ────────────────────────────────────────────────────

    grabReferenceStill(slot) {
        // Phase 2
    }
    swapReferenceShelf(slot) { }
    clearReferenceShelf() {
        this.referenceShelf = [];
    }
    clearFrameCache() { }

    // ── Helpers ────────────────────────────────────────────────────────────

    _halfToFloat(h) {
        const s = (h >> 15) & 1;
        const e = (h >> 10) & 31;
        const m = h & 1023;
        if (e === 0) return (s ? -1 : 1) * Math.pow(2, -14) * (m / 1024);
        if (e === 31) return m ? NaN : (s ? -Infinity : Infinity);
        return (s ? -1 : 1) * Math.pow(2, e - 15) * (1 + m / 1024);
    }

    _floatToHalfBits(v) {
        if (!Number.isFinite(v)) return v < 0 ? 0xfc00 : (Number.isNaN(v) ? 0x7e00 : 0x7c00);
        const f32 = new Float32Array(1);
        const u32 = new Uint32Array(f32.buffer);
        f32[0] = v;
        const x = u32[0];
        const sign = (x >>> 16) & 0x8000;
        let mantissa = x & 0x007fffff;
        let exponent = ((x >>> 23) & 0xff) - 127 + 15;
        if (exponent <= 0) {
            if (exponent < -10) return sign;
            mantissa = (mantissa | 0x00800000) >>> (1 - exponent);
            return sign | ((mantissa + 0x00001000) >>> 13);
        }
        if (exponent >= 31) return sign | 0x7bff;
        return sign | (exponent << 10) | ((mantissa + 0x00001000) >>> 13);
    }

    _float32ToHalfTextureData(data, width, height) {
        const total = width * height * 4;
        const out = new Uint16Array(total);
        for (let i = 0; i < total; i++) out[i] = this._floatToHalfBits(data[i] ?? 0);
        return out;
    }

    _setImageTexture(texture, isLinear) {
        if (this.textures.image && this.textures.image !== texture) {
            this.textures.image.destroy();
        }
        this.textures.image = texture;
        this.textureView = texture ? texture.createView() : null;
        this.isLinearTexture = isLinear;
    }

    _getIdentityTextureView() {
        if (!this._identityTextureView) {
            const data = new Float32Array(256 * 4);
            for (let i = 0; i < 256; i++) {
                data[i * 4] = i / 255;
                data[i * 4 + 1] = i / 255;
                data[i * 4 + 2] = i / 255;
                data[i * 4 + 3] = 1.0;
            }
            const format = this._textureFloatFormat || 'rgba16float';
            const uploadData = format === 'rgba16float'
                ? this._float32ToHalfTextureData(data, 256, 1)
                : data;
            const tex = this.device.createTexture({
                size: [256],
                dimension: '1d',
                format,
                usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
            });
            this.device.queue.writeTexture(
                { texture: tex },
                uploadData,
                { bytesPerRow: 256 * (format === 'rgba32float' ? 16 : 8) },
                { width: 256 },
            );
            this._identityTextureView = tex.createView({ dimension: '1d' });
        }
        return this._identityTextureView;
    }

    _getIdentityLutTextureView() {
        if (!this._identityLutTextureView) {
            const size = 33;
            const data = new Float32Array(size * size * size * 4);
            for (let z = 0; z < size; z++) {
                for (let y = 0; y < size; y++) {
                    for (let x = 0; x < size; x++) {
                        const idx = (z * size * size + y * size + x) * 4;
                        data[idx] = x / (size - 1);
                        data[idx + 1] = y / (size - 1);
                        data[idx + 2] = z / (size - 1);
                        data[idx + 3] = 1.0;
                    }
                }
            }
            const format = this._textureFloatFormat || 'rgba16float';
            const uploadData = format === 'rgba16float'
                ? this._float32ToHalfTextureData(data, size, size * size)
                : data;
            const tex = this.device.createTexture({
                size: [size, size, size],
                dimension: '3d',
                format,
                usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
            });
            this.device.queue.writeTexture(
                { texture: tex },
                uploadData,
                { bytesPerRow: size * (format === 'rgba32float' ? 16 : 8), rowsPerImage: size },
                { width: size, height: size, depthOrArrayLayers: size },
            );
            this._identityLutTextureView = tex.createView({ dimension: '3d' });
        }
        return this._identityLutTextureView;
    }

    _writeUniforms(lutStrength) {
        // Pack all grading params into the uniform buffer
        // Using a helper to write structure-of-arrays style
        const buf = new ArrayBuffer(512);
        const f32 = new Float32Array(buf);
        const i32 = new Int32Array(buf);
        let o = 0;

        f32[o++] = this.exposure;
        f32[o++] = this.gradingGamma[0]; f32[o++] = this.gradingGamma[1]; f32[o++] = this.gradingGamma[2];
        f32[o++] = this.saturation;
        f32[o++] = this.lift[0]; f32[o++] = this.lift[1]; f32[o++] = this.lift[2];
        f32[o++] = this.gradingGamma[0]; f32[o++] = this.gradingGamma[1]; f32[o++] = this.gradingGamma[2];
        f32[o++] = this.gain[0]; f32[o++] = this.gain[1]; f32[o++] = this.gain[2];
        f32[o++] = this.offset[0]; f32[o++] = this.offset[1]; f32[o++] = this.offset[2];
        f32[o++] = this.temperature;
        f32[o++] = this.tint;
        f32[o++] = this.contrast;
        f32[o++] = this.pivot;
        f32[o++] = this.colorBoost;
        f32[o++] = this.shadows;
        f32[o++] = this.highlights;
        f32[o++] = this.midDetail;
        f32[o++] = this.hueShift;
        f32[o++] = this.lumaMix;
        i32[o++] = this.colorScience;
        f32[o++] = this.logShadow[0]; f32[o++] = this.logShadow[1]; f32[o++] = this.logShadow[2];
        f32[o++] = this.logMidtone[0]; f32[o++] = this.logMidtone[1]; f32[o++] = this.logMidtone[2];
        f32[o++] = this.logHighlight[0]; f32[o++] = this.logHighlight[1]; f32[o++] = this.logHighlight[2];
        i32[o] = this.printerLightsR; o++; i32[o] = this.printerLightsG; o++; i32[o] = this.printerLightsB; o++;
        f32[o++] = this.softClip;
        f32[o++] = this.displayLutStrength;
        i32[o] = this.channelMode; o++;
        f32[o++] = this.grainAmount;
        f32[o++] = this.grainSize;
        f32[o++] = this.grainColor;
        i32[o] = this.grainAnimate ? 1 : 0; o++;
        f32[o++] = this.bloom;
        f32[o++] = this.halation;
        f32[o++] = this.diffusion;
        f32[o++] = this.lensDistortion;
        f32[o++] = this.lensFringe;
        f32[o++] = this.vignetteIntensity;
        f32[o++] = this.vignetteFalloff;
        i32[o] = this.dofEnabled ? 1 : 0; o++;
        f32[o++] = this.focusDistance;
        f32[o++] = this.aperture;
        i32[o] = this.apertureBlades; o++;
        f32[o++] = this.apertureRotation;
        f32[o++] = this.apertureAnamorphic;
        f32[o++] = this.zebraThreshold;
        i32[o] = this.falseColor ? 1 : 0; o++;
        i32[o] = this.zebra ? 1 : 0; o++;
        i32[o] = this.gamutWarning ? 1 : 0; o++;
        i32[o] = this.clippingMonitor ? 1 : 0; o++;
        i32[o] = this.focusPeaking ? 1 : 0; o++;
        f32[o++] = this.focusPeakThreshold;
        f32[o++] = this.denoise;
        i32[o] = this.frame; o++;
        f32[o++] = this.time;
        f32[o++] = lutStrength || 1.0;
        i32[o] = this.displayLutMode > 0 ? 1 : 0; o++; // lut_enabled
        i32[o] = this.lutIsDisplayTransform ? 1 : 0; o++;
        i32[o] = this.displayLutMode; o++;
        i32[o] = this.inputLutMode; o++;
        i32[o] = this.showDepth ? 1 : 0; o++;
        f32[o++] = this.curveMix;
        f32[o++] = this.secondaryCurveMix;
        f32[o++] = this.panX || 0;
        f32[o++] = this.panY || 0;
        f32[o++] = this.zoom || 1;
        f32[o++] = this.wipe ?? 0.5;
        i32[o] = this.wipeEnabled ? 1 : 0; o++;

        this.device.queue.writeBuffer(this.uniformBuffer, 0, f32.buffer, 0, Math.min(o * 4, 512));
    }

    getPipelineInfo() {
        return { api: 'webgpu', precision: 'f32', adapter: 'default' };
    }

    // ── Feature parity check ──────────────────────────────────────────────────
    // Returns true only when ALL viewer features have matching WebGPU shader
    // implementations. Until then, the viewer stays on WebGL for safety.
    // 3.5.0: false, and it was never true. Known gaps (KNOWN_ISSUES.md, the
    // one WebGPU entry): 8-bit input is not sRGB-decoded, the 3D LUT path
    // reads .cube data with the wrong stride, curves use the red curve for
    // every channel, the hue matrix is transposed, bloom samples outside its
    // target, and OCIO, masks, qualifiers, heatmap, gamut/clip warnings, the
    // scope signal, viewer f-stop and the scene-linear EXR export are absent.
    // WebGL is the default and the tested backend.
    static FEATURE_PARITY = false;
    isFeatureComplete() {
        return RadianceWebGPURenderer.FEATURE_PARITY;
    }

    // ── Missing method stubs (WebGL API compatibility) ────────────────────────
    // These are called by the viewer but not yet implemented in the WebGPU
    // shader pipeline. They safely no-op until the WGSL shaders are completed.

    setLinearFalseColor(v) { this.linearFalseColor = v; }
    setLensDistortionK2(v) { this.lensDistortionK2 = v; }
    setStreakThreshold(v) { this.streakThreshold = v; }
    setStreakLength(v) { this.streakLength = v; }
    setBilateralSigma(sigmaD, sigmaR) { this.bilateralSigmaD = sigmaD; this.bilateralSigmaR = sigmaR; }
    setBilateralHalfRes(enabled) { this.bilateralHalfRes = enabled; }
    setCurveSlope(r, g, b) { this.curveSlope = [r, g, b]; }
    setAnamorphicStreaks(v) { this.anamorphicStreaks = v; }
    setBokehPhysics(bias, soap, vig) { this.bokehHighlightBias = bias; this.bokehSoapBubble = soap; this.bokehOpticalVig = vig; }
    renderWaveform(targetCanvas, parade) { this.renderScope('waveform', targetCanvas, null, true, parade, false); }
    initWipeDragging() { /* stub: interactive wipe not yet in WebGPU */ }
    readPixels() { const r = this.readPixelsFloat32(1, 1, 1); return r; }
    updateReferenceStill() { /* stub */ }
    parseCubeFile(cubeText) { /* stub: .cube parsing not needed in WebGPU — use loadLUT() with pre-parsed data */ }
    static initDisplayP3(canvas) { return { isP3: false, isHDR: false }; }
}

// ── WebGPU availability check ──────────────────────────────────────────────

RadianceWebGPURenderer.isAvailable = function () {
    return typeof navigator !== 'undefined' && 'gpu' in navigator;
};

export { RadianceWebGPURenderer };
