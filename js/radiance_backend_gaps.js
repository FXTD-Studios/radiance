/**
 * What the opt-in WebGPU renderer does not do, in one list. The viewer's
 * badge, the Settings message and the panel notices are all built from it,
 * and backend_parity.test.mjs checks KNOWN_ISSUES.md names every entry, so
 * the list a person reads and the shipped behaviour cannot drift apart.
 *
 * Pure data, no DOM.
 */
export const WEBGPU_GAPS = Object.freeze([
    { id: 'masks', text: 'power-window masks', known: /mask/i },
    { id: 'qualifiers', text: 'HSL qualifiers', known: /qualifier/i },
    { id: 'ocio', text: 'OpenColorIO', known: /OCIO|OpenColorIO/ },
    { id: 'heatmap', text: 'the HDR heatmap', known: /heatmap/i },
    { id: 'warnings', text: 'gamut and clip warnings', known: /gamut\/clip\s+warnings/i },
    { id: 'viewerFstop', text: 'the viewer f-stop and gamma keys', known: /viewer\s+f-stop/i },
    { id: 'srgbDecode', text: 'sRGB decoding of 8-bit sources', known: /sRGB\s+decode\s+for\s+8-bit/i },
    { id: 'curves', text: 'per-channel curves (red is applied to all)', known: /red\s+curve\s+applied\s+to\s+every\s+channel/i },
    { id: 'hue', text: 'the hue rotation (its matrix is transposed)', known: /transposed\s+hue/i },
    { id: 'bloom', text: 'bloom registration', known: /misregistered\s+bloom/i },
    { id: 'scopes', text: 'float scopes (they read the 8-bit preview at 256 px, overlays included)', known: /scope\s+signal/i },
    { id: 'compare', text: 'compare B, Difference and Blink (wipe only)', known: /compare shows only\s+wipe/i },
    { id: 'gradedExr', text: 'the scene-linear graded EXR', known: /scene-linear\s+graded\s+EXR/i },
]);

/** One sentence naming every gap, for a tooltip or a log line. */
export function webgpuGapSentence() {
    const items = WEBGPU_GAPS.map((g) => g.text);
    return `WebGPU is experimental here. Not implemented or not correct on it: ${items.slice(0, -1).join(', ')} and ${items[items.length - 1]}. WebGL is the default and the tested path (Settings > Renderer).`;
}
