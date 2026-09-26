/**
 * BT.709 vectorscope geometry (3.5.0), shared by the GPU and CPU scopes.
 *
 * Cb = (B' - Y') / 1.8556, Cr = (R' - Y') / 1.5748 on the ENCODED signal, with
 * Y' = 0.2126 R' + 0.7152 G' + 0.0722 B' (ITU-R BT.709). Cb runs right, Cr up;
 * |Cb|, |Cr| <= 0.5. The targets are computed from 75 % and 100 % colour bars
 * with the same maths the trace uses, so they line up by construction. The old
 * scope placed targets at hard-coded angles through a screen-space rotation
 * that put red near -13 degrees (103 is correct) and used PAL U/V weights.
 */

/** Fraction of the half-size the 0.5 chroma circle reaches. */
export const SCOPE_RADIUS = 0.9;

export function cbcr(r, g, b) {
    const y = 0.2126 * r + 0.7152 * g + 0.0722 * b;
    return [(b - y) / 1.8556, (r - y) / 1.5748];
}

/** Canvas position of (Cb, Cr) in a w x h scope. */
export function toCanvas(cb, cr, w, h) {
    const half = Math.min(w, h) / 2;
    return [w / 2 + (cb / 0.5) * half * SCOPE_RADIUS, h / 2 - (cr / 0.5) * half * SCOPE_RADIUS];
}

const BARS = [
    ['R', [1, 0, 0], '#ff4444'], ['Mg', [1, 0, 1], '#ff44ff'], ['B', [0, 0, 1], '#4466ff'],
    ['Cy', [0, 1, 1], '#44ffff'], ['G', [0, 1, 0], '#44ff44'], ['Yl', [1, 1, 0], '#ffff44'],
];

/** Colour-bar targets at 75 % and 100 %. */
export function targets(w, h) {
    return BARS.map(([label, rgb, colour]) => ({
        label, colour,
        p75: toCanvas(...cbcr(...rgb.map((v) => v * 0.75)), w, h),
        p100: toCanvas(...cbcr(...rgb), w, h),
    }));
}

/** The skin-tone (I) line: 123 degrees counter-clockwise from +Cb. */
export function skinLine(w, h) {
    const a = 123 * Math.PI / 180;
    const half = Math.min(w, h) / 2 * SCOPE_RADIUS;
    return [[w / 2, h / 2], [w / 2 + Math.cos(a) * half, h / 2 - Math.sin(a) * half]];
}

/** Draw rings, axes, targets and the skin line onto a 2D context. */
export function drawGraticule(ctx, w, h, { labels = true, lineWidth = 1 } = {}) {
    const cx = w / 2, cy = h / 2, rad = Math.min(w, h) / 2 * SCOPE_RADIUS;
    ctx.save();
    ctx.strokeStyle = 'rgba(255,255,255,0.10)'; ctx.lineWidth = lineWidth;
    [0.25, 0.5, 0.75, 1.0].forEach((r) => { ctx.beginPath(); ctx.arc(cx, cy, rad * r, 0, Math.PI * 2); ctx.stroke(); });
    ctx.beginPath(); ctx.moveTo(cx - rad, cy); ctx.lineTo(cx + rad, cy);
    ctx.moveTo(cx, cy - rad); ctx.lineTo(cx, cy + rad); ctx.stroke();
    const [s0, s1] = skinLine(w, h);
    ctx.strokeStyle = 'rgba(255,150,110,0.55)'; ctx.setLineDash([4, 4]);
    ctx.beginPath(); ctx.moveTo(...s0); ctx.lineTo(...s1); ctx.stroke(); ctx.setLineDash([]);
    const box = Math.max(4, Math.round(Math.min(w, h) / 60));
    targets(w, h).forEach((t) => {
        ctx.strokeStyle = t.colour;
        ctx.strokeRect(t.p75[0] - box, t.p75[1] - box, box * 2, box * 2);
        ctx.beginPath(); ctx.arc(t.p100[0], t.p100[1], box * 0.5, 0, Math.PI * 2); ctx.stroke();
        if (labels) {
            ctx.fillStyle = t.colour;
            ctx.font = `${Math.max(9, Math.round(Math.min(w, h) / 30))}px monospace`;
            ctx.fillText(t.label, t.p75[0] + box + 2, t.p75[1] + box);
        }
    });
    ctx.restore();
}
