/**
 * FLUX 3 Image layout boxes drawn in the Viewer.
 *
 * Pure data and functions, no DOM, so the rules are testable in Node.
 * radiance_viewer.js draws the boxes and holds the panel.
 *
 * A box is { x, y, w, h, desc } in image UV (0..1, top-left origin), so it
 * follows zoom and pan and survives a new run at another size. The prompt gets
 * the rows ComfyUI's "FLUX 3 Image Layout Prompt" node writes:
 *   [{"id": "obj_1", "bbox": [ymin, xmin, ymax, xmax], "desc": "..."}]
 * with bbox on a 0..1000 grid (comfy_extras/nodes_bounding_boxes.py _norm_bbox),
 * cited in the prompt as <obj_1>, <obj_2>...
 */

export const LAYOUT_LIMITS = Object.freeze({ maxBoxes: 32, minSize: 0.01, maxDesc: 500 });

const num = (v) => typeof v === 'number' && Number.isFinite(v);
const clamp01 = (v) => Math.min(1, Math.max(0, v));
const round4 = (v) => Math.round(v * 1e4) / 1e4;

/** A box from two corners in UV, clamped to the picture. null when too small. */
export function boxFromCorners(a, b, desc = '') {
    if (!a || !b || ![a.u, a.v, b.u, b.v].every(num)) return null;
    const x0 = clamp01(Math.min(a.u, b.u)), x1 = clamp01(Math.max(a.u, b.u));
    const y0 = clamp01(Math.min(a.v, b.v)), y1 = clamp01(Math.max(a.v, b.v));
    if (x1 - x0 < LAYOUT_LIMITS.minSize || y1 - y0 < LAYOUT_LIMITS.minSize) return null;
    return { x: round4(x0), y: round4(y0), w: round4(x1 - x0), h: round4(y1 - y0), desc: String(desc || '') };
}

function cleanBox(b) {
    if (!b || ![b.x, b.y, b.w, b.h].every(num)) return null;
    return boxFromCorners({ u: b.x, v: b.y }, { u: b.x + b.w, v: b.y + b.h },
        String(b.desc ?? '').slice(0, LAYOUT_LIMITS.maxDesc));
}

/** Boxes saved on the node (radiance_viewer_layout). Bad entries are dropped. */
export function parseStoredBoxes(json) {
    let list;
    try { list = typeof json === 'string' ? JSON.parse(json) : json; } catch { return []; }
    if (!Array.isArray(list)) return [];
    return list.map(cleanBox).filter(Boolean).slice(0, LAYOUT_LIMITS.maxBoxes);
}

export function serializeBoxes(boxes) {
    return JSON.stringify((boxes || []).map(cleanBox).filter(Boolean));
}

const grid = (v) => Math.max(0, Math.min(1000, Math.round(v * 1000)));

/** The prompt's layout rows, in drawing order (draw the background first). */
export function layoutRows(boxes) {
    return (boxes || []).map(cleanBox).filter(Boolean).map((b, i) => ({
        id: `obj_${i + 1}`,
        bbox: [grid(b.y), grid(b.x), grid(b.y + b.h), grid(b.x + b.w)],
        desc: b.desc.trim(),
    }));
}

/** The layout table as the Prompt's layout input takes it; '' for no boxes. */
export function layoutJSON(boxes) {
    const rows = layoutRows(boxes);
    return rows.length ? JSON.stringify(rows) : '';
}

/** Boxes back from a layout table (the Prompt's layout value). [] when it is not one. */
export function boxesFromLayout(json) {
    let rows;
    try { rows = JSON.parse(json); } catch { return []; }
    if (!Array.isArray(rows)) return [];
    const out = [];
    for (const r of rows) {
        const bb = r?.bbox;
        if (!Array.isArray(bb) || bb.length !== 4 || !bb.every(num)) continue;
        const [ymin, xmin, ymax, xmax] = bb.map((v) => v / 1000);
        const box = boxFromCorners({ u: xmin, v: ymin }, { u: xmax, v: ymax }, r.desc || '');
        if (box) out.push(box);
    }
    return out.slice(0, LAYOUT_LIMITS.maxBoxes);
}

/** Rows that have no description: the model gets a box with nothing to put in it. */
export function undescribed(boxes) {
    return layoutRows(boxes).filter((r) => !r.desc).map((r) => r.id);
}
