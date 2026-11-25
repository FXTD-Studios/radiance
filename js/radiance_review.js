/**
 * Review marks: strokes drawn on a frame, notes and bookmarks.
 *
 * Pure state and coordinate maths, no DOM. Marks never touch pixels: they live
 * beside the image and are drawn over it.
 *
 * Positions are stored as image UV (0..1 across the picture), not canvas
 * pixels, so a mark stays on the same feature through zoom, pan, window resize
 * and a change of resolution. Each mark belongs to one frame and is shown only
 * on that frame.
 *
 * Saved form is versioned JSON with hard caps, so a workflow cannot grow
 * without bound and a corrupt or hostile value loads as "no marks".
 */

export const REVIEW_VERSION = 1;
export const LIMITS = Object.freeze({
    // The saved JSON rides in the workflow, and so in every PNG ComfyUI writes
    // with it: keep it small. 1e-4 of the frame is under half a pixel at 4K.
    maxMarks: 500, maxPointsPerStroke: 200, maxNoteChars: 500, maxLabelChars: 80, maxJsonChars: 150000,
});

const KINDS = ['stroke', 'note', 'bookmark'];
const clamp01 = (v) => Math.min(1, Math.max(0, v));
const num = (v) => typeof v === 'number' && Number.isFinite(v);
const round4 = (v) => Math.round(v * 1e4) / 1e4;

/** view = { zoom, panX, panY, width, height }: canvas = uv * size * zoom + pan. */
export function canvasToUV(x, y, view) {
    const { zoom, panX, panY, width, height } = view || {};
    if (![zoom, panX, panY, width, height].every(num) || zoom <= 0 || width <= 0 || height <= 0) return null;
    return { u: (x - panX) / (width * zoom), v: (y - panY) / (height * zoom) };
}

export function uvToCanvas(u, v, view) {
    const { zoom, panX, panY, width, height } = view || {};
    if (![zoom, panX, panY, width, height].every(num)) return null;
    return { x: u * width * zoom + panX, y: v * height * zoom + panY };
}

/** Keep at most `max` points, always the first and the last, evenly spaced between. */
export function thinPoints(pts, max = LIMITS.maxPointsPerStroke) {
    if (pts.length <= max) return pts;
    const out = [];
    for (let i = 0; i < max; i++) out.push(pts[Math.round((i * (pts.length - 1)) / (max - 1))]);
    return out;
}

export class ReviewStore {
    constructor() {
        this.marks = [];
        this._seq = 0;
    }

    _id() { return `m${++this._seq}`; }

    _add(mark) {
        if (this.marks.length >= LIMITS.maxMarks) return null;       // refuse, never silently drop an older mark
        this.marks.push(mark);
        return mark;
    }

    /** A stroke of UV points on a frame. Needs two points; returns the mark or null. */
    addStroke(frame, pts, color = '#ffee44') {
        if (!Number.isInteger(frame) || frame < 0 || !Array.isArray(pts)) return null;
        const clean = pts.filter((p) => p && num(p.u) && num(p.v)).map((p) => ({ u: round4(p.u), v: round4(p.v) }));
        if (clean.length < 2) return null;
        return this._add({ id: this._id(), kind: 'stroke', frame, color: String(color).slice(0, 16), pts: thinPoints(clean) });
    }

    addNote(frame, u, v, text) {
        if (!Number.isInteger(frame) || frame < 0 || !num(u) || !num(v)) return null;
        const t = String(text ?? '').trim().slice(0, LIMITS.maxNoteChars);
        if (!t) return null;
        return this._add({ id: this._id(), kind: 'note', frame, u: clamp01(u), v: clamp01(v), text: t });
    }

    /** One bookmark per frame: marking a bookmarked frame updates its label. */
    addBookmark(frame, label = '') {
        if (!Number.isInteger(frame) || frame < 0) return null;
        const l = String(label).trim().slice(0, LIMITS.maxLabelChars);
        const existing = this.marks.find((m) => m.kind === 'bookmark' && m.frame === frame);
        if (existing) { existing.label = l; return existing; }
        return this._add({ id: this._id(), kind: 'bookmark', frame, label: l });
    }

    forFrame(frame, kind) {
        return this.marks.filter((m) => m.frame === frame && (!kind || m.kind === kind));
    }

    bookmarks() {
        return this.marks.filter((m) => m.kind === 'bookmark').sort((a, b) => a.frame - b.frame);
    }

    /** The next bookmarked frame after (dir 1) or before (dir -1) `frame`, wrapping. null if none. */
    stepBookmark(frame, dir = 1) {
        const b = this.bookmarks();
        if (!b.length) return null;
        const next = dir >= 0 ? b.find((m) => m.frame > frame) : [...b].reverse().find((m) => m.frame < frame);
        return (next || (dir >= 0 ? b[0] : b[b.length - 1])).frame;
    }

    remove(id) {
        const n = this.marks.length;
        this.marks = this.marks.filter((m) => m.id !== id);
        return this.marks.length !== n;
    }

    clearFrame(frame, kind) {
        const n = this.marks.length;
        this.marks = this.marks.filter((m) => !(m.frame === frame && (!kind || m.kind === kind)));
        return n - this.marks.length;
    }

    clear(kind) {
        const n = this.marks.length;
        this.marks = kind ? this.marks.filter((m) => m.kind !== kind) : [];
        return n - this.marks.length;
    }

    /** Strokes on `frame` as canvas-pixel polylines for the current view. */
    canvasStrokes(frame, view, extra = []) {
        const out = [];
        for (const m of [...this.forFrame(frame, 'stroke'), ...extra]) {
            const pts = m.pts.map((p) => uvToCanvas(p.u, p.v, view)).filter(Boolean);
            if (pts.length >= 2) out.push({ pts, color: m.color });
        }
        return out;
    }

    toJSON() {
        return { version: REVIEW_VERSION, marks: this.marks };
    }

    /** Serialised form, or null if it would exceed the size cap. */
    serialize() {
        const s = JSON.stringify(this.toJSON());
        return s.length <= LIMITS.maxJsonChars ? s : null;
    }

    /** Never throws. Anything unreadable, unknown-version or oversized loads as an empty store. */
    static parse(input) {
        const store = new ReviewStore();
        try {
            if (typeof input === 'string' && input.length > LIMITS.maxJsonChars) return store;
            const data = typeof input === 'string' ? JSON.parse(input) : input;
            if (!data || data.version !== REVIEW_VERSION || !Array.isArray(data.marks)) return store;
            for (const m of data.marks.slice(0, LIMITS.maxMarks)) {
                if (!m || !KINDS.includes(m.kind)) continue;
                if (m.kind === 'stroke') store.addStroke(m.frame, m.pts, m.color);
                else if (m.kind === 'note') store.addNote(m.frame, m.u, m.v, m.text);
                else store.addBookmark(m.frame, m.label);
            }
        } catch { /* corrupt value: start empty */ }
        return store;
    }
}
