/**
 * Bounded decode window for viewer sequence playback.
 *
 * DEFECT this closes (viewer tab dies while loading a long shot): the sequence
 * loader used to fire one `new Image()` and one `fetch()` per frame the moment
 * a result arrived, with no concurrency limit, and it stored every decoded
 * buffer in `frameHDRData[]` and `frameImages[]`. Those two arrays were reset
 * only when a new generation started and were never evicted from, so the
 * retained pixel data grew with the length of the sequence.
 *
 * The GPU texture LRU in radiance_webgl.js is correctly bounded, and that is
 * exactly what masked this: the textures were capped, the source pixel arrays
 * behind them were not. A 300-frame 1080p RGBA shot therefore opened 300
 * sockets at once and held 300 x 33 MB, roughly 10 GB of Float32Array, in one
 * browser tab. 10,000 frames is roughly 330 GB, so the tab died during load and
 * the bounded texture cache never got a chance to help.
 *
 * Frames are already on disk, written by the node, so the viewer's job is to
 * page them rather than to hold them. This class does that: a fixed number of
 * decoded frames is retained around the playhead, frames are fetched on demand
 * as the user scrubs, everything that falls outside the window is evicted, and
 * at most `concurrency` fetches are ever in flight.
 *
 * Both bounds are explicit numbers rather than a consequence of the sequence
 * length. `windowSize` caps the frame count; `maxBytes` caps the retained
 * pixel data, which is the bound that matters at 4K where a single RGBA fp32
 * frame is 141 MB. Whichever binds first wins, and the playhead frame is never
 * evicted.
 */

/** Decoded frames retained around the playhead. */
export const DEFAULT_FRAME_WINDOW = 16;

/** Retained decoded pixel bytes. 16 x 1080p RGBA fp32 is ~531 MB. */
export const DEFAULT_FRAME_WINDOW_BYTES = 768 * 1024 * 1024;

/** Simultaneous sidecar fetches. The old code opened one socket per frame. */
export const DEFAULT_FETCH_CONCURRENCY = 4;

/**
 * Default size accounting for a decoded frame payload.
 *
 * Counts the pixel arrays only. An `Image` holds its decoded bitmap outside the
 * JS heap and reports no byteLength, so it is charged at its pixel footprint
 * when the dimensions are known.
 */
export function measureFramePayload(payload) {
    if (!payload) return 0;
    let bytes = 0;
    const hdr = payload.hdr || payload;
    if (hdr) {
        if (hdr.data && typeof hdr.data.byteLength === 'number') bytes += hdr.data.byteLength;
        if (hdr.fp16data && typeof hdr.fp16data.byteLength === 'number') bytes += hdr.fp16data.byteLength;
    }
    for (const key of ['img', 'bracketLow', 'bracketHigh']) {
        const img = payload[key];
        if (img && img.width && img.height) bytes += img.width * img.height * 4;
    }
    return bytes;
}

export class RadianceFrameWindow {
    /**
     * @param {object} opts
     * @param {number} [opts.windowSize]  max decoded frames retained
     * @param {number} [opts.maxBytes]    max retained decoded pixel bytes
     * @param {number} [opts.concurrency] max simultaneous loads
     * @param {(entry:any, idx:number) => Promise<any>} [opts.load]
     * @param {(idx:number, payload:any) => void} [opts.onReady]
     * @param {(idx:number, payload:any) => void} [opts.onEvict]
     * @param {(err:any, idx:number) => void} [opts.onError]
     * @param {(payload:any) => number} [opts.measure]
     */
    constructor(opts = {}) {
        this.windowSize = Math.max(1, Math.floor(opts.windowSize) || DEFAULT_FRAME_WINDOW);
        this.maxBytes = Math.max(1, Math.floor(opts.maxBytes) || DEFAULT_FRAME_WINDOW_BYTES);
        this.concurrency = Math.max(1, Math.floor(opts.concurrency) || DEFAULT_FETCH_CONCURRENCY);

        this._load = typeof opts.load === 'function' ? opts.load : () => Promise.resolve(null);
        this._onReady = typeof opts.onReady === 'function' ? opts.onReady : null;
        this._onEvict = typeof opts.onEvict === 'function' ? opts.onEvict : null;
        this._onError = typeof opts.onError === 'function' ? opts.onError : null;
        this._measure = typeof opts.measure === 'function' ? opts.measure : measureFramePayload;

        this.entries = [];
        this.playhead = 0;

        this._resident = new Map();   // idx -> payload
        this._bytes = new Map();      // idx -> charged size
        this._pending = new Map();    // idx -> promise
        this._queue = [];             // idx awaiting a concurrency slot
        this._queued = new Set();
        this._inFlight = 0;
        this._generation = 0;

        // Diagnostics. peakInFlight and peakResident are what the memory
        // bound is measured against, so they are counted rather than asserted.
        this.retainedBytes = 0;
        this.peakInFlight = 0;
        this.peakResident = 0;
        this.peakBytes = 0;
        this.loadCount = 0;
        this.evictCount = 0;
    }

    /**
     * Point the window at a new sequence. Everything from the previous one is
     * dropped and every in-flight load for it is abandoned on arrival.
     */
    setSequence(entries, playhead = 0) {
        this.clear();
        this.entries = Array.isArray(entries) ? entries : [];
        this.playhead = this._clamp(playhead);
        this._refill();
        return this;
    }

    get length() {
        return this.entries.length;
    }

    /** Decoded frames currently held. This is the number the bound is on. */
    get residentCount() {
        return this._resident.size;
    }

    get inFlight() {
        return this._inFlight;
    }

    get queuedCount() {
        return this._queue.length;
    }

    has(idx) {
        return this._resident.has(idx);
    }

    get(idx) {
        return this._resident.get(idx) || null;
    }

    /** Frame indices the window is currently trying to hold, inclusive. */
    span() {
        const n = this.entries.length;
        if (n === 0) return { start: 0, end: -1 };
        const size = Math.min(this.windowSize, n);
        // Biased forward: playback and scrubbing both move forward far more
        // often than back, so reading ahead hits more often than reading behind.
        const behind = Math.floor((size - 1) / 3);
        let start = this.playhead - behind;
        if (start < 0) start = 0;
        if (start + size > n) start = n - size;
        return { start, end: start + size - 1 };
    }

    inSpan(idx) {
        const { start, end } = this.span();
        return idx >= start && idx <= end;
    }

    /**
     * Move the playhead. Evicts what has fallen outside the window, cancels
     * queued loads that are no longer wanted, and schedules the new ones.
     * Returns the payload for `idx` when it is already resident.
     */
    setPlayhead(idx) {
        this.playhead = this._clamp(idx);
        this._evictOutsideSpan();
        this._dropQueuedOutsideSpan();
        this._refill();
        return this.get(this.playhead);
    }

    /**
     * Resolve to the payload for one frame, loading it if it is not resident.
     * Used by the scrub path so a jump outside the window still displays.
     */
    ensure(idx) {
        const i = this._clamp(idx);
        if (this._resident.has(i)) return Promise.resolve(this._resident.get(i));
        const pending = this._pending.get(i);
        if (pending) return pending;
        // Jump the queue: this frame is being waited on, the read-ahead is not.
        this._queued.delete(i);
        this._queue = this._queue.filter((q) => q !== i);
        this._queue.unshift(i);
        this._queued.add(i);
        this._pump();
        return this._pending.get(i) || Promise.resolve(null);
    }

    /**
     * True when every frame in the current window is resident. This replaces
     * the old `_allFramesReady()`, which walked all N frames from every frame's
     * onload handler and so cost O(N^2) over a load, on the main thread,
     * interleaved with N decompressions. 10,000 frames was 100M iterations.
     */
    isWindowReady() {
        const { start, end } = this.span();
        if (end < start) return true;
        for (let i = start; i <= end; i++) {
            if (!this._resident.has(i)) return false;
        }
        return true;
    }

    /** Drop everything. In-flight loads are abandoned when they land. */
    clear() {
        for (const [idx, payload] of this._resident) {
            if (this._onEvict) {
                try { this._onEvict(idx, payload); } catch { /* an evict hook must not break the window */ }
            }
        }
        this._resident.clear();
        this._bytes.clear();
        this._queue = [];
        this._queued.clear();
        this._pending.clear();
        this.retainedBytes = 0;
        this._generation++;
        this.entries = [];
    }

    // ── internals ───────────────────────────────────────────────────────────

    _clamp(idx) {
        const n = this.entries.length;
        if (n === 0) return 0;
        const i = Math.floor(Number(idx) || 0);
        return i < 0 ? 0 : (i >= n ? n - 1 : i);
    }

    _refill() {
        const { start, end } = this.span();
        if (end < start) return;
        // Nearest to the playhead first, so a scrub shows something quickly.
        const wanted = [];
        for (let i = start; i <= end; i++) {
            if (this._resident.has(i) || this._pending.has(i) || this._queued.has(i)) continue;
            wanted.push(i);
        }
        wanted.sort((a, b) => Math.abs(a - this.playhead) - Math.abs(b - this.playhead));
        for (const i of wanted) {
            this._queue.push(i);
            this._queued.add(i);
        }
        this._pump();
    }

    _pump() {
        while (this._inFlight < this.concurrency && this._queue.length > 0) {
            const idx = this._queue.shift();
            this._queued.delete(idx);
            if (this._resident.has(idx) || this._pending.has(idx)) continue;
            this._start(idx);
        }
    }

    _start(idx) {
        const gen = this._generation;
        this._inFlight++;
        this.loadCount++;
        if (this._inFlight > this.peakInFlight) this.peakInFlight = this._inFlight;

        const promise = Promise.resolve()
            .then(() => this._load(this.entries[idx], idx))
            .then((payload) => {
                if (gen !== this._generation) return null;
                if (payload) this._insert(idx, payload);
                return payload;
            })
            .catch((err) => {
                if (this._onError) {
                    try { this._onError(err, idx); } catch { /* an error hook must not break the window */ }
                }
                return null;
            })
            .then((payload) => {
                this._inFlight--;
                this._pending.delete(idx);
                this._pump();
                return payload;
            });

        this._pending.set(idx, promise);
    }

    _insert(idx, payload) {
        if (this._resident.has(idx)) this._release(idx);
        const size = Math.max(0, Number(this._measure(payload)) || 0);
        this._resident.set(idx, payload);
        this._bytes.set(idx, size);
        this.retainedBytes += size;
        this._trim();
        if (!this._resident.has(idx)) return;  // trimmed straight back out
        if (this._resident.size > this.peakResident) this.peakResident = this._resident.size;
        if (this.retainedBytes > this.peakBytes) this.peakBytes = this.retainedBytes;
        if (this._onReady) {
            try { this._onReady(idx, payload); } catch { /* a ready hook must not break the window */ }
        }
    }

    /** Enforce both bounds, evicting furthest-from-playhead first. */
    _trim() {
        while (
            this._resident.size > 1 &&
            (this._resident.size > this.windowSize || this.retainedBytes > this.maxBytes)
        ) {
            let victim = null;
            let worst = -1;
            for (const idx of this._resident.keys()) {
                if (idx === this.playhead) continue;   // never evict what is on screen
                const d = Math.abs(idx - this.playhead);
                if (d > worst) { worst = d; victim = idx; }
            }
            if (victim === null) break;
            this._release(victim);
            this.evictCount++;
        }
    }

    _evictOutsideSpan() {
        const { start, end } = this.span();
        for (const idx of [...this._resident.keys()]) {
            if (idx === this.playhead) continue;
            if (idx < start || idx > end) {
                this._release(idx);
                this.evictCount++;
            }
        }
    }

    _dropQueuedOutsideSpan() {
        const { start, end } = this.span();
        const keep = [];
        for (const idx of this._queue) {
            if (idx >= start && idx <= end) keep.push(idx);
            else this._queued.delete(idx);
        }
        this._queue = keep;
    }

    _release(idx) {
        const payload = this._resident.get(idx);
        this.retainedBytes -= this._bytes.get(idx) || 0;
        if (this.retainedBytes < 0) this.retainedBytes = 0;
        this._resident.delete(idx);
        this._bytes.delete(idx);
        if (this._onEvict) {
            try { this._onEvict(idx, payload); } catch { /* an evict hook must not break the window */ }
        }
    }
}
