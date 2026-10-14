/**
 * Compare bookkeeping that needs no GPU: which frame of B goes with a frame of A,
 * and what to say when B is not the same size.
 *
 * Pure functions, no DOM. The viewer calls these and shows the result; the
 * drawing itself (wipe, difference, blink) stays in the renderer.
 */

const FPS_TOLERANCE = 0.01;

function finitePositive(x) {
    return typeof x === 'number' && Number.isFinite(x) && x > 0;
}

/**
 * The frame of B shown with frame `idx` of A.
 *
 * With no frame rates, or equal ones, frames pair by index, as the viewer has
 * always done. When both rates are known and differ, frames pair by time, so a
 * 24 fps B follows a 30 fps A at the same seconds, not the same frame number.
 * Past the end of B the last frame is held and `held` says so: a silent hold
 * reads as B being identical to A's last frames.
 *
 * Returns { index, held }. index is -1 when B has no frames.
 */
export function mapCompareFrame(idx, countB, { aFps, bFps } = {}) {
    if (!(countB > 0)) return { index: -1, held: false };
    let i = Number.isFinite(idx) ? Math.max(0, Math.floor(idx)) : 0;
    if (finitePositive(aFps) && finitePositive(bFps) && Math.abs(aFps - bFps) > FPS_TOLERANCE) {
        // The B frame on screen at A's frame start: floor, not round (round picks a frame that has not started).
        i = Math.floor((i / aFps) * bFps + 1e-6);
    }
    if (i > countB - 1) return { index: countB - 1, held: true };
    return { index: i, held: false };
}

/**
 * How B sits against A. `fit` is the viewer's compareFit ('pixel' or 'fit').
 * Returns { match, note }; note is null when the sizes agree.
 */
export function compareGeometry(a, b, fit = 'pixel') {
    const aw = a?.width, ah = a?.height, bw = b?.width, bh = b?.height;
    if (![aw, ah, bw, bh].every((v) => Number.isFinite(v) && v > 0)) {
        return { match: true, note: null };            // sizes unknown: say nothing
    }
    if (aw === bw && ah === bh) return { match: true, note: null };
    const how = fit === 'fit' ? 'scaled to fit A' : 'shown pixel for pixel, centred on A';
    return { match: false, note: `B is ${bw}×${bh}, A is ${aw}×${ah}: B is ${how}.` };
}
