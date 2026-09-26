/**
 * SMPTE ST 12 timecode for the Viewer (3.5.0).
 *
 * The viewer formatted timecode four different ways, all of them with the real
 * frame rate as the frame modulus: `Math.round(f % 29.97)` returns 30 for frame
 * 539 ("00:00:17:30", a timecode that cannot exist), and mixing fractional fps
 * for seconds with rounded fps for frames drifted one second every ~1000 s at
 * 23.976 against the dock.
 *
 * Timecode counts frames against an INTEGER timebase: 24 for 23.976, 30 for
 * 29.97, 60 for 59.94. At 29.97 and 59.94 the broadcast convention is
 * drop-frame, which skips frame NUMBERS 0-1 (0-3 at 59.94) at the start of
 * every minute except every tenth, so the clock stays on wall time. Drop-frame
 * is written with ';' before the frames field.
 */

/** The integer timebase timecode counts in. */
export function timebase(fps) {
    return Math.max(1, Math.round(Number(fps) || 24));
}

/** Drop-frame applies to 29.97 and 59.94 only. */
export function isDropFrameRate(fps) {
    return Math.abs(fps - 29.97) < 0.01 || Math.abs(fps - 59.94) < 0.01;
}

/**
 * Frame index (0-based) to "HH:MM:SS:FF" (non-drop) or "HH:MM:SS;FF" (drop).
 * @param {number} frame
 * @param {number} fps
 * @param {{dropFrame?: boolean}} [opts] drop-frame defaults to on at 29.97 / 59.94
 */
export function smpteTimecode(frame, fps, opts = {}) {
    const tb = timebase(fps);
    let f = Math.max(0, Math.floor(Number(frame) || 0));
    const drop = (opts.dropFrame ?? true) && isDropFrameRate(fps);
    if (drop) {
        const dropN = Math.round(tb / 15);                 // 2 at 29.97, 4 at 59.94
        const per10Min = tb * 600 - dropN * 9;              // 17982 at 29.97
        const perMin = tb * 60 - dropN;                     // 1798 at 29.97
        const d = Math.floor(f / per10Min);
        const m = f % per10Min;
        f += dropN * 9 * d + (m > dropN ? dropN * Math.floor((m - dropN) / perMin) : 0);
    }
    const ff = f % tb;
    const totalSec = Math.floor(f / tb);
    const ss = totalSec % 60;
    const mm = Math.floor(totalSec / 60) % 60;
    const hh = Math.floor(totalSec / 3600) % 24;
    const p = (n) => String(n).padStart(2, '0');
    return `${p(hh)}:${p(mm)}:${p(ss)}${drop ? ';' : ':'}${p(ff)}`;
}

/** Common playback rates for the Viewer's fps menu. */
export const FPS_CHOICES = [23.976, 24, 25, 29.97, 30, 48, 50, 59.94, 60];
