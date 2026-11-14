/**
 * Review report: review marks plus an optional QC report, as JSON and Markdown.
 * Pure functions, no DOM. The QC part is embedded as given; if no QC run exists
 * it says so (qc: null) instead of implying a pass.
 */
export const REPORT_SCHEMA = 'radiance.review.report';
export const REPORT_VERSION = 1;

const byFrame = (a, b) => a.frame - b.frame || String(a.id).localeCompare(String(b.id));
const cell = (s) => String(s ?? '').replace(/\r?\n/g, ' ').replace(/\|/g, '\\|');

/** store: ReviewStore. meta: { title, frames, fps, source }. qc: radiance.qc.report or null. */
export function buildReviewReport(store, meta = {}, qc = null) {
    const marks = [...(store?.marks || [])].sort(byFrame);
    const frames = Number.isInteger(meta.frames) ? meta.frames : null;
    return {
        schema: REPORT_SCHEMA,
        version: REPORT_VERSION,
        title: String(meta.title ?? ''),
        source: meta.source ?? null,
        frames,
        fps: Number.isFinite(meta.fps) ? meta.fps : null,
        counts: {
            strokes: marks.filter((m) => m.kind === 'stroke').length,
            notes: marks.filter((m) => m.kind === 'note').length,
            bookmarks: marks.filter((m) => m.kind === 'bookmark').length,
        },
        // Out-of-range frames are kept and flagged, never dropped.
        marks: marks.map((m) => ({ ...m, outOfRange: frames != null && m.frame >= frames })),
        qc: qc && qc.schema === 'radiance.qc.report' ? qc : null,
    };
}

export function reportToMarkdown(report) {
    const L = [`# Review report${report.title ? `: ${report.title}` : ''}`, ''];
    const c = report.counts;
    L.push(`Frames: ${report.frames ?? 'unknown'}. Notes: ${c.notes}. Bookmarks: ${c.bookmarks}. Drawn strokes: ${c.strokes}.`, '');
    const notes = report.marks.filter((m) => m.kind === 'note' || m.kind === 'bookmark');
    L.push('## Notes and bookmarks', '');
    if (!notes.length) L.push('None.', '');
    else {
        L.push('| Frame | Kind | Text |', '| --- | --- | --- |');
        for (const m of notes) {
            const text = (m.kind === 'note' ? m.text : m.label) + (m.outOfRange ? ' (frame beyond sequence length)' : '');
            L.push(`| ${m.frame} | ${m.kind} | ${cell(text)} |`);
        }
        L.push('');
    }
    L.push('## QC', '');
    if (!report.qc) L.push('No QC run was attached. This report makes no statement about media quality.', '');
    else {
        L.push(`Overall: ${report.qc.overall}${report.qc.complete ? '' : ' (incomplete run)'}. Domain: ${report.qc.domain}.`, '');
        L.push('| Check | Status |', '| --- | --- |');
        for (const k of report.qc.checks) L.push(`| ${cell(k.id ?? k.name)} | ${cell(k.status)} |`);
        L.push('');
    }
    return L.join('\n');
}
