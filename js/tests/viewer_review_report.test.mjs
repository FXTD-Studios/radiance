import test from 'node:test';
import assert from 'node:assert/strict';
import { ReviewStore } from '../radiance_review.js';
import { buildReviewReport, reportToMarkdown, REPORT_SCHEMA } from '../radiance_review_report.js';

const store = () => {
    const s = new ReviewStore();
    s.addNote(5, 0.5, 0.5, 'flare | here\nline2');
    s.addBookmark(2, 'cut');
    s.addStroke(1, [{ u: 0, v: 0 }, { u: 1, v: 1 }]);
    s.addNote(50, 0.1, 0.1, 'late');
    return s;
};

test('report counts, sorts by frame and flags out-of-range marks', () => {
    const r = buildReviewReport(store(), { title: 'sh010', frames: 10 });
    assert.equal(r.schema, REPORT_SCHEMA);
    assert.deepEqual(r.counts, { strokes: 1, notes: 2, bookmarks: 1 });
    assert.deepEqual(r.marks.map((m) => m.frame), [1, 2, 5, 50]);
    assert.equal(r.marks.at(-1).outOfRange, true);
    assert.equal(r.marks[0].outOfRange, false);
});

test('unknown length flags nothing', () => {
    assert.ok(buildReviewReport(store(), {}).marks.every((m) => m.outOfRange === false));
});

test('no QC attached is stated, never a pass', () => {
    const r = buildReviewReport(store(), { frames: 10 }, { overall: 'PASS' });
    assert.equal(r.qc, null);
    const md = reportToMarkdown(r);
    assert.match(md, /No QC run was attached/);
    assert.doesNotMatch(md, /PASS/);
});

test('a QC report is embedded and tabulated', () => {
    const qc = { schema: 'radiance.qc.report', overall: 'WARN', complete: false, domain: 'd', checks: [{ id: 'over_range', status: 'WARN' }] };
    const md = reportToMarkdown(buildReviewReport(store(), { frames: 10 }, qc));
    assert.match(md, /Overall: WARN \(incomplete run\)/);
    assert.match(md, /\| over_range \| WARN \|/);
});

test('markdown escapes pipes and newlines in notes', () => {
    const md = reportToMarkdown(buildReviewReport(store(), { frames: 10 }));
    assert.match(md, /flare \\\| here line2/);
    assert.match(md, /frame beyond sequence length/);
});

test('empty store still produces a report', () => {
    const r = buildReviewReport(new ReviewStore(), { frames: 3 });
    assert.match(reportToMarkdown(r), /None\./);
});
