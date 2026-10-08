/**
 * Whole viewers in a browser, used the way a session uses them.
 *
 * Each of these was reproduced in the October 2026 review with a real viewer
 * in headless Chromium: two viewers on one graph, a video loaded straight
 * into the viewer, frames that 404 after a restart, a lost GPU context, a
 * deleted node. None of it shows up on an Object.create'd instance, so the
 * viewers here are built through the extension exactly as ComfyUI builds
 * them (see sessionharness.html and session_support.mjs).
 *
 * Skips when Playwright is unavailable.
 *
 * Run: node --test js/tests/viewer_session.test.mjs
 */
import test, { after } from 'node:test';
import assert from 'node:assert/strict';
import { startSession } from './session_support.mjs';

const session = await startSession();
const skip = session ? false : 'Playwright is not installed: the viewers cannot be driven.';
after(() => session?.close());

/** Run `fn` in a fresh page; page errors fail the test. */
async function inPage(fn, arg) {
    const page = await session.open();
    try {
        const out = await page.evaluate(fn, arg);
        return { out, errors: page.errors, page };
    } finally {
        await page.close();
    }
}

// ── C5: one panel per viewer ────────────────────────────────────────────────

test('two viewers each keep a full control panel of their own', { skip }, async () => {
    const { out, errors } = await inPage(async () => {
        const count = (v) => v.rightControlPanel?.querySelectorAll('*').length || 0;
        const a = __make(); await __sleep(300);
        const A = a.radianceViewer;
        const alone = count(A);
        const b = __make(); await __sleep(300);
        const B = b.radianceViewer;
        const r = { alone, aWithB: count(A), b: count(B),
            aPanelInA: A.rightControlPanel.contains(A.controlsPanel),
            bPanelInB: B.rightControlPanel.contains(B.controlsPanel),
            distinct: A.controlsPanel !== B.controlsPanel };

        // Exposure on B's GRADE tab changes B, and only B.
        const gradeTab = B.controlsPanel.querySelector('[data-tab-id="grade"]');
        gradeTab.click();
        const row = [...B.controlsPanel.querySelectorAll('.radiance-ref-slider')]
            .find((el) => el.querySelector('label')?.textContent.trim() === 'Exposure');
        const input = row.querySelector('input[type="range"]');
        input.value = '2';
        input.dispatchEvent(new Event('input', { bubbles: true }));
        r.exposure = { A: A.exposure || 0, B: B.exposure || 0 };

        // Deleting B leaves A whole.
        __remove(b); await __sleep(200);
        r.aAfterRemoveB = count(A);
        r.aPanelStillInA = A.rightControlPanel.contains(A.controlsPanel);
        __remove(a);
        return r;
    });
    assert.deepEqual(errors, []);
    assert.ok(out.alone > 50, `a lone viewer's panel has ${out.alone} elements`);
    assert.ok(out.aWithB > 50, `viewer A's panel dropped to ${out.aWithB} elements when B was added`);
    assert.ok(out.b > 50);
    assert.ok(out.distinct && out.aPanelInA && out.bPanelInB, JSON.stringify(out));
    assert.deepEqual(out.exposure, { A: 0, B: 2 }, 'moving Exposure in B did not change B alone');
    assert.ok(out.aAfterRemoveB > 50, `deleting B left A with ${out.aAfterRemoveB} panel elements`);
    assert.ok(out.aPanelStillInA);
});

// ── M11, M12: the renderer starts once, the GPU budget is the page's ───────

test('the renderer compiles its shaders once per viewer', { skip }, async () => {
    const { out, errors } = await inPage(async () => {
        const P = window.RadianceWebGLRenderer.prototype;
        let builds = 0, lostHandlers = 0;
        const createPrograms = P.createPrograms;
        P.createPrograms = function () { builds++; return createPrograms.call(this); };
        const add = HTMLCanvasElement.prototype.addEventListener;
        HTMLCanvasElement.prototype.addEventListener = function (type, ...rest) {
            if (type === 'webglcontextlost') lostHandlers++;
            return add.call(this, type, ...rest);
        };
        const n = __make(); await __sleep(200);
        const r = { builds, lostHandlers, backend: n.radianceViewer._gpuBackend };
        __remove(n);
        return r;
    });
    assert.deepEqual(errors, []);
    assert.equal(out.backend, 'webgl');
    assert.equal(out.builds, 1, `shaders were compiled ${out.builds} times for one viewer`);
    assert.equal(out.lostHandlers, 1, `${out.lostHandlers} context-loss handlers for one renderer`);
});

test('the GPU frame cache budget is shared by the viewers on the page', { skip }, async () => {
    const { out, errors } = await inPage(async () => {
        const a = __make(); await __sleep(100);
        const one = a.radianceViewer.renderer._frameCacheByteBudget;
        const b = __make(); await __sleep(100);
        const two = [a.radianceViewer.renderer._frameCacheByteBudget, b.radianceViewer.renderer._frameCacheByteBudget];
        __remove(b); await __sleep(50);
        const back = a.radianceViewer.renderer._frameCacheByteBudget;
        __remove(a);
        return { one, two, back };
    });
    assert.deepEqual(errors, []);
    assert.ok(out.one > 0);
    assert.deepEqual(out.two, [Math.floor(out.one / 2), Math.floor(out.one / 2)],
        'each viewer kept the whole budget');
    assert.equal(out.back, out.one, 'a deleted viewer kept its share');
});

// ── H12: a deleted viewer is freed ──────────────────────────────────────────

test('a deleted viewer removes every page listener and can be collected', { skip }, async () => {
    const page = await session.open();
    try {
        const before = await page.evaluate(() => __listenerCount());
        // Defined in a scope of its own: a closure made next to 'v' would
        // share its context and keep the viewer alive itself.
        await page.evaluate(() => {
            window.__fields = () => {
                const x = window.__ref.deref();
                return x && { hdrData: !!x.hdrData, image: !!x.image, compareImage: !!x.compareImage,
                    zdepthImage: !!x.zdepthImage, frames: (x.frameHDRData || []).filter(Boolean).length };
            };
        });
        await page.evaluate(async () => {
            const n = __make(); const v = n.radianceViewer;
            n.onExecuted(__frames(3, 320, 180));
            await __until(() => v.hdrData);
            await __sleep(300);
            window.__ref = new WeakRef(v);
            __remove(n);
        });
        const fields = await page.evaluate(() => window.__fields());
        const left = await page.evaluate(() => __listenerCount());
        const cdp = await page.context().newCDPSession(page);
        let collected = false;
        for (let i = 0; i < 40 && !collected; i++) {
            await page.evaluate(() => new Promise((r) => setTimeout(r, 250)));
            await cdp.send('HeapProfiler.collectGarbage');
            collected = await page.evaluate(() => window.__ref.deref() === undefined);
        }
        assert.deepEqual(page.errors, []);
        assert.equal(left, before, `a deleted viewer left ${left - before} window/document listeners`);
        assert.deepEqual(fields, { hdrData: false, image: false, compareImage: false, zdepthImage: false, frames: 0 },
            'destroy() left frame data on the instance');
        assert.ok(collected, 'the deleted viewer is still reachable after garbage collection');
    } finally {
        await page.close();
    }
});

// ── H7: a frame change does not rebuild the panel under the user ───────────

test('a slider being dragged survives scrubbing, and so does the channel search', { skip }, async () => {
    const { out, errors } = await inPage(async () => {
        const n = __make(); const v = n.radianceViewer;
        n.onExecuted(__frames(12, 96, 54));
        await __until(() => v.hdrData && v._frameWindow?.isWindowReady(), 30000);
        const r = {};
        const panel = v.controlsPanel;
        const row = (label) => [...panel.querySelectorAll('.radiance-ref-slider')]
            .find((el) => el.querySelector('label')?.textContent.trim() === label);

        // GRADE: hold Exposure, scrub, keep dragging.
        panel.querySelector('[data-tab-id="grade"]').click();
        const input = row('Exposure').querySelector('input[type="range"]');
        input.dispatchEvent(new PointerEvent('pointerdown', { bubbles: true }));
        for (const f of [1, 2, 3, 4, 5]) { v.setFrame(f); await __sleep(30); }
        // Anything else that redraws the panel waits too: OCIO finishing
        // loading in the background did it mid-drag.
        v._ocioSetStatus?.('info', 'OCIO ready');
        v._lastRenderContent?.();
        r.gradeSliderStillThere = input.isConnected;
        input.value = '1.5';
        input.dispatchEvent(new Event('input', { bubbles: true }));
        r.exposureAfterDrag = v.exposure;
        window.dispatchEvent(new PointerEvent('pointerup'));
        input.blur();
        await __sleep(20);
        r.redrawnAfterRelease = !input.isConnected;

        // EFFECTS: the same with Amount.
        panel.querySelector('[data-tab-id="effects"]').click();
        const amount = row('Amount').querySelector('input[type="range"]');
        amount.dispatchEvent(new PointerEvent('pointerdown', { bubbles: true }));
        for (const f of [6, 7, 8]) { v.setFrame(f); await __sleep(30); }
        r.effectsSliderStillThere = amount.isConnected;
        window.dispatchEvent(new PointerEvent('pointerup'));

        // INSPECTOR: type a search, scrub; the text stays and the readouts follow.
        panel.querySelector('[data-tab-id="inspector"]').click();
        const search = panel.querySelector('.radiance-ref-search');
        search.focus();
        search.value = 'Gr';
        search.dispatchEvent(new Event('input', { bubbles: true }));
        for (const f of [9, 10, 11]) { v.setFrame(f); await __sleep(30); }
        r.searchStillThere = search.isConnected;
        r.searchValue = search.value;
        r.searchFocused = document.activeElement === search;
        const kv = [...panel.querySelectorAll('.radiance-ref-kv .k')].find((k) => k.textContent === 'Frame');
        r.frameReadout = kv?.nextElementSibling?.textContent;
        __remove(n);
        return r;
    });
    assert.deepEqual(errors, []);
    assert.equal(out.gradeSliderStillThere, true, 'scrubbing rebuilt the GRADE panel under a held slider');
    assert.equal(out.exposureAfterDrag, 1.5, 'the held slider no longer drives the grade');
    assert.equal(out.redrawnAfterRelease, true, 'the redraw that waited never ran');
    assert.equal(out.effectsSliderStillThere, true, 'scrubbing rebuilt the EFFECTS panel under a held slider');
    assert.equal(out.searchStillThere, true, 'scrubbing rebuilt the Inspector under the search box');
    assert.equal(out.searchValue, 'Gr', 'the channel search text was wiped');
    assert.equal(out.searchFocused, true, 'the search box lost focus');
    assert.equal(out.frameReadout, '12', 'the Inspector readouts did not follow the frame');
});

// ── H13: playback does one frame's work per frame ───────────────────────────

test('showing a float frame neither builds a full-size placeholder nor decodes floats', { skip }, async () => {
    const { out, errors } = await inPage(async () => {
        // Full-frame read-backs only (thumbnails and sparklines read tiny ones).
        let readBacks = 0, statsOnMain = 0;
        const gid = CanvasRenderingContext2D.prototype.getImageData;
        CanvasRenderingContext2D.prototype.getImageData = function (...a) {
            if (a[2] * a[3] >= 320 * 180) readBacks++;
            return gid.apply(this, a);
        };
        const n = __make(); const v = n.radianceViewer;
        const zs = v._zoneStatsFromFloats;
        v._zoneStatsFromFloats = function (...a) { statsOnMain++; return zs.apply(this, a); };
        n.onExecuted(__frames(8, 320, 180));
        await __until(() => v.hdrData && v._frameWindow?.isWindowReady());
        await __sleep(200);
        for (let f = 1; f < 8; f++) v.setFrame(f);
        const r = {
            readBacks, statsOnMain,
            image: !!v.image, imageW: v.image?.width, imageData: !!v.imageData,
            ownKeys: Object.keys(v.hdrData),
            hasHalves: v.hdrData.fp16data instanceof Uint16Array,
            // fp16 RGBA 320x180 plus the 8-bit PNG proxy, per frame
            bytesPerFrame: v._frameWindow.retainedBytes / v._frameWindow.residentCount,
        };
        CanvasRenderingContext2D.prototype.getImageData = gid;
        __remove(n);
        return r;
    });
    assert.deepEqual(errors, []);
    assert.equal(out.readBacks, 0, `${out.readBacks} full-frame canvas read-backs to load and step through 8 float frames`);
    assert.equal(out.statsOnMain, 0, 'zone statistics were computed on the main thread');
    assert.ok(out.image && out.imageData && out.imageW === 320, 'the stand-in image lost its size or truthiness');
    assert.ok(!out.ownKeys.includes('data') && out.hasHalves, 'a frame still carries a float copy');
    assert.equal(out.bytesPerFrame, 320 * 180 * 4 * 2 + 320 * 180 * 4);
});

test('the zone statistics and nit badge follow the frame on screen', { skip }, async () => {
    const { out, errors } = await inPage(async () => {
        const n = __make(); const v = n.radianceViewer;
        n.onExecuted(__frames(6, 64, 48, {}, 'g'));
        await __until(() => v.hdrData && v._frameWindow?.isWindowReady());
        const r = [];
        for (const f of [3, 1, 5]) {
            v.setFrame(f);
            r.push({ f, p50: +v._hdrZoneStats.p50.toFixed(3), badge: v.hdrPeakInfo?.textContent });
        }
        // Playback reaches frames through setFrame too.
        v.setFrame(0);
        v.togglePlayback();
        await __sleep(150);
        v.togglePlayback();
        r.push({ f: v.currentFrame, p50: +v._hdrZoneStats.p50.toFixed(3), badge: v.hdrPeakInfo?.textContent });
        __remove(n);
        return r;
    });
    assert.deepEqual(errors, []);
    for (const { f, p50 } of out) {
        assert.ok(Math.abs(p50 - 0.05 * (f + 1)) < 0.002, `frame ${f}: p50 ${p50}, the frame is ${0.05 * (f + 1)}`);
    }
    assert.notEqual(out[0].badge, out[1].badge, 'the nit badge did not change with the frame');
});

test('during playback the scopes update a few times a second, not twice a frame', { skip }, async () => {
    const { out, errors } = await inPage(async () => {
        const n = __make(); const v = n.radianceViewer;
        n.onExecuted(__frames(48, 64, 48));
        await __until(() => v.hdrData && v._frameWindow?.isWindowReady());
        v._setReferenceTab('scopes');
        await __sleep(200);
        // Counted, not drawn: four software-GL read-backs take long enough
        // here to slow playback below the rate the limit is about.
        let scopes = 0, frames = 0;
        v._updateReferenceScopes = () => { scopes++; };
        const sf = v.setFrame.bind(v);
        v.setFrame = (i) => { frames++; sf(i); };
        // Until 24 frames have played, however long a loaded machine takes.
        const t0 = performance.now();
        v.togglePlayback();
        await __until(() => frames >= 24, 20000);
        const seconds = (performance.now() - t0) / 1000;
        v.togglePlayback();
        __remove(n);
        return { scopes, frames, seconds };
    });
    assert.deepEqual(errors, []);
    assert.ok(out.frames >= 24, `playback only advanced ${out.frames} frames`);
    // Two a frame before: one from render(), one from setFrame. Now about
    // one at most (a frame landing redraws too), and no more than about six
    // a second.
    assert.ok(out.scopes <= Math.ceil(out.frames * 1.25) + 2, `${out.scopes} scope redraws for ${out.frames} frames`);
    assert.ok(out.scopes <= out.seconds * 6.5 + 2,
        `${out.scopes} scope redraws in ${out.seconds.toFixed(1)} s of playback (${out.frames} frames)`);
});

// ── M16: EV range from the lowest non-zero percentile ───────────────────────

test('a frame with true black still reports its EV range', { skip }, async () => {
    const { out, errors } = await inPage(async () => {
        const n = __make(); const v = n.radianceViewer;
        n.onExecuted(__frames(1, 128, 64, {}, 'black'));
        await __until(() => v._hdrZoneStats);
        const r = { ev: v._hdrZoneStats.evRange, p1: v._hdrZoneStats.p1, badge: v.hdrPeakInfo.textContent };
        __remove(n);
        return r;
    });
    assert.deepEqual(errors, []);
    assert.equal(out.p1, 0, 'test premise: the darkest 1% is black');
    // Half black, the rest 0.01 to 41.6 evenly in stops: p99 of the frame
    // over p1 of what is lit is about 11.7 stops.
    assert.ok(out.ev > 11 && out.ev < 12.5, `EV range ${out.ev}`);
    assert.match(out.badge, new RegExp(`${out.ev.toFixed(1)} EV`));
});

// ── C6: a video loaded into the viewer steps, scrubs and plays by frame ────

test('stepping, scrubbing and the arrow keys move a video picture frame by frame', { skip }, async () => {
    const { out, errors } = await inPage(async () => {
        const n = __make('simple'); const v = n.radianceViewer;
        await v.loadVideo('/view?filename=clip24.webm&type=temp');
        await __until(() => v._videoContainerFps && !v._videoProbing, 25000);
        const vid = v.videoEl;
        const settle = () => __until(() => !vid.seeking, 3000).then(() => __sleep(80));
        // The fixture's frame n is grey at 8 n (js/tests/fixtures/clip24.webm).
        const shown = () => {
            const c = v._videoCanvas.getContext('2d').getImageData(v._videoCanvas.width >> 1, v._videoCanvas.height >> 1, 1, 1).data;
            return Math.round(c[0] / 8);
        };
        const at = () => ({ frame: v.currentFrame, picture: shown(), time: +vid.currentTime.toFixed(4),
            scrub: +v._sbScrub.value, dock: +v.sequenceRange.value, label: v._sbFrame.textContent });
        await settle();
        const r = { fps: v._videoContainerFps, total: v.totalFrames, start: at() };
        v._sbNext.click(); await settle(); v._sbNext.click(); await settle();
        r.afterNext = at();
        v._sbScrub.value = '20'; v._sbScrub.dispatchEvent(new Event('input')); await settle();
        r.afterScrub = at();
        v.container.dispatchEvent(new PointerEvent('pointerenter'));
        document.body.dispatchEvent(new KeyboardEvent('keydown', { key: 'ArrowLeft', code: 'ArrowLeft', bubbles: true }));
        await settle();
        r.afterArrow = at();
        v.sequenceRange.value = '7'; v.sequenceRange.dispatchEvent(new Event('input')); await settle();
        r.afterDock = at();
        v.togglePlayback(); await __sleep(400); v.togglePlayback();
        await __sleep(150); await settle();
        r.afterPlay = at();
        __remove(n);
        return r;
    });
    assert.deepEqual(errors, []);
    assert.equal(out.fps, 24, 'the container frame rate was not read');
    assert.equal(out.total, 30);
    const check = (s, frame, what) => {
        assert.equal(s.frame, frame, `${what}: frame ${s.frame}`);
        assert.equal(s.picture, frame, `${what}: the picture shows frame ${s.picture}, the counter says ${frame}`);
        assert.ok(Math.abs(s.time - (frame + 0.5) / 24) < 1e-3, `${what}: video time ${s.time}`);
        assert.equal(s.scrub, frame, `${what}: simple-bar scrubber at ${s.scrub}`);
        assert.equal(s.dock, frame, `${what}: dock range at ${s.dock}`);
        assert.equal(s.label, `${frame + 1} / 30`);
    };
    check(out.start, 0, 'loaded');
    check(out.afterNext, 2, 'two clicks on the next button');
    check(out.afterScrub, 20, 'scrubbing');
    check(out.afterArrow, 19, 'the left arrow');
    check(out.afterDock, 7, 'the dock scrubber');
    assert.ok(out.afterPlay.frame > 7, `playback did not move the counter (${out.afterPlay.frame})`);
    assert.equal(out.afterPlay.scrub, out.afterPlay.frame, 'the scrubber did not follow playback');
    assert.equal(out.afterPlay.picture, out.afterPlay.frame, 'after playback the counter and the picture disagree');
});

// ── H14: failures say so ────────────────────────────────────────────────────

test('frames that are gone from the server show a message, not a blank viewer', { skip }, async () => {
    const { out, errors } = await inPage(async () => {
        const n = __make(); const v = n.radianceViewer;
        // A saved result after a ComfyUI restart: every temp file 404s.
        n.onExecuted({ radiance_images: [{ filename: 'gone_0.png', subfolder: '', type: 'temp',
            hdr_sidecar: 'gone_0.rhdr', has_hdr: true, hdr_primary: true }], fps: [24] });
        await __until(() => v.container.querySelector('.radiance-viewer-message'), 8000);
        const msg = v.container.querySelector('.radiance-viewer-message[data-kind="frames"]');
        const r = { text: msg?.textContent || '', visible: !!msg && msg.getBoundingClientRect().width > 0 };
        // The next run's frames clear it.
        n.onExecuted(__frames(2));
        await __until(() => v.hdrData);
        r.clearedByNewFrames = !v.container.querySelector('.radiance-viewer-message[data-kind="frames"]');
        __remove(n);
        return r;
    });
    assert.deepEqual(errors, []);
    assert.ok(out.visible, 'no message is shown');
    assert.match(out.text, /no longer on the server/);
    assert.ok(out.clearedByNewFrames, 'the message stayed over a frame that loaded');
});

test('a missing or corrupt float sidecar is reported for what it is, with no uncaught errors', { skip }, async () => {
    const { out, errors } = await inPage(async () => {
        const run = async (sidecar) => {
            window.__unhandled.length = 0;
            const n = __make(); const v = n.radianceViewer;
            n.onExecuted({ radiance_images: [{ filename: 'f_64x48_0.png', subfolder: '', type: 'temp',
                hdr_sidecar: sidecar, has_hdr: true, hdr_primary: true }], fps: [24] });
            await __until(() => v.image && v.bitDepthInfo.textContent.includes('PROXY'), 8000);
            await __sleep(300);
            const r = { reason: v._currentFallbackReason(), badge: v.bitDepthInfo.textContent, unhandled: [...window.__unhandled] };
            __remove(n);
            return r;
        };
        return { missing: await run('gone_0.rhdr'), corrupt: await run('bad.rhdr'), truncated: await run('trunc.rhdr') };
    });
    assert.deepEqual(errors, []);
    assert.match(out.missing.reason, /no longer on the server/, out.missing.reason);
    assert.doesNotMatch(out.missing.reason, /DecompressionStream/);
    for (const k of ['missing', 'corrupt', 'truncated']) {
        assert.deepEqual(out[k].unhandled, [], `${k}: uncaught promise errors`);
        assert.match(out[k].badge, /PROXY 8-BIT/, `${k}: ${out[k].badge}`);
    }
});

test('a video that cannot be played says so', { skip }, async () => {
    const { out, errors } = await inPage(async () => {
        const n = __make(); const v = n.radianceViewer;
        await v.loadVideo('/view?filename=gone.webm&type=temp');
        await __until(() => /no longer on the server/.test(v.container.querySelector('.radiance-viewer-message[data-kind="video"]')?.textContent || ''), 8000);
        const r = { text: v.container.querySelector('.radiance-viewer-message[data-kind="video"]')?.textContent || '' };
        __remove(n);
        return r;
    });
    assert.deepEqual(errors, []);
    assert.match(out.text, /no longer on the server/);
});

test('a lost GPU context is shown, and on restore the frame, LUT and curves come back', { skip }, async () => {
    const { out, errors } = await inPage(async () => {
        const n = __make(); const v = n.radianceViewer;
        n.onExecuted(__frames(2, 64, 48));
        await __until(() => v.hdrData);
        await __sleep(200);
        v.renderer.loadLUT(new Float32Array(2 * 2 * 2 * 3).fill(0.5), 2);
        const gl = v.renderer.gl;
        const ext = gl.getExtension('WEBGL_lose_context');
        ext.loseContext();
        await __until(() => v.renderer._contextLost);
        await __sleep(100);
        const msg = v.container.querySelector('.radiance-viewer-message[data-kind="context"]');
        const r = { lostMessage: msg?.textContent || '' };
        ext.restoreContext();
        await __until(() => !v.renderer._contextLost, 5000);
        await __sleep(200);
        r.messageGone = !v.container.querySelector('.radiance-viewer-message[data-kind="context"]');
        r.image = !!v.renderer.textures.image;
        r.lut = !!v.renderer.textures.lut;
        r.curves = !!v.renderer.curveLutTexture;
        v.render();
        const px = new Uint8Array(4);
        gl.readPixels(v.glCanvas.width >> 1, v.glCanvas.height >> 1, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, px);
        r.centre = [...px];
        __remove(n);
        return r;
    });
    assert.deepEqual(errors, []);
    assert.match(out.lostMessage, /GPU context was lost/);
    assert.ok(out.messageGone, 'the message stayed after the context came back');
    assert.ok(out.image, 'the frame was not uploaded again');
    assert.ok(out.lut, 'the display LUT was not uploaded again');
    assert.ok(out.curves, 'the curve table was not uploaded again');
    assert.ok(out.centre[0] > 20, `the restored picture is black: ${out.centre}`);
});

// ── The grain ticker runs only for animated grain ───────────────────────────

test('the grain ticker runs only while animated grain is on', { skip }, async () => {
    const { out, errors } = await inPage(async () => {
        const n = __make(); const v = n.radianceViewer;
        n.onExecuted(__frames(1, 64, 48));
        await __until(() => v.hdrData);
        await __sleep(300);
        const r = { off: !!v._grainRAF };
        v.grain = 0.2; v.grainAnimate = false; v.render(); await __sleep(100);
        r.staticGrain = !!v._grainRAF;
        let ticks = 0;
        const setTime = v.renderer.setTime.bind(v.renderer);
        v.renderer.setTime = (t) => { ticks++; setTime(t); };
        v.grainAnimate = true; v.render();
        r.animated = !!v._grainRAF;
        r.timeMoves = await __until(() => ticks >= 2, 8000);
        v.grain = 0;
        r.stoppedAfterOff = await __until(() => !v._grainRAF, 8000);
        __remove(n);
        return r;
    });
    assert.deepEqual(errors, []);
    assert.equal(out.off, false, 'the ticker runs with grain off');
    assert.equal(out.staticGrain, false, 'the ticker runs for grain that does not move');
    assert.equal(out.animated, true, 'animated grain did not start the ticker');
    assert.equal(out.timeMoves, true, 'the grain does not move');
    assert.equal(out.stoppedAfterOff, true, 'the ticker kept running after grain was turned off');
});

// ── H16: B is B's float frame, through the same view, placed by pixel size ──

test('compare B uses its float frame through the same view, placed 1:1 or fitted', { skip }, async () => {
    const { out, errors } = await inPage(async () => {
        const n = __make(); const v = n.radianceViewer;
        // A: 64x48 at 0.18. B: 32x24, its float frame also 0.18 with a bright
        // top-left pixel (9.0, to check B is the right way up), its preview
        // PNG a different grey (46), so the test can tell which one is drawn.
        const msg = __frames(4, 64, 48);
        for (let i = 0; i < 4; i++) {
            msg.radiance_images.push({ filename: 'b_32x24_46.png', subfolder: '', type: 'temp', is_compare: true,
                hdr_sidecar: 'f_32x24_9.rhdr', has_hdr: true, frame: i, source_width: 32, source_height: 24 });
        }
        n.onExecuted(msg);
        await __until(() => v.hdrData && v.compareHDR, 15000);
        const gl = v.renderer.gl;
        const read = (x, y) => {
            v.render();
            const px = new Uint8Array(4);
            gl.readPixels(x, y, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, px);
            return px[0];
        };
        const r = { floatB: !!v.compareHDR };
        v.setCompareMode('none'); r.a = read(32, 24);
        // Rows are read bottom-up: B's top-left pixel, 1:1 and centred, is A's
        // pixel (16, 12) from the top, row 48 - 1 - 12 = 35 from the bottom.
        v.setCompareMode('b'); r.bCentre = read(32, 24); r.bCorner = read(2, 2);
        r.bTopLeft = read(16, 35); r.bBelowTopLeft = read(16, 33);
        v.setCompareMode('difference'); r.diffCentre = read(32, 24); r.diffCorner = read(2, 2);
        v.setCompareMode('b'); v.setCompareFit('fit'); r.fitCorner = read(2, 2);
        v.setCompareFit('pixel');
        __remove(n);
        return r;
    });
    assert.deepEqual(errors, []);
    assert.ok(out.floatB, "B's float sidecar was not loaded");
    assert.ok(out.a > 20, `test premise: A shows ${out.a}`);
    assert.ok(Math.abs(out.bCentre - out.a) <= 2, `B (${out.bCentre}) is not drawn through A's view (${out.a})`);
    assert.ok(out.diffCentre <= 3, `difference where A and B match is ${out.diffCentre}`);
    assert.ok(out.bCorner <= 2, `a smaller B was stretched over A instead of placed 1:1 (${out.bCorner})`);
    assert.ok(out.diffCorner > 20, 'outside B the difference should show A');
    assert.ok(Math.abs(out.fitCorner - out.a) <= 2, `fitted, B should cover the corner (${out.fitCorner})`);
    assert.ok(out.bTopLeft > out.bBelowTopLeft + 30, `B is upside down or misplaced: ${out.bTopLeft} vs ${out.bBelowTopLeft}`);
});
