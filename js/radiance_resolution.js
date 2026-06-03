import { app } from "../../../scripts/app.js";
// FIX 5: was "../../scripts/app.js" — extensions in custom_nodes/Radiance/web/
// need three ../ to reach ComfyUI's scripts/ directory.

/**
 * Radiance Resolution — Widget Management (v2.3)
 *
 * FIX 5: Import path corrected (see above).
 * FIX 6: Switched from nodeCreated hook + nested setTimeout to beforeRegisterNodeDef
 *         which is the standard Radiance pattern. nodeCreated fires for every node
 *         in the graph including unrelated ones; beforeRegisterNodeDef targets
 *         exactly our node type before registration, avoiding the fragile 10ms
 *         race condition on widget availability.
 *
 * FEATURE: Shows/hides mp_target and mp_aspect_ratio widgets based on whether
 *          mp_target > 0, alongside the existing video/batch toggle.
 */

function setWidgetVisible(widget, visible) {
    if (!widget) return;
    if (visible) {
        if (widget.type === "hidden") {
            widget.type = widget._origType || "INT";
            widget.computeSize = widget._origComputeSize || (() => [200, 20]);
        }
    } else {
        if (widget.type !== "hidden") {
            widget._origType = widget.type;
            widget._origComputeSize = widget.computeSize;
            widget.type = "hidden";
            widget.computeSize = () => [0, -4];
        }
    }
}

function refreshNodeSize(node) {
    // Single deferred resize — avoids double-setTimeout pattern
    requestAnimationFrame(() => {
        const sz = node.computeSize();
        sz[0] = Math.max(sz[0], node.size[0]);
        sz[1] = Math.max(sz[1], node.size[1]);
        node.size = sz;
        app.graph.setDirtyCanvas(true, true);
    });
}

app.registerExtension({
    name: "FXTD.Radiance.Resolution",
    // FIX 6: beforeRegisterNodeDef is the correct hook — targets our node only,
    // widgets exist immediately in onNodeCreated, no timing races.
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        const comfyClass = nodeType.comfyClass || nodeType.ComfyClass || nodeData.name;
        if (comfyClass !== "RadianceResolution") return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;

            const find = (name) => this.widgets?.find(w => w.name === name);

            const enableVideoW  = find("enable_video");
            const videoFramesW  = find("video_frames");
            const frameRateW    = find("frame_rate");
            const batchSizeW    = find("batch_size");
            const mpTargetW     = find("mp_target");
            const mpAspectW     = find("mp_aspect_ratio");

            const toggleFields = () => {
                if (!enableVideoW) return;

                const isVideo = enableVideoW.value === true
                    || enableVideoW.value === "true"
                    || enableVideoW.value === "True";

                setWidgetVisible(videoFramesW, isVideo);
                setWidgetVisible(frameRateW,   isVideo);
                setWidgetVisible(batchSizeW,   !isVideo);

                // FEATURE: mp_aspect_ratio only visible when mp_target > 0
                if (mpTargetW) {
                    const mpActive = parseFloat(mpTargetW.value) > 0;
                    setWidgetVisible(mpAspectW, mpActive);
                }

                refreshNodeSize(this);
            };

            // Wire callbacks
            if (enableVideoW) {
                const orig = enableVideoW.callback;
                enableVideoW.callback = function () {
                    if (orig) orig.apply(this, arguments);
                    toggleFields();
                };
            }

            if (mpTargetW) {
                const orig = mpTargetW.callback;
                mpTargetW.callback = function () {
                    if (orig) orig.apply(this, arguments);
                    toggleFields();
                };
            }

            // Initial state — widgets exist here (no setTimeout needed)
            toggleFields();

            return r;
        };
    }
});
