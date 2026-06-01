import { app } from "../../../scripts/app.js";

/**
 * Radiance Universal I/O Widget Management (v2.3)
 * Handles dynamic visibility for Digital Cinema Read and Write nodes.
 *
 * Fixes applied:
 *  FIX 1: Node names updated to match nodes_io.py after ◎ was removed from
 *          NODE_CLASS_MAPPINGS keys. Extension was completely dead before.
 *  FIX 2: Widget hiding uses w.type = "hidden" (valid ComfyUI API) instead of
 *          "converted-widget" which is not a recognised hide token — all
 *          show/hide logic was a no-op.
 *  FIX 3: origType forEach now guards against undefined widgets with `if (w)`.
 *  FIX 4: Read node calls sourceWidget.callback() at init so label is correct
 *          when a saved workflow is restored.
 *  FIX 5: Read node isVideo check extended to include .avi .mkv .webm,
 *          matching the Python reader's accepted extensions exactly.
 */

// FIX 5: Single source-of-truth for video extensions — mirrors Python read().
const VIDEO_EXTENSIONS = [".mp4", ".mov", ".gif", ".webp", ".avi", ".mkv", ".webm"];

app.registerExtension({
	name: "Radiance.IO",
	async beforeRegisterNodeDef(nodeType, nodeData, app) {
		const comfyClass = nodeType.comfyClass || nodeType.ComfyClass || nodeData.name;

		// 1. Digital Cinema Read — label intelligence
		// FIX 1: was "◎ RadianceDigitalCinemaRead" — ◎ removed from Python mapping key.
		if (comfyClass === "RadianceDigitalCinemaRead" || comfyClass === "◎ RadianceDigitalCinemaRead") {
			const onNodeCreated = nodeType.prototype.onNodeCreated;
			nodeType.prototype.onNodeCreated = function () {
				const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;

				const sourceWidget = this.widgets.find(w => w.name === "source_path");

				if (sourceWidget) {
					// FIX 5: use shared VIDEO_EXTENSIONS list
					const updateLabel = () => {
						const val = (sourceWidget.value || "").toLowerCase();
						const isVideo = VIDEO_EXTENSIONS.some(ext => val.endsWith(ext));
						sourceWidget.label = isVideo ? "SOURCE (VIDEO)" : "SOURCE (SEQUENCE/IMAGE)";
					};

					sourceWidget.callback = updateLabel;

					// FIX 4: run immediately so restored workflows show the right label
					setTimeout(updateLabel, 20);
				}

				return r;
			};
		}

		// 2. Digital Cinema Write — smart show/hide toggles
		// FIX 1: was "◎ RadianceDigitalCinemaWrite"
		if (comfyClass === "RadianceDigitalCinemaWrite" || comfyClass === "◎ RadianceDigitalCinemaWrite") {
			const onNodeCreated = nodeType.prototype.onNodeCreated;
			nodeType.prototype.onNodeCreated = function () {
				const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;

				const writeModeWidget   = this.widgets.find(w => w.name === "write_mode");
				const formatWidget      = this.widgets.find(w => w.name === "output_format");
				const qualityWidget     = this.widgets.find(w => w.name === "quality");
				const fpsWidget         = this.widgets.find(w => w.name === "fps");
				const startFrameWidget  = this.widgets.find(w => w.name === "start_frame");
				const bitDepthWidget    = this.widgets.find(w => w.name === "bit_depth");
				const compressionWidget = this.widgets.find(w => w.name === "compression");
				const alphaModeWidget   = this.widgets.find(w => w.name === "alpha_mode");
				const metadataWidget    = this.widgets.find(w => w.name === "custom_metadata");

				// FIX 3: guard against undefined before accessing .type
				const allWidgets = [
					writeModeWidget, formatWidget, qualityWidget, fpsWidget,
					startFrameWidget, bitDepthWidget, compressionWidget,
					alphaModeWidget, metadataWidget,
				];
				allWidgets.forEach(w => { if (w) w.origType = w.type; });

				// FIX 2: helper uses "hidden" (valid ComfyUI token) not "converted-widget"
				const setVisible = (w, visible) => {
					if (!w) return;
					w.type = visible ? (w.origType || "text") : "hidden";
				};

				const updateWidgets = () => {
					const mode   = writeModeWidget ? writeModeWidget.value : "Video";
					const fmt    = formatWidget    ? formatWidget.value    : "";
					const is_exr = fmt.includes("EXR");
					const is_png = fmt.includes("PNG");
					const is_jpg = fmt.includes("JPEG");

					const isVideo    = mode === "Video";
					const isSequence = mode === "Sequence";
					const isSingle   = mode === "Single Image";
					const isSeqLike  = isSequence || isSingle;

					// Quality label context
					if (qualityWidget) {
						qualityWidget.label = isVideo ? "QUALITY (CRF)" : is_jpg ? "QUALITY (JPEG %)" : "QUALITY";
					}

					// FPS only relevant for video
					setVisible(fpsWidget, isVideo);

					// Start frame only relevant for sequences
					setVisible(startFrameWidget, isSequence);

					// EXR/PNG-specific controls only in sequence/single modes
					setVisible(bitDepthWidget,    isSeqLike && (is_exr || is_png));
					setVisible(compressionWidget, isSeqLike && is_exr);
					setVisible(alphaModeWidget,   isSeqLike);
					setVisible(metadataWidget,    isSeqLike);
				};

				if (writeModeWidget) writeModeWidget.callback = updateWidgets;
				if (formatWidget)    formatWidget.callback    = updateWidgets;

				// Initial state — defer one tick so widgets are fully initialised
				setTimeout(updateWidgets, 20);

				return r;
			};
		}
	}
});
