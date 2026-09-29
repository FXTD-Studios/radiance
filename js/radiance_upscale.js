import { app } from "../../../scripts/app.js";

import {
    fitNodeSize,
    forceWidgetReinsert as _forceWidgetReinsert,
    setWidgetVisible as _setWidgetVisible,
} from "./radiance_widget_utils.js";

// Widget helpers now live in radiance_widget_utils.js; this module's only
// local difference was the "text" fallback type, which is passed through.
function setWidgetVisible(widget, visible, node) {
    _setWidgetVisible(widget, visible, node, { fallbackType: "text" });
}

/**
 * AI Upscale Widget Visibility (v1.0)
 * Hides SUPIR-specific widgets when a non-SUPIR model is selected.
 *
 * SUPIR-only widgets: sdxl_model_name, supir_prompt, vae, clip
 * These are shown only when model_name is one of the SUPIR model identifiers.
 *
 * Uses the same three-mechanism visibility pattern as radiance_io.js:
 *   1. widget.options.hidden      — Nodes 2.0 Vue reactive filter
 *   2. widget.hidden              — LiteGraph getLayoutWidgets() exclusion
 *   3. type="hidden"+computeSize  — physical height collapse (all widget types)
 *      + draw=()=>{}              — prevents text bleeding on STRING widgets
 *      + inputEl/element display  — hides DOM node for customtext widgets
 */

// ALBABIT-FIX: SUPIR model name identifiers — must match _SUPIR_MODELS in upscale.py.
const SUPIR_MODEL_NAMES = ["SUPIR-v0F_fp16", "SUPIR-v0Q_fp16"];

app.registerExtension({
	name: "Radiance.AIUpscale",
	async beforeRegisterNodeDef(nodeType, nodeData, app) {
		if (nodeData.name !== "RadianceAIUpscale") return;

		const onNodeCreated = nodeType.prototype.onNodeCreated;
		nodeType.prototype.onNodeCreated = function () {
			const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
			const node = this;

			const modelWidget       = this.widgets.find(w => w.name === "model_name");
			const sdxlModelWidget   = this.widgets.find(w => w.name === "sdxl_model_name");
			const supirPromptWidget = this.widgets.find(w => w.name === "supir_prompt");
			const vaeWidget         = this.widgets.find(w => w.name === "vae");
			const clipWidget        = this.widgets.find(w => w.name === "clip");

			const updateSupirWidgets = () => {
				const isSupir = modelWidget
					? SUPIR_MODEL_NAMES.includes(modelWidget.value)
					: false;

				setWidgetVisible(sdxlModelWidget,   isSupir, node);
				setWidgetVisible(supirPromptWidget, isSupir, node);
				setWidgetVisible(vaeWidget,         isSupir, node);
				setWidgetVisible(clipWidget,        isSupir, node);

				fitNodeSize(node);
			};

			if (modelWidget) {
				const origCb = modelWidget.callback;
				modelWidget.callback = function () {
					if (origCb) origCb.apply(this, arguments);
					updateSupirWidgets();
				};
			}

			// 100ms delay — matches radiance_io.js pattern to ensure all widgets
			// are fully initialised before the first hide/show pass.
			setTimeout(updateSupirWidgets, 100);

			return r;
		};
	}
});
