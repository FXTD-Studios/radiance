import { app } from "../../scripts/app.js";
import { ComfyWidgets } from "../../scripts/widgets.js";

app.registerExtension({
    name: "Radiance.ShowText",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        const comfyClass = nodeType.comfyClass || nodeType.ComfyClass || nodeData.name;
        if (comfyClass === "RadianceShowText") {
            // Hook into onExecuted to receive the text from the Python backend
            const onExecuted = nodeType.prototype.onExecuted;
            nodeType.prototype.onExecuted = function (message) {
                onExecuted?.apply(this, arguments);

                if (this.widgets) {
                    // Remove existing text_display widgets to prevent duplicates
                    const pos = this.widgets.findIndex((w) => w.name === "text_display");
                    if (pos !== -1) {
                        for (let i = pos; i < this.widgets.length; i++) {
                            this.widgets[i].onRemove?.();
                        }
                        this.widgets.length = pos;
                    }
                }

                if (message && message.text) {
                    let widget = this.widgets?.find((w) => w.name === "text_display");
                    if (!widget) {
                        // Create a new STRING widget if it doesn't exist
                        widget = ComfyWidgets["STRING"](
                            this, "text_display", ["STRING", { multiline: true }], app
                        ).widget;

                        // Style it to look like a read-only display
                        if (widget.inputEl) {
                            widget.inputEl.readOnly = true;
                            widget.inputEl.style.opacity = 0.9;
                            widget.inputEl.style.backgroundColor = "transparent";
                            widget.inputEl.style.border = "1px solid #444";
                            widget.inputEl.style.borderRadius = "4px";
                            widget.inputEl.style.padding = "6px";
                            widget.inputEl.style.boxSizing = "border-box";
                            widget.inputEl.style.marginTop = "10px";
                            widget.inputEl.style.minHeight = "60px";
                            widget.inputEl.style.resize = "vertical";
                        }
                    }

                    // Join all text items with newlines if it's a list
                    widget.value = message.text.join("\n\n");

                    // Auto-resize node to fit content
                    if (this.computeSize) {
                        const sz = this.computeSize();
                        // ensure minimum size
                        sz[0] = Math.max(sz[0], this.size[0]);
                        sz[1] = Math.max(sz[1], this.size[1]);
                        this.onResize?.(sz);
                    }
                    app.graph.setDirtyCanvas(true, true);
                }
            };
        }
    }
});
