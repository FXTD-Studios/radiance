import { app } from "../../scripts/app.js";

import {
    forceWidgetReinsert as _forceWidgetReinsert,
    setWidgetVisible as _setWidgetVisible,
} from "./radiance_widget_utils.js";

// Widget helpers now live in radiance_widget_utils.js; this module's only
// local difference was the "number" fallback type, which is passed through.
function setWidgetVisible(widget, visible, node) {
    // Forward the changed-boolean — callers gate refreshNodeSize on it.
    return _setWidgetVisible(widget, visible, node, { fallbackType: "number" });
}

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
 *
 *  2026-07: RadianceWrite per-format widget visibility rewritten from scratch.
 *          The previous block targeted "RadianceDigitalCinemaWrite" and its
 *          pre-v3 write_mode/bit_depth/alpha_mode widgets, none of which exist
 *          on that class in the Beta (reduced to a 4-parameter shim) -- dead
 *          code that never matched anything. The full widget set now lives on
 *          "RadianceWrite" behind a single flat `format` dropdown, so the new
 *          version derives the group from the format's prefix directly.
 */

// Mirrors radiance/core/video.py VIDEO_EXTENSIONS exactly. The previous seven
// entries disagreed with Python in both directions -- it listed ".webp", which
// Python reads as a still image, and omitted ".mxf", ".m2ts" and everything
// else a camera writes -- so the frontend hid the wrong widgets for the files
// this pack exists to open. tests/test_read_surface.py parses both lists and
// fails if they drift apart again.
const VIDEO_EXTENSIONS = [".asf", ".avi", ".braw", ".dv", ".flv", ".gif", ".m2ts", ".m2v", ".m4v", ".mkv", ".mov", ".mp4", ".mpeg", ".mpg", ".mts", ".mxf", ".ogv", ".r3d", ".ts", ".vob", ".webm", ".wmv", ".y4m", ".yuv"];

// FIX 6: Intercept browser file inputs to allow selecting images, videos, and high-fidelity sequences (EXR/DPX/HDR)
try {
	const descriptor = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, "accept");
	if (descriptor && descriptor.set) {
		const originalSet = descriptor.set;
		Object.defineProperty(HTMLInputElement.prototype, "accept", {
			configurable: true,
			enumerable: true,
			get: descriptor.get,
			set: function (value) {
				if (value === "image/*" || value === "video/*") {
					value = "image/*,video/*,.exr,.dpx,.hdr";
				}
				originalSet.call(this, value);
			}
		});
	}
} catch (e) {
	console.warn("[Radiance.IO] Failed to patch HTMLInputElement.accept", e);
}

// Previously this replaced HTMLInputElement.prototype.click outright, which put
// a Radiance frame on every input click in the whole application, ComfyUI's and
// every other extension's. A capture-phase listener reaches the same inputs just
// before the picker opens without touching a shared prototype.
document.addEventListener("click", (ev) => {
	const el = ev.target;
	if (el instanceof HTMLInputElement && el.type === "file" &&
		(el.accept === "image/*" || el.accept === "video/*")) {
		el.accept = "image/*,video/*,.exr,.dpx,.hdr";
	}
}, true);

// FIX 7: Sleek Obsidian & Neon SVG placeholders for unsupported formats (Videos, EXR, DPX, HDR Sequences)
const VIDEO_PLACEHOLDER_SVG = `data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 512 512" width="512" height="512">
  <defs>
    <linearGradient id="bgGrad" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="%230a0f1d"/>
      <stop offset="100%" stop-color="%2307070a"/>
    </linearGradient>
    <linearGradient id="glowGrad" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="%2300f2ff"/>
      <stop offset="100%" stop-color="%23ff007f"/>
    </linearGradient>
    <filter id="glow" x="-20%" y="-20%" width="140%" height="140%">
      <feGaussianBlur stdDeviation="15" result="blur" />
      <feComposite in="SourceGraphic" in2="blur" operator="over" />
    </filter>
  </defs>
  <rect width="512" height="512" rx="24" fill="url(%23bgGrad)" stroke="%231a2035" stroke-width="2"/>
  <g opacity="0.05">
    <path d="M0 64h512M0 128h512M0 192h512M0 256h512M0 320h512M0 384h512M0 448h512" stroke="%23ffffff" stroke-width="1"/>
    <path d="M64 0v512M128 0v512M192 0v512M256 0v512M320 0v512M384 0v512M448 0v512" stroke="%23ffffff" stroke-width="1"/>
  </g>
  <circle cx="256" cy="220" r="80" fill="url(%23glowGrad)" opacity="0.15" filter="url(%23glow)"/>
  <g filter="url(%23glow)" transform="translate(186, 150)">
    <rect x="-10" y="-10" width="160" height="140" rx="16" fill="none" stroke="url(%23glowGrad)" stroke-width="6"/>
    <rect x="5" y="5" width="20" height="20" rx="4" fill="%2300f2ff" opacity="0.8"/>
    <rect x="115" y="5" width="20" height="20" rx="4" fill="%2300f2ff" opacity="0.8"/>
    <rect x="5" y="95" width="20" height="20" rx="4" fill="%23ff007f" opacity="0.8"/>
    <rect x="115" y="95" width="20" height="20" rx="4" fill="%23ff007f" opacity="0.8"/>
    <path d="M60 35 L100 60 L60 85 Z" fill="url(%23glowGrad)" stroke="%23ffffff" stroke-width="2" stroke-linejoin="round"/>
  </g>
  <text x="256" y="370" text-anchor="middle" fill="%23ffffff" font-family="'Inter', -apple-system, sans-serif" font-weight="800" font-size="28" letter-spacing="4">VIDEO SOURCE</text>
  <text x="256" y="405" text-anchor="middle" fill="%2300f2ff" font-family="'Courier New', monospace" font-weight="700" font-size="16" opacity="0.8">LOADED IN RADIANCE</text>
  <rect x="186" y="435" width="140" height="30" rx="15" fill="%23141a2e" stroke="%2300f2ff" stroke-width="1.5" opacity="0.9"/>
  <text x="256" y="455" text-anchor="middle" fill="%23ffffff" font-family="'Inter', -apple-system, sans-serif" font-weight="700" font-size="12" letter-spacing="1">MP4 / MOV / MKV</text>
</svg>`;

const SEQUENCE_PLACEHOLDER_SVG = `data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 512 512" width="512" height="512">
  <defs>
    <linearGradient id="bgGrad" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="%23110d05"/>
      <stop offset="100%" stop-color="%23070502"/>
    </linearGradient>
    <linearGradient id="glowGrad" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="%23eab308"/>
      <stop offset="100%" stop-color="%23ef4444"/>
    </linearGradient>
    <filter id="glow" x="-20%" y="-20%" width="140%" height="140%">
      <feGaussianBlur stdDeviation="15" result="blur" />
      <feComposite in="SourceGraphic" in2="blur" operator="over" />
    </filter>
  </defs>
  <rect width="512" height="512" rx="24" fill="url(%23bgGrad)" stroke="%233a2a0a" stroke-width="2"/>
  <g opacity="0.05">
    <path d="M0 64h512M0 128h512M0 192h512M0 256h512M0 320h512M0 384h512M0 448h512" stroke="%23ffffff" stroke-width="1"/>
    <path d="M64 0v512M128 0v512M192 0v512M256 0v512M320 0v512M384 0v512M448 0v512" stroke="%23ffffff" stroke-width="1"/>
  </g>
  <circle cx="256" cy="220" r="80" fill="url(%23glowGrad)" opacity="0.15" filter="url(%23glow)"/>
  <g filter="url(%23glow)" transform="translate(196, 140)">
    <rect x="20" y="-10" width="100" height="110" rx="12" fill="%23141a2e" stroke="%23ef4444" stroke-width="3" opacity="0.5"/>
    <rect x="10" y="5" width="100" height="110" rx="12" fill="%23141a2e" stroke="%23eab308" stroke-width="3" opacity="0.8"/>
    <rect x="0" y="20" width="100" height="110" rx="12" fill="%23141a2e" stroke="url(%23glowGrad)" stroke-width="4"/>
    <circle cx="50" cy="65" r="18" fill="url(%23glowGrad)" opacity="0.8"/>
    <path d="M20 100 L45 75 L60 90 L80 65 L100 95 Z" fill="url(%23bgGrad)" opacity="0.9"/>
  </g>
  <text x="256" y="370" text-anchor="middle" fill="%23ffffff" font-family="'Inter', -apple-system, sans-serif" font-weight="800" font-size="28" letter-spacing="4">IMAGE SEQUENCE</text>
  <text x="256" y="405" text-anchor="middle" fill="%23eab308" font-family="'Courier New', monospace" font-weight="700" font-size="16" opacity="0.8">LOADED IN RADIANCE</text>
  <rect x="186" y="435" width="140" height="30" rx="15" fill="%23241c0e" stroke="%23eab308" stroke-width="1.5" opacity="0.9"/>
  <text x="256" y="455" text-anchor="middle" fill="%23ffffff" font-family="'Inter', -apple-system, sans-serif" font-weight="700" font-size="12" letter-spacing="1">EXR / DPX / HDR</text>
</svg>`;

// Helper function to resolve placeholders dynamically
function resolvePlaceholder(srcVal) {
	if (typeof srcVal !== "string") return srcVal;
	if (srcVal.includes("/api/view") || srcVal.includes("/view")) {
		try {
			const url = new URL(srcVal, window.location.origin);
			const filename = url.searchParams.get("filename") || "";
			const lowerFile = filename.toLowerCase();
			if (lowerFile.endsWith(".mp4") || lowerFile.endsWith(".mov") || lowerFile.endsWith(".mkv") || lowerFile.endsWith(".webm") || lowerFile.endsWith(".avi")) {
				return VIDEO_PLACEHOLDER_SVG;
			} else if (lowerFile.endsWith(".exr") || lowerFile.endsWith(".dpx") || lowerFile.endsWith(".hdr")) {
				return SEQUENCE_PLACEHOLDER_SVG;
			}
		} catch (e) {}
	}
	return srcVal;
}

// Hook HTMLImageElement src property to dynamically replace non-image preview assets
try {
	const srcDescriptor = Object.getOwnPropertyDescriptor(HTMLImageElement.prototype, "src");
	if (srcDescriptor && srcDescriptor.set) {
		const originalSrcSet = srcDescriptor.set;
		Object.defineProperty(HTMLImageElement.prototype, "src", {
			configurable: true,
			enumerable: true,
			get: srcDescriptor.get,
			set: function (value) {
				const resolved = resolvePlaceholder(value);
				originalSrcSet.call(this, resolved);
			}
		});
	}
} catch (e) {
	console.warn("[Radiance.IO] Failed to patch HTMLImageElement.src", e);
}

// Hook Element.prototype.setAttribute as a fail-safe uploader fallback —
// img.setAttribute("src", ...) bypasses the HTMLImageElement.src property
// hook above. (Restored from beta/main; the inline widget helpers that used
// to follow it live in radiance_widget_utils.js now.)
try {
	const originalSetAttribute = Element.prototype.setAttribute;
	Element.prototype.setAttribute = function (name, value) {
		let resolved = value;
		if (name === "src" && this.tagName === "IMG") {
			resolved = resolvePlaceholder(value);
		}
		return originalSetAttribute.call(this, name, resolved);
	};
} catch (e) {
	console.warn("[Radiance.IO] Failed to patch Element.prototype.setAttribute", e);
}
function refreshNodeSize(node) {
	if (!node.computeSize) return;
	const sz = node.computeSize();
	const newWidth = Math.max(node.size[0], sz[0]);
	const newHeight = sz[1];
	if (node.size[0] === newWidth && node.size[1] === newHeight) return;
	node.setSize([newWidth, newHeight]);
	node.setDirtyCanvas(true, true);
}

app.registerExtension({
	name: "Radiance.IO",
	async beforeRegisterNodeDef(nodeType, nodeData, app) {

		// ── RadianceRead ────────────────────────────────────────────
		// Fifteen widgets, most of which do not apply to whatever you just
		// picked. Nuke's Read shows you the file's format, range and layers
		// and hides the rest; this does the same three things:
		//   1. RELOAD button
		//   2. only the widgets that apply to the detected media type
		//   3. an info line, and a layer dropdown filled from the actual file
		if (nodeData.name === "RadianceRead") {
			const onNodeCreated = nodeType.prototype.onNodeCreated;
			nodeType.prototype.onNodeCreated = function () {
				const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
				const node = this;

				const reloadWidget = node.widgets?.find(w => w.name === "reload");
				if (reloadWidget) {
					node.addWidget("button", "RELOAD", "reload", () => {
						reloadWidget.value = (reloadWidget.value || 0) + 1;
						node.__radianceReadRefresh?.(true);
						node.setDirtyCanvas(true);
					});
				}

				const get = (name) => node.widgets?.find(w => w.name === name);
				const browseW = get("browse");
				const pathW = get("path");
				const mediaW = get("media_type");
				const layerW = get("layer");

				// The layer widget is a STRING on the Python side on purpose --
				// ComfyUI validates a combo's value against the list it was
				// built with, so a workflow saved with layer="diffuse" would
				// refuse to load on a machine whose INPUT_TYPES never saw that
				// file. Here it gains a dropdown; the value stays a string, and
				// typing a layer name by hand still works.
				if (layerW) {
					layerW.options = layerW.options || {};
					layerW.__radianceChoices = ["auto"];
				}

				const setLayerChoices = (choices) => {
					if (!layerW) return;
					layerW.__radianceChoices = choices?.length ? choices : ["auto"];
					// Offer them through the widget's own value cycle so the
					// user can click through without typing.
					layerW.options.values = layerW.__radianceChoices;
				};

				const resolvedPath = () => {
					const b = (browseW?.value || "").trim();
					if (b && b !== "none") return b;
					return (pathW?.value || "").trim();
				};

				const kindOf = (p) => {
					const v = (p || "").toLowerCase();
					if (/[%#*]/.test(v) || v.endsWith("/") || v.endsWith("\\")) return "sequence";
					if (v.endsWith(".exr") || v.endsWith(".sxr")) return "exr";
					if (VIDEO_EXTENSIONS.some(e => v.endsWith(e))) return "video";
					if (v) return "image";
					return "";
				};

				node.__radianceReadInfo = "";

				// ── Inline video preview ────────────────────────────────
				// A real <video> for what the browser can decode (H.264/H.265
				// MP4 & MOV, WebM); when the codec is beyond it (ProRes,
				// DNxHR, MXF) the element's error event swaps in a poster
				// JPEG of the first frame from /radiance/media/poster.
				const previewBox = document.createElement("div");
				previewBox.style.cssText =
					"width:100%;aspect-ratio:16/9;background:#07070a;" +
					"border-radius:8px;overflow:hidden;display:none;";
				const videoEl = document.createElement("video");
				videoEl.muted = true;
				videoEl.loop = true;
				videoEl.controls = true;
				videoEl.playsInline = true;
				videoEl.preload = "metadata";
				videoEl.style.cssText = "width:100%;height:100%;object-fit:contain;";
				const posterEl = document.createElement("img");
				posterEl.style.cssText = "width:100%;height:100%;object-fit:contain;display:none;";
				previewBox.appendChild(videoEl);
				previewBox.appendChild(posterEl);
				videoEl.addEventListener("error", () => {
					// Browser cannot decode this codec; fall back to a still.
					if (!node.__radiancePreviewPath) return;
					videoEl.style.display = "none";
					posterEl.style.display = "block";
					posterEl.src = `/radiance/media/poster?path=${encodeURIComponent(node.__radiancePreviewPath)}`;
				});
				posterEl.addEventListener("error", () => {
					previewBox.style.display = "none";   // nothing previewable
				});
				const previewWidget = node.addDOMWidget
					? node.addDOMWidget("video_preview", "custom", previewBox, { serialize: false })
					: null;

				const setPreview = (p, isVid) => {
					if (!previewWidget) return;
					if (!isVid || !p) {
						if (node.__radiancePreviewPath) {
							node.__radiancePreviewPath = "";
							videoEl.removeAttribute("src");
							videoEl.load?.();
							posterEl.removeAttribute("src");
							previewBox.style.display = "none";
							refreshNodeSize(node);
						}
						return;
					}
					if (node.__radiancePreviewPath === p) return;
					node.__radiancePreviewPath = p;
					posterEl.style.display = "none";
					videoEl.style.display = "block";
					previewBox.style.display = "block";
					videoEl.src = `/radiance/media/preview?path=${encodeURIComponent(p)}`;
					refreshNodeSize(node);
				};

				const applyVisibility = () => {
					const chosen = (mediaW?.value || "Auto");
					const kind = chosen === "Auto" ? kindOf(resolvedPath()) : chosen.toLowerCase();
					const isVid = kind === "video";
					const isSeq = kind === "sequence";
					const isExr = kind === "exr";
					const framed = isVid || isSeq;

					setWidgetVisible(get("start_frame"), framed, node);
					setWidgetVisible(get("end_frame"), framed, node);
					setWidgetVisible(get("frame_step"), framed, node);
					setWidgetVisible(get("max_video_frames"), isVid, node);
					setWidgetVisible(get("missing_frames"), isSeq, node);
					setWidgetVisible(layerW, isExr || isSeq, node);
					// Premultiplied is only meaningful where an alpha can exist.
					setWidgetVisible(get("premultiplied"), kind !== "video" || isExr, node);
					setWidgetVisible(reloadWidget, false, node);

					// raw = "no transform": the colour-space dropdown is ignored
					// by the Python reader, so say so instead of looking live.
					const rawW = get("raw");
					const csW = get("color_space");
					if (csW) {
						const isRaw = (rawW?.value || "") === "raw (no transform)";
						csW.disabled = isRaw;
						csW.label = isRaw ? "color_space (ignored — raw)" : "color_space";
					}

					setPreview(resolvedPath(), isVid);
				};

				let pending = null;
				const refresh = (force) => {
					applyVisibility();
					const p = resolvedPath();
					if (!p) { node.__radianceReadInfo = ""; node.setDirtyCanvas(true); return; }
					if (!force && p === node.__radianceReadLastPath) return;
					node.__radianceReadLastPath = p;

					clearTimeout(pending);
					pending = setTimeout(async () => {
						try {
							const q = `?path=${encodeURIComponent(p)}`;
							const res = await fetch(`/radiance/media/info${q}`);
							if (res.ok) {
								const data = await res.json();
								node.__radianceReadInfo = data.summary || data.error || "";
							} else {
								// 403 outside the allowed roots is expected on a
								// facility NAS. The node still reads the file; it
								// just cannot preview what is in it from here.
								node.__radianceReadInfo = "";
							}
							if (kindOf(p) === "exr") {
								const lres = await fetch(`/radiance/media/layers${q}`);
								if (lres.ok) setLayerChoices((await lres.json()).layers);
							}
						} catch (e) {
							node.__radianceReadInfo = "";
						}
						node.setDirtyCanvas(true);
					}, 120);
				};
				node.__radianceReadRefresh = refresh;

				for (const w of [browseW, pathW, mediaW, get("raw")]) {
					if (!w) continue;
					const prev = w.callback;
					w.callback = function (...args) {
						const out = prev ? prev.apply(this, args) : undefined;
						refresh(false);
						return out;
					};
				}

				setTimeout(() => refresh(true), 50);
				return r;
			};

			// The info line, drawn where Nuke's Read puts its format readout.
			const onDrawForeground = nodeType.prototype.onDrawForeground;
			nodeType.prototype.onDrawForeground = function (ctx) {
				onDrawForeground?.apply(this, arguments);
				if (this.flags?.collapsed || !this.__radianceReadInfo) return;
				ctx.save();
				ctx.font = "11px monospace";
				ctx.fillStyle = "#8ab4d8";
				ctx.textAlign = "left";
				const max = Math.max(40, this.size[0] - 20);
				let text = this.__radianceReadInfo;
				while (text.length > 8 && ctx.measureText(text).width > max) {
					text = text.slice(0, -2);
				}
				if (text !== this.__radianceReadInfo) text += "…";
				ctx.fillText(text, 10, this.size[1] - 6);
				ctx.restore();
			};
		}

		// 1. Digital Cinema Read — label intelligence
		// FIX 1: was "◎ RadianceDigitalCinemaRead" — ◎ removed from Python mapping key.
		if (nodeData.name === "RadianceDigitalCinemaRead") {
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

		// 2. RadianceWrite — per-format widget visibility
		//
		// ALBABIT-FIX: this block previously targeted "RadianceDigitalCinemaWrite"
		// and looked for widgets (write_mode, bit_depth, alpha_mode...) that only
		// existed on that class in the pre-v3 fork. In the Beta, that class was
		// reduced to a 4-parameter shim, and all the real widgets live on
		// "RadianceWrite" itself, behind a single flat 'format' dropdown (prefix
		// "IMG │"/"SEQ │"/"VID │" — no separate write_mode widget). The old code
		// was dead: it never matched a class that actually has these widgets.
		// Rewritten from scratch against the current widget set, deriving the
		// group directly from the format prefix rather than a second widget.
		if (nodeData.name === "RadianceWrite") {
			const onNodeCreated = nodeType.prototype.onNodeCreated;
			nodeType.prototype.onNodeCreated = function () {
				const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
				const node = this;

				const formatWidget       = node.widgets.find(w => w.name === "format");
				const fpsWidget          = node.widgets.find(w => w.name === "fps");
				const qualityWidget      = node.widgets.find(w => w.name === "quality");
				const exrCompWidget      = node.widgets.find(w => w.name === "exr_compression");
				const startFrameWidget   = node.widgets.find(w => w.name === "start_frame");
				const framePaddingWidget = node.widgets.find(w => w.name === "frame_padding");
				const audioSourceWidget  = node.widgets.find(w => w.name === "audio_source");
				const broadcastWidget    = node.widgets.find(w => w.name === "broadcast_safe");
				const versionWidget      = node.widgets.find(w => w.name === "version");
				const outputPathWidget   = node.widgets.find(w => w.name === "output_path");
				const filenameWidget     = node.widgets.find(w => w.name === "filename");
				const overwriteWidget    = node.widgets.find(w => w.name === "overwrite");

				// ── Resolved-path readout ───────────────────────────────
				// output_path + filename + version + format combine invisibly;
				// this asks the Python side (the same logic that will write)
				// what will actually land on disk, and draws it on the node.
				let resolvePending = null;
				const refreshResolvedPath = () => {
					clearTimeout(resolvePending);
					resolvePending = setTimeout(async () => {
						try {
							const q = new URLSearchParams({
								output_path: outputPathWidget?.value || "",
								format: formatWidget?.value || "",
								filename: filenameWidget?.value || "",
								version: String(versionWidget?.value ?? 1),
								start_frame: String(startFrameWidget?.value ?? 1001),
								frame_padding: String(framePaddingWidget?.value ?? 4),
								overwrite: String(!!overwriteWidget?.value),
							});
							const res = await fetch(`/radiance/media/resolve_write?${q}`);
							if (res.ok) {
								const data = await res.json();
								node.__radianceWriteTarget = data.path || "";
								node.__radianceWriteNote = data.note || "";
							}
						} catch (e) {
							node.__radianceWriteTarget = "";
							node.__radianceWriteNote = "";
						}
						node.setDirtyCanvas(true);
					}, 150);
				};

				const updateWidgets = () => {
					const fmt = formatWidget ? formatWidget.value : "";
					const isImg = fmt.startsWith("IMG");
					const isSeq = fmt.startsWith("SEQ");
					const isVid = fmt.startsWith("VID");
					const isExr = fmt.includes("EXR");
					// Legal-range clamping is an 8-bit/video concept; the Python
					// side ignores broadcast_safe for float formats (EXR, HDR,
					// 32-bit float TIFF, DPX), so do not show it for them.
					const isFloatFmt = isExr || fmt.includes("HDR") ||
						fmt.includes("32-bit float") || fmt.includes("DPX");
					// quality only actually does something for video CRF and
					// JPEG/WEBP images (nodes_io.py::_save_pil_image) -- every
					// other image/sequence format ignores it entirely.
					const isJpgWebp = isImg && (fmt.includes("JPEG") || fmt.includes("WEBP"));

					if (qualityWidget) {
						qualityWidget.label = isVid ? "quality (CRF)"
							: isJpgWebp ? "quality (JPEG/WEBP)"
							: "quality";
					}
					if (versionWidget) {
						const v = Math.max(0, versionWidget.value | 0);
						versionWidget.label = `version  (v${String(v).padStart(4, "0")})`;
					}

					let changed = false;
					if (setWidgetVisible(fpsWidget,          isVid, node)) changed = true;
					if (setWidgetVisible(qualityWidget,       isVid || isJpgWebp, node)) changed = true;
					if (setWidgetVisible(exrCompWidget,       isExr, node)) changed = true;
					if (setWidgetVisible(startFrameWidget,    isSeq, node)) changed = true;
					if (setWidgetVisible(framePaddingWidget,  isSeq, node)) changed = true;
					if (setWidgetVisible(audioSourceWidget,   isVid, node)) changed = true;
					if (setWidgetVisible(broadcastWidget,     !isFloatFmt, node)) changed = true;

					if (changed) refreshNodeSize(node);

					// updateWidgets is also polled every 250 ms; only re-ask
					// the server when an ingredient of the path changed.
					const sig = [
						outputPathWidget?.value, filenameWidget?.value,
						versionWidget?.value, fmt,
						startFrameWidget?.value, framePaddingWidget?.value,
						overwriteWidget?.value,
					].join(" ");
					if (sig !== node.__radianceWriteSig) {
						node.__radianceWriteSig = sig;
						refreshResolvedPath();
					}
				};

				if (formatWidget) {
					const origCallback = formatWidget.callback;
					formatWidget.callback = function (...args) {
						const res = origCallback ? origCallback.apply(this, args) : undefined;
						updateWidgets();
						return res;
					};
				}

				// Poll as a state-based fallback (undo/redo, workflow load,
				// upstream mute/bypass) -- same pattern as radiance_vae_widgets.js.
				node._radianceWriteSyncInterval = setInterval(updateWidgets, 250);
				const origOnRemoved = node.onRemoved;
				node.onRemoved = function () {
					if (node._radianceWriteSyncInterval) {
						clearInterval(node._radianceWriteSyncInterval);
						node._radianceWriteSyncInterval = null;
					}
					if (origOnRemoved) origOnRemoved.apply(this, arguments);
				};

				setTimeout(updateWidgets, 150);
				return r;
			};

			// The resolved-path readout, drawn where Nuke's Write shows its
			// file field: exactly what will land on disk, plus one note line.
			const onDrawForegroundWrite = nodeType.prototype.onDrawForeground;
			nodeType.prototype.onDrawForeground = function (ctx) {
				onDrawForegroundWrite?.apply(this, arguments);
				if (this.flags?.collapsed || !this.__radianceWriteTarget) return;
				ctx.save();
				ctx.font = "11px monospace";
				ctx.textAlign = "left";
				const max = Math.max(40, this.size[0] - 20);
				const fit = (t) => {
					let s = t;
					// Elide from the LEFT: the tail of a path is the part that matters.
					while (s.length > 8 && ctx.measureText("…" + s).width > max) {
						s = s.slice(2);
					}
					return s === t ? t : "…" + s;
				};
				ctx.fillStyle = "#8fd18f";
				ctx.fillText(fit(this.__radianceWriteTarget), 10, this.size[1] - 18);
				if (this.__radianceWriteNote) {
					ctx.fillStyle = "#98a4b8";
					let note = this.__radianceWriteNote;
					while (note.length > 8 && ctx.measureText(note).width > max) {
						note = note.slice(0, -2);
					}
					ctx.fillText(note, 10, this.size[1] - 5);
				}
				ctx.restore();
			};

			// Re-apply after a saved workflow restores this node — onNodeCreated
			// runs before ComfyUI deserializes widget values.
			const onConfigure = nodeType.prototype.onConfigure;
			nodeType.prototype.onConfigure = function (info) {
				const r = onConfigure ? onConfigure.apply(this, arguments) : undefined;
				const node = this;
				const formatWidget = node.widgets?.find(w => w.name === "format");
				if (formatWidget && typeof formatWidget.callback === "function") {
					setTimeout(() => formatWidget.callback(formatWidget.value), 150);
					setTimeout(() => formatWidget.callback(formatWidget.value), 600);
				}
				return r;
			};
		}
	}
});
