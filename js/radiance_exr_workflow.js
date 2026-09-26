import { app } from "../../scripts/app.js";

/**
 * Radiance EXR Workflow Loader (v1.0)
 *
 * Makes .exr files behave like ComfyUI PNGs: dragging an EXR written by
 * RadianceWrite onto the canvas restores the embedded workflow.
 *
 * RadianceWrite (nodes_io.py) embeds "workflow" and "prompt" as EXR header
 * string attributes. This extension intercepts app.handleFile for .exr,
 * parses ONLY the EXR header in the browser (no pixel decode), extracts the
 * "workflow" attribute, and loads it via app.loadGraphData.
 *
 * EXR header layout (single-part):
 *   [0..3]  magic 0x76 0x2f 0x31 0x01
 *   [4..7]  version/flags
 *   then repeated attributes until an empty name byte:
 *     name\0  type\0  int32 size (LE)  value[size]
 * String attribute values are raw bytes (no length prefix inside value).
 */

const EXR_MAGIC = [0x76, 0x2f, 0x31, 0x01];
const MAX_HEADER_SCAN = 8 * 1024 * 1024; // sane upper bound for header walk

function parseExrHeaderAttributes(buffer) {
	const view = new DataView(buffer);
	const bytes = new Uint8Array(buffer);
	if (bytes.length < 9) return null;
	for (let i = 0; i < 4; i++) {
		if (bytes[i] !== EXR_MAGIC[i]) return null;
	}

	const attrs = {};
	let off = 8;
	const limit = Math.min(bytes.length, MAX_HEADER_SCAN);
	const readCString = () => {
		let end = off;
		while (end < limit && bytes[end] !== 0) end++;
		if (end >= limit) return null;
		const s = new TextDecoder("utf-8").decode(bytes.subarray(off, end));
		off = end + 1;
		return s;
	};

	while (off < limit) {
		if (bytes[off] === 0) break; // empty name → end of header
		const name = readCString();
		if (name === null) return attrs;
		const type = readCString();
		if (type === null) return attrs;
		if (off + 4 > limit) return attrs;
		const size = view.getInt32(off, true);
		off += 4;
		if (size < 0 || off + size > bytes.length) return attrs;
		if (type === "string") {
			attrs[name] = new TextDecoder("utf-8").decode(
				bytes.subarray(off, off + size)
			);
		}
		off += size;
	}
	return attrs;
}

async function tryLoadExrWorkflow(file) {
	try {
		// EXRs can be many gigabytes; workflow metadata lives in the header, so
		// never copy the full image into browser memory just to inspect it.
		const buffer = await file.slice(0, MAX_HEADER_SCAN).arrayBuffer();
		const attrs = parseExrHeaderAttributes(buffer);
		if (!attrs) return false;

		if (attrs.workflow) {
			const workflow = JSON.parse(attrs.workflow);
			await app.loadGraphData(workflow, true, true, file.name);
			return true;
		}
		if (attrs.prompt) {
			// No graph layout, but the API prompt is recoverable.
			const prompt = JSON.parse(attrs.prompt);
			if (app.loadApiJson) {
				app.loadApiJson(prompt, file.name);
				return true;
			}
		}
	} catch (err) {
		console.warn("[Radiance] EXR workflow extraction failed:", err);
	}
	return false;
}

app.registerExtension({
	name: "radiance.exr_workflow",
	async setup() {
		const origHandleFile = app.handleFile?.bind(app);
		if (!origHandleFile) return;
		app.handleFile = async function (file, ...rest) {
			if (file && /\.exr$/i.test(file.name || "")) {
				const loaded = await tryLoadExrWorkflow(file);
				if (loaded) return;
				// fall through — no embedded workflow; let ComfyUI handle it
			}
			return origHandleFile(file, ...rest);
		};
	},
});
