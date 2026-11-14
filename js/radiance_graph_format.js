// ◎ Radiance — open a saved graph in whichever format it was stored.
//
// ComfyUI has two graph formats. The UI workflow ({nodes: [...], links: [...]})
// is what app.loadGraphData opens. The API prompt ({id: {class_type, inputs}})
// is what a node receives as its hidden "prompt" input, and what the Project
// Manager node stored in every .rad it saved until it learnt to keep the UI
// workflow instead. loadGraphData cannot open an API prompt, so those library
// entries failed to open. app.loadApiJson can.

/**
 * True for an API-format prompt: an object of nodes keyed by id, each with a
 * class_type, and no UI "nodes" array.
 *
 * @param {*} graph  Parsed graph JSON.
 * @returns {boolean}
 */
export function isApiPromptGraph(graph) {
    if (!graph || typeof graph !== "object" || Array.isArray(graph)) return false;
    if (Array.isArray(graph.nodes)) return false;
    const values = Object.values(graph);
    return values.length > 0
        && values.every((node) => node && typeof node === "object" && typeof node.class_type === "string");
}

/**
 * Open a saved graph: UI workflows through loadGraphData, API prompts through
 * loadApiJson. An API prompt has no layout to merge, so appending one throws.
 *
 * @param {object} app      ComfyUI's app object.
 * @param {object} graph    Parsed graph JSON.
 * @param {boolean} append  Merge into the open graph instead of replacing it.
 * @param {string} name     Shown by ComfyUI as the workflow name.
 */
export function openSavedGraph(app, graph, append = false, name = "workflow") {
    if (!isApiPromptGraph(graph)) return app.loadGraphData(graph, append);
    if (append) {
        throw new Error("This workflow was saved in API format and has no layout to merge; open it instead.");
    }
    if (typeof app.loadApiJson !== "function") {
        throw new Error("This ComfyUI version cannot open API-format workflows.");
    }
    return app.loadApiJson(graph, name);
}

export default { isApiPromptGraph, openSavedGraph };
