import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import { fitNodeSize } from "./radiance_widget_utils.js";

// ALBABIT-FIX: the Sampler's video preview. sample() sends every latent frame
// at each step; this player loops them under the node, with a bar to pause
// and pick a moment. A node that only previews images never gets one.

// The node an execution id names: "538", or "12:538" inside a subgraph.
function nodeByExecutionId(id) {
    let graph = app.rootGraph;
    let node = null;
    for (const part of String(id).split(":")) {
        node = graph?.getNodeById(part);
        graph = node?.subgraph;
    }
    return node;
}

function element(tag, css) {
    const el = document.createElement(tag);
    el.style.cssText = css;
    return el;
}

function createPlayer(node) {
    const box = element("div", "width:100%;display:flex;flex-direction:column;gap:6px;");
    const canvas = element("canvas", "width:100%;border-radius:6px;background:#000;cursor:pointer;");
    const bar = element("div", "display:flex;align-items:center;gap:8px;height:16px;");
    const button = element("div", "flex:none;width:14px;cursor:pointer;color:#e2e2e6;font-size:11px;text-align:center;");
    const track = element("div", "position:relative;flex:1;height:4px;border-radius:2px;background:#40404a;cursor:pointer;");
    const fill = element("div", "position:absolute;left:0;top:0;bottom:0;border-radius:2px;background:#2f7cf6;");
    const knob = element("div", "position:absolute;top:50%;width:12px;height:12px;margin:-6px 0 0 -6px;border-radius:50%;background:#f5f5f8;");
    const time = element("div", "flex:none;min-width:38px;text-align:right;color:#96969e;font-size:11px;");
    track.append(fill, knob);
    bar.append(button, track, time);
    box.append(canvas, bar);
    node.addDOMWidget("animated_preview", "custom", box, { serialize: false });

    const ctx = canvas.getContext("2d");
    const player = { images: [], index: 0, fps: 3, playing: true, timer: null };

    const draw = () => {
        const image = player.images[player.index];
        if (!image) return;
        ctx.drawImage(image, 0, 0, canvas.width, canvas.height);
        const at = player.images.length > 1 ? player.index / (player.images.length - 1) : 0;
        fill.style.width = `${at * 100}%`;
        knob.style.left = `${at * 100}%`;
        time.textContent = `${(player.index / player.fps).toFixed(1)} s`;
        button.textContent = player.playing ? "❚❚" : "▶︎";
    };
    const setPlaying = (playing) => {
        player.playing = playing;
        clearInterval(player.timer);
        player.timer = playing
            ? setInterval(() => { player.index = (player.index + 1) % player.images.length; draw(); }, 1000 / player.fps)
            : null;
        draw();
    };
    const seek = (event) => {
        const rect = track.getBoundingClientRect();
        const at = Math.min(Math.max((event.clientX - rect.left) / rect.width, 0), 1);
        player.index = Math.round(at * (player.images.length - 1));
        draw();
    };

    canvas.addEventListener("click", () => setPlaying(!player.playing));
    button.addEventListener("click", () => setPlaying(!player.playing));
    // Dragging on the bar picks a moment and leaves the loop paused there.
    track.addEventListener("pointerdown", (event) => {
        event.stopPropagation();
        track.setPointerCapture(event.pointerId);
        setPlaying(false);
        seek(event);
    });
    track.addEventListener("pointermove", (event) => {
        if (track.hasPointerCapture(event.pointerId)) seek(event);
    });

    player.show = async (urls, fps) => {
        const images = urls.map((url) => Object.assign(new Image(), { src: url }));
        await Promise.all(images.map((image) => image.decode()));
        const first = player.images.length === 0;
        player.images = images;
        player.index %= images.length;
        canvas.width = images[0].naturalWidth;
        canvas.height = images[0].naturalHeight;
        if (first || fps !== player.fps) {
            player.fps = fps;
            setPlaying(player.playing);
        } else {
            draw();
        }
        if (first) fitNodeSize(node);
    };

    const onRemoved = node.onRemoved;
    node.onRemoved = function () {
        clearInterval(player.timer);
        return onRemoved?.apply(this, arguments);
    };
    return player;
}

app.registerExtension({
    name: "FXTD.RadianceSamplerPreview",
    setup() {
        api.addEventListener("radiance.sampler.preview", ({ detail }) => {
            const node = nodeByExecutionId(detail.node);
            if (node?.type !== "RadianceSamplerPro") return;
            node._radiancePreview ??= createPlayer(node);
            node._radiancePreview.show(detail.frames, detail.fps);
        });
    },
});
