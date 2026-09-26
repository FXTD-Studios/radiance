import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

/**
 * ═══════════════════════════════════════════════════════════════════════════════
 *                    RADIANCE VFX MASK EDITOR (PRO)
 *                      Radiance © 2024-2026
 *
 * High-end masking tool featuring Vector Rotoscoping (Polygons/Splines),
 * Procedural Qualifiers (HSL), and Soft-Brush painting with full undo/redo.
 * ═══════════════════════════════════════════════════════════════════════════════
 */

// ─── Configuration ───────────────────────────────────────────────────────────
const MAX_UNDO_STATES = 30;
const ZOOM_MIN = 0.05;
const ZOOM_MAX = 20;
const ZOOM_FACTOR = 1.15;
const POINT_HIT_RADIUS_PX = 10; // screen-space pixels
const FLOOD_FILL_YIELD_INTERVAL = 50000; // yield every N pixels to avoid blocking

class RadianceMaskEditor {
    constructor() {
        this.node = null;
        this.imageEl = null;
        this._cachedSourceData = null; // Cached source image pixel data

        // Display
        this.zoom = 1.0;
        this.panX = 0;
        this.panY = 0;
        this.isPanning = false;

        // Tools: brush, eraser, polygon, magic_wand
        this.tool = 'brush';

        // Brush Settings
        this.brushSize = 50;
        this.brushOpacity = 1.0;
        this.brushHardness = 0.2;
        this.isDrawing = false;
        this.lastPaintPos = null; // Track last paint coordinate for stroke interpolation

        // Vector State
        this.polygons = [];       // Array of { points: [{x,y}], closed: bool, feather: float }
        this.currentPoly = null;
        this.activePoint = null;

        // Vertex Dragging State
        this.isDraggingVertex = false;
        this.draggingPoly = null;
        this.draggingPointIndex = -1;

        // View Mode
        this.viewMode = 'overlay'; // overlay, matte, false_color

        // Qualifiers (HSL)
        this.qualifier = {
            enabled: false,
            h: 0.5, hW: 0.1, hS: 0.05,
            s: 0.5, sW: 0.5, sS: 0.1,
            l: 0.5, lW: 0.5, lS: 0.1
        };

        // History (undo/redo)
        this.history = [];
        this.historyIndex = -1;

        // RAF gating for paint performance
        this._rafPending = false;

        // Bound listener refs for cleanup
        this._boundListeners = [];

        // Unique ID counter for slider elements (avoids DOM ID collisions)
        this._sliderId = 0;

        this.createUI();
        this.setupEventListeners();
    }

    // ═══════════════════════════════════════════════════════════════════════════
    //                              UI CREATION
    // ═══════════════════════════════════════════════════════════════════════════

    createUI() {
        // Overlay container matching Radiance standard
        this.overlay = document.createElement("div");
        this.overlay.className = "radiance-mask-overlay radiance-glass-dock";
        this.overlay.style.cssText = `
            position: fixed; top: 0; left: 0; width: 100vw; height: 100vh;
            background: rgba(10, 10, 15, 0.95);
            z-index: 9999; display: none; align-items: center; justify-content: center;
            font-family: 'Inter', sans-serif; backdrop-filter: blur(14px);
        `;

        this.container = document.createElement("div");
        this.container.style.cssText = `
            width: 95vw; height: 95vh;
            background: #0a0a0f; border-radius: 8px;
            display: flex; flex-direction: column;
            box-shadow: 0 10px 40px rgba(0,0,0,0.8);
            border: 1px solid rgba(60, 70, 100, 0.25);
            overflow: hidden;
        `;

        // ── TOPBAR ─────────────────────────────────────────────────────────────
        const topbar = document.createElement("div");
        topbar.style.cssText = `
            flex: 0 0 44px; background: rgba(16, 16, 24, 0.95);
            display: flex; align-items: center; justify-content: space-between;
            padding: 0 16px; border-bottom: 1px solid rgba(60, 70, 100, 0.3);
        `;

        const title = document.createElement("div");
        title.innerHTML = "<span style='color:#00a8ff; font-weight:600;'>◎ Radiance</span> VFX Mask Editor";
        title.style.color = "#fff"; title.style.fontSize = "14px";

        const actionsDiv = document.createElement("div");
        actionsDiv.style.display = "flex"; actionsDiv.style.gap = "8px";

        const closeBtn = this.createBtn("Cancel", () => this.hide(), true);
        const saveBtn = this.createBtn("Save & Return", () => this.saveMask());
        saveBtn.style.background = "#00a8ff"; saveBtn.style.color = "#fff"; saveBtn.style.border = "none";

        actionsDiv.appendChild(closeBtn);
        actionsDiv.appendChild(saveBtn);

        topbar.appendChild(title);
        topbar.appendChild(actionsDiv);

        // ── MAIN CONTENT ───────────────────────────────────────────────────────
        const contentArea = document.createElement("div");
        contentArea.style.cssText = "flex: 1; display: flex; overflow: hidden;";

        // ── LEFT TOOLBAR (Tools) ───────────────────────────────────────────────
        const leftToolbar = document.createElement("div");
        leftToolbar.style.cssText = `
            flex: 0 0 50px; background: #0e0e16;
            border-right: 1px solid rgba(60, 70, 100, 0.2);
            display: flex; flex-direction: column; align-items: center; padding-top: 10px; gap: 8px;
        `;

        this.toolBtns = {};
        const addTool = (id, icon, tooltip) => {
            const btn = document.createElement("button");
            btn.innerHTML = icon; btn.title = tooltip;
            btn.style.cssText = `
                width: 36px; height: 36px; border-radius: 6px; border: 1px solid transparent;
                background: transparent; color: #a0a0b0; cursor: pointer; font-size: 16px;
                display: flex; align-items: center; justify-content: center; transition: 0.2s;
            `;
            btn.onclick = () => this.setTool(id);
            this.toolBtns[id] = btn;
            leftToolbar.appendChild(btn);
        };

        addTool('brush', '○', 'Soft Brush (B)');
        addTool('eraser', '◌', 'Eraser (E)');
        addTool('polygon', '⬡', 'Polygon/Spline (P)');
        addTool('magic_wand', '✦', 'Magic Wand (W)');

        // ── CANVAS AREA ────────────────────────────────────────────────────────
        this.canvasContainer = document.createElement("div");
        this.canvasContainer.style.cssText = `
            flex: 1; background: #000; position: relative; overflow: hidden; cursor: crosshair;
        `;

        this.displayCanvas = document.createElement("canvas");
        this.displayCanvas.style.cssText = "position: absolute; top: 0; left: 0; transform-origin: top left;";

        // Raster Mask Context
        this.maskCanvas = document.createElement("canvas");
        this.maskCtx = this.maskCanvas.getContext('2d', { willReadFrequently: true });
        this.displayCtx = this.displayCanvas.getContext('2d');

        this.canvasContainer.appendChild(this.displayCanvas);

        // ── RIGHT PROPERTIES PANEL ─────────────────────────────────────────────
        const rightPanel = document.createElement("div");
        rightPanel.style.cssText = `
            flex: 0 0 280px; background: #12121a;
            border-left: 1px solid rgba(60, 70, 100, 0.2);
            padding: 16px; color: #ddd; display: flex; flex-direction: column; gap: 16px;
            overflow-y: auto; font-size: 13px;
        `;

        // Tool Settings Group
        const createGroup = (title) => {
            const g = document.createElement("div");
            g.style.cssText = "display: flex; flex-direction: column; gap: 12px; background: rgba(255,255,255,0.02); padding: 12px; border-radius: 6px; border: 1px solid rgba(255,255,255,0.05);";
            const h = document.createElement("div");
            h.innerText = title; h.style.cssText = "font-size: 11px; color: #00a8ff; text-transform: uppercase; letter-spacing: 1px; border-bottom: 1px solid rgba(0,168,255,0.2); padding-bottom: 4px;";
            g.appendChild(h);
            return g;
        };

        const viewGroup = createGroup("Visualization");
        const viewSelect = document.createElement("select");
        viewSelect.style.cssText = "background: #1a1a24; color: #fff; border: 1px solid #3c4664; padding: 4px; border-radius: 4px;";
        ['overlay', 'matte', 'false_color'].forEach(v => {
            const opt = document.createElement("option"); opt.value = v; opt.innerText = v.replace('_', ' ').toUpperCase();
            viewSelect.appendChild(opt);
        });
        viewSelect.onchange = (e) => { this.viewMode = e.target.value; this.drawDisplay(); };
        viewGroup.appendChild(viewSelect);

        this.brushGroup = createGroup("Raster Brush Settings");
        this.brushGroup.appendChild(this.createSlider("Size", 1, 500, 1, this.brushSize, (v) => this.brushSize = v));
        this.brushGroup.appendChild(this.createSlider("Opacity", 0.01, 1.0, 0.01, this.brushOpacity, (v) => this.brushOpacity = v));
        this.brushGroup.appendChild(this.createSlider("Softness", 0.0, 1.0, 0.01, this.brushHardness, (v) => this.brushHardness = v));

        this.vectorGroup = createGroup("Vector Shape Settings");
        this.vectorGroup.style.display = "none";
        const closePolyBtn = this.createBtn("Close / Fill Shape", () => this.closePolygon());
        const delPolyBtn = this.createBtn("Delete Shape", () => this.deletePolygon(), true);
        this.vectorFeather = this.createSlider("Feather", 0, 100, 1, 0, (v) => {
            if (this.currentPoly) { this.currentPoly.feather = v; this.drawDisplay(); }
        });
        this.vectorGroup.appendChild(closePolyBtn);
        this.vectorGroup.appendChild(this.vectorFeather);
        this.vectorGroup.appendChild(delPolyBtn);

        const qualifierGroup = createGroup("HSL Qualifier Keyer");
        qualifierGroup.style.display = "none";

        this.qToggle = document.createElement("button");
        this.qToggle.innerText = "Enable Qualifier";
        this.qToggle.style.cssText = "padding:6px; background: rgba(255,255,255,0.05); color:#aaa; border:1px solid rgba(255,255,255,0.1); border-radius:4px; cursor:pointer;";
        this.qToggle.onclick = () => {
            this.qualifier.enabled = !this.qualifier.enabled;
            this._updateQualifierToggleUI();
            this.drawDisplay();
        };
        qualifierGroup.appendChild(this.qToggle);

        const updateQ = (key, val) => { this.qualifier[key] = val; if (this.qualifier.enabled) this.drawDisplay(); };

        qualifierGroup.appendChild(this.createSlider("Hue", 0, 1, 0.01, this.qualifier.h, v => updateQ('h', v)));
        qualifierGroup.appendChild(this.createSlider("Hue Softness", 0, 1, 0.01, this.qualifier.hS, v => updateQ('hS', v)));
        qualifierGroup.appendChild(this.createSlider("Sat", 0, 1, 0.01, this.qualifier.s, v => updateQ('s', v)));
        qualifierGroup.appendChild(this.createSlider("Sat Softness", 0, 1, 0.01, this.qualifier.sS, v => updateQ('sS', v)));
        qualifierGroup.appendChild(this.createSlider("Lum", 0, 1, 0.01, this.qualifier.l, v => updateQ('l', v)));
        qualifierGroup.appendChild(this.createSlider("Lum Softness", 0, 1, 0.01, this.qualifier.lS, v => updateQ('lS', v)));

        const qBakeBtn = this.createBtn("Bake Qualifier to Mask", () => this.bakeQualifier());
        qualifierGroup.appendChild(qBakeBtn);

        this.qualifierGroup = qualifierGroup;

        // Magic Wand settings
        this.wandGroup = createGroup("Magic Wand (Flood Fill)");
        this.wandGroup.style.display = "none";
        this.wandTolerance = 0.1;
        this.wandGroup.appendChild(this.createSlider("Tolerance", 0, 1, 0.01, this.wandTolerance, (v) => this.wandTolerance = v));

        const actionsGroup = createGroup("Global Actions");
        actionsGroup.style.marginTop = "auto";
        const clearBtn = this.createBtn("Clear All Masks", () => this.clearMask(), true);
        actionsGroup.appendChild(clearBtn);

        rightPanel.appendChild(viewGroup);
        rightPanel.appendChild(this.brushGroup);
        rightPanel.appendChild(this.vectorGroup);
        rightPanel.appendChild(this.qualifierGroup);
        rightPanel.appendChild(this.wandGroup);
        rightPanel.appendChild(actionsGroup);

        // Assembly
        contentArea.appendChild(leftToolbar);
        contentArea.appendChild(this.canvasContainer);
        contentArea.appendChild(rightPanel);

        this.container.appendChild(topbar);
        this.container.appendChild(contentArea);
        this.overlay.appendChild(this.container);
        document.body.appendChild(this.overlay);

        this.setTool('brush');
    }

    _updateQualifierToggleUI() {
        const on = this.qualifier.enabled;
        this.qToggle.style.background = on ? "rgba(0,168,255,0.2)" : "rgba(255,255,255,0.05)";
        this.qToggle.style.color = on ? "#00a8ff" : "#aaa";
        this.qToggle.style.borderColor = on ? "#00a8ff" : "rgba(255,255,255,0.1)";
    }

    createBtn(text, onClick, danger = false) {
        const btn = document.createElement("button");
        btn.innerText = text;
        btn.style.cssText = `
            padding: 6px 12px; border-radius: 4px; cursor: pointer; font-size: 13px;
            background: ${danger ? 'rgba(255,50,50,0.1)' : 'rgba(255,255,255,0.05)'};
            color: ${danger ? '#f55' : '#ccc'};
            border: 1px solid ${danger ? 'rgba(255,50,50,0.3)' : 'rgba(255,255,255,0.1)'};
            transition: 0.2s;
        `;
        btn.onclick = onClick;
        return btn;
    }

    createSlider(label, min, max, step, val, onChange) {
        // Use unique ID to avoid DOM collisions
        const uid = `rad_sl_${this._sliderId++}`;

        const wrap = document.createElement("div");
        wrap.style.display = "flex"; wrap.style.flexDirection = "column"; wrap.style.gap = "4px";

        const head = document.createElement("div");
        head.style.display = "flex"; head.style.justifyContent = "space-between"; head.style.fontSize = "12px";

        const labelSpan = document.createElement("span");
        labelSpan.style.color = "#aaa";
        labelSpan.textContent = label;

        const valSpan = document.createElement("span");
        valSpan.id = uid;
        valSpan.textContent = val;

        head.appendChild(labelSpan);
        head.appendChild(valSpan);

        const slider = document.createElement("input");
        slider.type = "range"; slider.min = min; slider.max = max; slider.step = step; slider.value = val;
        slider.style.width = "100%"; slider.style.accentColor = "#00a8ff";

        slider.oninput = (e) => {
            const v = parseFloat(e.target.value);
            valSpan.textContent = v;
            onChange(v);
        };

        wrap.appendChild(head);
        wrap.appendChild(slider);
        return wrap;
    }

    setTool(tool) {
        this.tool = tool;
        Object.keys(this.toolBtns).forEach(k => {
            const b = this.toolBtns[k];
            if (k === tool) {
                b.style.background = "rgba(0,168,255,0.2)";
                b.style.borderColor = "#00a8ff"; b.style.color = "#00a8ff";
            } else {
                b.style.background = "transparent";
                b.style.borderColor = "transparent"; b.style.color = "#a0a0b0";
            }
        });

        if (tool === 'polygon') {
            this.brushGroup.style.display = "none";
            this.vectorGroup.style.display = "flex";
            this.wandGroup.style.display = "none";
            this.qualifierGroup.style.display = "none";
            if (!this.currentPoly) this.startPolygon();
        } else if (tool === 'magic_wand') {
            this.brushGroup.style.display = "none";
            this.vectorGroup.style.display = "none";
            this.wandGroup.style.display = "flex";
            this.qualifierGroup.style.display = "flex"; // Contextual pairing
            this.currentPoly = null;
        } else {
            this.brushGroup.style.display = "flex";
            this.vectorGroup.style.display = "none";
            this.wandGroup.style.display = "none";
            this.qualifierGroup.style.display = "none";
            this.currentPoly = null;
        }

        this.drawDisplay();
    }

    // ═══════════════════════════════════════════════════════════════════════════
    //                           EVENT LISTENERS
    // ═══════════════════════════════════════════════════════════════════════════

    /**
     * Register a global listener with automatic cleanup tracking.
     */
    _addGlobalListener(target, event, handler, options) {
        target.addEventListener(event, handler, options);
        this._boundListeners.push({ target, event, handler, options });
    }

    /**
     * Remove all tracked global listeners. Called on hide().
     */
    _removeGlobalListeners() {
        for (const { target, event, handler, options } of this._boundListeners) {
            target.removeEventListener(event, handler, options);
        }
        this._boundListeners = [];
    }

    setupEventListeners() {
        this._resizeObserver = new ResizeObserver(() => {
            if (this.overlay.style.display === "flex") this.drawDisplay();
        });
        this._resizeObserver.observe(this.canvasContainer);

        // Use pointer events for pressure sensitivity
        this.canvasContainer.addEventListener('pointerdown', (e) => {
            e.preventDefault();
            if (e.button === 1 || (e.button === 0 && e.shiftKey) || e.button === 2) {
                this.isPanning = true;
                this.canvasContainer.style.cursor = 'grabbing';
                return;
            }

            const pos = this.getLogicalPos(e.clientX, e.clientY);

            if (this.tool === 'magic_wand') {
                this.saveUndoState();
                this.executeMagicWand(pos);
                return;
            }

            if (this.tool === 'polygon') {
                // 1. Check if clicking near start point of current polygon to close
                if (this.currentPoly && this.currentPoly.points.length > 2) {
                    const first = this.currentPoly.points[0];
                    const dist = Math.hypot(first.x - pos.x, first.y - pos.y);
                    if (dist * this.zoom < POINT_HIT_RADIUS_PX) {
                        this.closePolygon();
                        return;
                    }
                }

                // 2. Check if clicking near any existing point in any polygon to drag
                for (const poly of this.polygons) {
                    for (let i = 0; i < poly.points.length; i++) {
                        const p = poly.points[i];
                        const dist = Math.hypot(p.x - pos.x, p.y - pos.y);
                        if (dist * this.zoom < POINT_HIT_RADIUS_PX) {
                            this.isDraggingVertex = true;
                            this.draggingPoly = poly;
                            this.draggingPointIndex = i;
                            this.canvasContainer.style.cursor = 'grabbing';
                            return;
                        }
                    }
                }

                // 3. Otherwise, append point
                if (!this.currentPoly) this.startPolygon();
                this.currentPoly.points.push({ x: pos.x, y: pos.y });
                this.saveUndoState();
                this.drawDisplay();
                return;
            }

            if (e.button === 0) {
                this.isDrawing = true;
                this.lastPaintPos = pos; // Set initial paint position
                this.saveUndoState();
                this.paint(e);
            }
        });

        this.canvasContainer.addEventListener('pointermove', (e) => {
            const pos = this.getLogicalPos(e.clientX, e.clientY);
            if (this.isPanning) {
                this.panX += e.movementX;
                this.panY += e.movementY;
                this.updateCanvasTransform();
            } else if (this.isDraggingVertex) {
                this.draggingPoly.points[this.draggingPointIndex] = { x: pos.x, y: pos.y };
                this._scheduleDisplayUpdate();
            } else if (this.isDrawing) {
                this.paint(e);
            } else {
                // Interactive hover for polygon
                if (this.tool === 'polygon' && this.currentPoly && !this.currentPoly.closed) {
                    this.drawDisplay(pos);
                }
            }
        });

        // Context menu block (for pan)
        this.canvasContainer.addEventListener('contextmenu', e => e.preventDefault());

        this.canvasContainer.addEventListener('wheel', (e) => {
            e.preventDefault();
            const rect = this.canvasContainer.getBoundingClientRect();
            const mouseX = e.clientX - rect.left;
            const mouseY = e.clientY - rect.top;

            const oldZoom = this.zoom;
            if (e.deltaY < 0) this.zoom *= ZOOM_FACTOR;
            else this.zoom /= ZOOM_FACTOR;
            this.zoom = Math.max(ZOOM_MIN, Math.min(this.zoom, ZOOM_MAX));

            this.panX = mouseX - (mouseX - this.panX) * (this.zoom / oldZoom);
            this.panY = mouseY - (mouseY - this.panY) * (this.zoom / oldZoom);
            this.updateCanvasTransform();
        });

        // NOTE: pointerup and keydown are registered as global listeners
        // only when the editor is shown, and removed on hide. See show()/hide().
    }

    /**
     * Attach global listeners that should only be active while the editor is open.
     */
    _attachActiveListeners() {
        this._addGlobalListener(window, 'pointerup', () => {
            this.isPanning = false;
            if (this.isDraggingVertex) {
                this.isDraggingVertex = false;
                this.draggingPoly = null;
                this.draggingPointIndex = -1;
                this.saveUndoState();
                this.drawDisplay();
            }
            if (this.isDrawing) {
                this.isDrawing = false;
                this.lastPaintPos = null; // Reset continuous paint
                this.drawDisplay(); // Final commit draw
            }
            this.canvasContainer.style.cursor = 'crosshair';
        });

        this._addGlobalListener(document, 'keydown', (e) => {
            if (e.key === "z" && (e.ctrlKey || e.metaKey) && !e.shiftKey) {
                e.preventDefault();
                this.undo();
            } else if (e.key === "z" && (e.ctrlKey || e.metaKey) && e.shiftKey) {
                e.preventDefault();
                this.redo();
            } else if (e.key === "y" && (e.ctrlKey || e.metaKey)) {
                e.preventDefault();
                this.redo();
            } else if (e.key === "Escape") {
                this.hide();
            } else if (e.key.toLowerCase() === "b") {
                this.setTool('brush');
            } else if (e.key.toLowerCase() === "e") {
                this.setTool('eraser');
            } else if (e.key.toLowerCase() === "p") {
                this.setTool('polygon');
            } else if (e.key.toLowerCase() === "w") {
                this.setTool('magic_wand');
            } else if (e.key === "Enter" && this.tool === 'polygon') {
                this.closePolygon();
            }
        });
    }

    // ═══════════════════════════════════════════════════════════════════════════
    //                         COORDINATE HELPERS
    // ═══════════════════════════════════════════════════════════════════════════

    getLogicalPos(clientX, clientY) {
        const rect = this.canvasContainer.getBoundingClientRect();
        return {
            x: (clientX - rect.left - this.panX) / this.zoom,
            y: (clientY - rect.top - this.panY) / this.zoom
        };
    }

    updateCanvasTransform() {
        this.displayCanvas.style.transform = `translate(${this.panX}px, ${this.panY}px) scale(${this.zoom})`;
    }

    // ═══════════════════════════════════════════════════════════════════════════
    //                          RASTER PAINTING
    // ═══════════════════════════════════════════════════════════════════════════

    paint(e) {
        const pos = this.getLogicalPos(e.clientX, e.clientY);
        const pressure = e.pressure !== undefined ? Math.max(0.1, e.pressure) : 1.0;

        const ctx = this.maskCtx;
        ctx.save();

        if (this.tool === 'eraser') {
            ctx.globalCompositeOperation = 'destination-out';
        } else {
            ctx.globalCompositeOperation = 'source-over';
        }

        const radius = (this.brushSize / 2) * pressure;
        const opac = this.brushOpacity * pressure;

        const drawStamp = (cx, cy) => {
            const grad = ctx.createRadialGradient(cx, cy, radius * this.brushHardness, cx, cy, radius);
            grad.addColorStop(0, `rgba(255,255,255, ${opac})`);
            grad.addColorStop(1, `rgba(255,255,255, 0)`);
            ctx.fillStyle = grad;
            ctx.beginPath();
            ctx.arc(cx, cy, radius, 0, Math.PI * 2);
            ctx.fill();
        };

        if (this.lastPaintPos) {
            const dist = Math.hypot(pos.x - this.lastPaintPos.x, pos.y - this.lastPaintPos.y);
            const step = Math.max(1, radius * 0.1); // Stamp every 10% of the radius
            if (dist > step) {
                for (let d = 0; d < dist; d += step) {
                    const t = d / dist;
                    const cx = this.lastPaintPos.x + (pos.x - this.lastPaintPos.x) * t;
                    const cy = this.lastPaintPos.y + (pos.y - this.lastPaintPos.y) * t;
                    drawStamp(cx, cy);
                }
            }
        }
        drawStamp(pos.x, pos.y);
        ctx.restore();

        this.lastPaintPos = pos;

        // RAF-gated display update — avoids compositing on every single pointermove
        this._scheduleDisplayUpdate();
    }

    /**
     * Schedule a drawDisplay via requestAnimationFrame.
     * Coalesces multiple calls per frame into a single composite.
     */
    _scheduleDisplayUpdate(hoverPos = null) {
        if (this._rafPending) return;
        this._rafPending = true;
        requestAnimationFrame(() => {
            this._rafPending = false;
            this.drawDisplay(hoverPos);
        });
    }

    // ═══════════════════════════════════════════════════════════════════════════
    //                        VECTOR ROTOSCOPING
    // ═══════════════════════════════════════════════════════════════════════════

    startPolygon() {
        this.currentPoly = { points: [], closed: false, feather: 0 };
        this.polygons.push(this.currentPoly);
    }

    closePolygon() {
        if (this.currentPoly && this.currentPoly.points.length > 2) {
            this.saveUndoState();
            this.currentPoly.closed = true;
            // No longer bake directly to raster canvas to keep vector roto non-destructive
            this.currentPoly = null;
            this.drawDisplay();
        }
    }

    deletePolygon() {
        if (this.polygons.length > 0) {
            this.saveUndoState();
            this.polygons.pop();
            this.currentPoly = null;
            this.drawDisplay();
        }
    }

    bakePolygon(poly, ctx = this.maskCtx) {
        ctx.save();
        ctx.fillStyle = "rgba(255,255,255,1)";
        ctx.beginPath();
        poly.points.forEach((p, i) => {
            if (i === 0) ctx.moveTo(p.x, p.y);
            else ctx.lineTo(p.x, p.y);
        });
        ctx.closePath();

        // Basic feather simulation via shadowBlur
        if (poly.feather > 0) {
            ctx.shadowColor = "white";
            ctx.shadowBlur = poly.feather;
        }
        ctx.fill();
        ctx.restore();
    }

    getCombinedMaskCanvas() {
        const tempCanvas = document.createElement("canvas");
        tempCanvas.width = this.maskCanvas.width;
        tempCanvas.height = this.maskCanvas.height;
        const tempCtx = tempCanvas.getContext('2d');

        // 1. Draw the raster paint mask
        tempCtx.drawImage(this.maskCanvas, 0, 0);

        // 2. Draw all closed dynamic vector polygons
        this.polygons.forEach(poly => {
            if (poly.closed && poly.points.length > 2) {
                this.bakePolygon(poly, tempCtx);
            }
        });
        return tempCanvas;
    }

    rebuildMaskFromState() {
        // Mixed raster/vector workflow: true rebuild requires undo snapshot.
        this.undo();
    }

    clearMask() {
        this.saveUndoState();
        this.maskCtx.clearRect(0, 0, this.maskCanvas.width, this.maskCanvas.height);
        this.polygons = [];
        this.currentPoly = null;
        this.drawDisplay();
    }

    // ═══════════════════════════════════════════════════════════════════════════
    //                       UNDO / REDO (FIXED)
    // ═══════════════════════════════════════════════════════════════════════════

    saveUndoState() {
        // Discard any redo states ahead of current position
        if (this.historyIndex < this.history.length - 1) {
            this.history = this.history.slice(0, this.historyIndex + 1);
        }

        // Save raster payload + vector metadata clone
        this.history.push({
            imgData: this.maskCtx.getImageData(0, 0, this.maskCanvas.width, this.maskCanvas.height),
            polys: JSON.parse(JSON.stringify(this.polygons))
        });

        // Evict oldest if over capacity — keep historyIndex aligned
        if (this.history.length > MAX_UNDO_STATES) {
            this.history.shift();
            // historyIndex stays the same: it now correctly points to the
            // same logical entry (which shifted down by one, but we didn't
            // increment it yet, and the push added one, so net = same)
        }

        this.historyIndex = this.history.length - 1;
    }

    undo() {
        // historyIndex points to the most recently saved state (current).
        // To undo, we need to go back one step.
        if (this.historyIndex > 0) {
            this.historyIndex--;
            this._restoreState(this.history[this.historyIndex]);
        } else if (this.historyIndex === 0) {
            // Already at the earliest snapshot — nothing to undo to.
            // Optionally: restore initial blank state.
        }
    }

    redo() {
        if (this.historyIndex < this.history.length - 1) {
            this.historyIndex++;
            this._restoreState(this.history[this.historyIndex]);
        }
    }

    _restoreState(state) {
        this.maskCtx.putImageData(state.imgData, 0, 0);
        this.polygons = JSON.parse(JSON.stringify(state.polys));
        this.currentPoly = this.polygons.find(p => !p.closed) || null;
        this.drawDisplay();
    }

    // ═══════════════════════════════════════════════════════════════════════════
    //                     HSL QUALIFIER & MAGIC WAND
    // ═══════════════════════════════════════════════════════════════════════════

    rgbToHsl(r, g, b) {
        r /= 255; g /= 255; b /= 255;
        const max = Math.max(r, g, b), min = Math.min(r, g, b);
        let h, s, l = (max + min) / 2;
        if (max === min) {
            h = s = 0;
        } else {
            const d = max - min;
            s = l > 0.5 ? d / (2 - max - min) : d / (max + min);
            switch (max) {
                case r: h = (g - b) / d + (g < b ? 6 : 0); break;
                case g: h = (b - r) / d + 2; break;
                case b: h = (r - g) / d + 4; break;
            }
            h /= 6;
        }
        return [h, s, l];
    }

    smoothstep(edge0, edge1, x) {
        if (edge1 === edge0) return x < edge0 ? 0 : 1; // Guard against division by zero
        const t = Math.max(0, Math.min(1, (x - edge0) / (edge1 - edge0)));
        return t * t * (3 - 2 * t);
    }

    evalQualifierAlpha(r, g, b) {
        if (!this.qualifier.enabled) return 0;

        const [h, s, l] = this.rgbToHsl(r, g, b);
        const q = this.qualifier;

        // Hue distance with wrapping
        let hDist = Math.abs(h - q.h);
        if (hDist > 0.5) hDist = 1.0 - hDist;

        const hAlpha = 1.0 - this.smoothstep(q.hW, q.hW + q.hS, hDist);
        const sAlpha = 1.0 - this.smoothstep(q.sW, q.sW + q.sS, Math.abs(s - q.s));
        const lAlpha = 1.0 - this.smoothstep(q.lW, q.lW + q.lS, Math.abs(l - q.l));

        return hAlpha * sAlpha * lAlpha;
    }

    /**
     * Cache the source image pixels so we don't re-draw to a temp canvas
     * every time magic wand or bake qualifier runs.
     */
    _getSourceImageData() {
        if (this._cachedSourceData &&
            this._cachedSourceData.width === this.maskCanvas.width &&
            this._cachedSourceData.height === this.maskCanvas.height) {
            return this._cachedSourceData;
        }

        const width = this.maskCanvas.width;
        const height = this.maskCanvas.height;
        const tempCanvas = document.createElement("canvas");
        tempCanvas.width = width; tempCanvas.height = height;
        const tCtx = tempCanvas.getContext('2d');
        tCtx.drawImage(this.imageEl, 0, 0);
        this._cachedSourceData = tCtx.getImageData(0, 0, width, height);
        return this._cachedSourceData;
    }

    executeMagicWand(pos) {
        if (!this.imageEl) return;

        const width = this.maskCanvas.width;
        const height = this.maskCanvas.height;
        const startX = Math.floor(pos.x);
        const startY = Math.floor(pos.y);

        if (startX < 0 || startX >= width || startY < 0 || startY >= height) return;

        const srcData = this._getSourceImageData().data;

        // Destination mask
        const maskDataObj = this.maskCtx.getImageData(0, 0, width, height);
        const mData = maskDataObj.data;

        // Pick color at startX, startY
        const startIdx = (startY * width + startX) * 4;
        const startR = srcData[startIdx];
        const startG = srcData[startIdx + 1];
        const startB = srcData[startIdx + 2];

        const tol = this.wandTolerance;
        const maxDistSq = (441.67 * tol) * (441.67 * tol); // Pre-square for faster compare

        // Scanline flood fill — much faster than naive 4-neighbor stack
        const visited = new Uint8Array(width * height);

        const colorMatch = (idx4) => {
            const dr = srcData[idx4] - startR;
            const dg = srcData[idx4 + 1] - startG;
            const db = srcData[idx4 + 2] - startB;
            return (dr * dr + dg * dg + db * db) <= maxDistSq;
        };

        const fillPixel = (idx4) => {
            mData[idx4] = 255;
            mData[idx4 + 1] = 255;
            mData[idx4 + 2] = 255;
            mData[idx4 + 3] = 255;
        };

        const stack = [[startX, startY]];

        while (stack.length > 0) {
            let [x, y] = stack.pop();
            const lineIdx = y * width;

            // Walk left
            let lx = x;
            while (lx >= 0 && !visited[lineIdx + lx] && colorMatch((lineIdx + lx) * 4)) {
                lx--;
            }
            lx++; // Back to last valid

            // Walk right
            let rx = x;
            while (rx < width && !visited[lineIdx + rx] && colorMatch((lineIdx + rx) * 4)) {
                rx++;
            }
            rx--; // Back to last valid

            // Fill the span and check neighbors above/below
            let aboveAdded = false;
            let belowAdded = false;

            for (let cx = lx; cx <= rx; cx++) {
                const ci = lineIdx + cx;
                if (visited[ci]) continue;
                visited[ci] = 1;
                fillPixel(ci * 4);

                // Check above
                if (y > 0) {
                    const aboveIdx = ci - width;
                    if (!visited[aboveIdx] && colorMatch(aboveIdx * 4)) {
                        if (!aboveAdded) {
                            stack.push([cx, y - 1]);
                            aboveAdded = true;
                        }
                    } else {
                        aboveAdded = false;
                    }
                }

                // Check below
                if (y < height - 1) {
                    const belowIdx = ci + width;
                    if (!visited[belowIdx] && colorMatch(belowIdx * 4)) {
                        if (!belowAdded) {
                            stack.push([cx, y + 1]);
                            belowAdded = true;
                        }
                    } else {
                        belowAdded = false;
                    }
                }
            }
        }

        this.maskCtx.putImageData(maskDataObj, 0, 0);
        this.drawDisplay();
    }

    bakeQualifier() {
        if (!this.qualifier.enabled || !this.imageEl) return;
        this.saveUndoState();

        const width = this.maskCanvas.width;
        const height = this.maskCanvas.height;

        const srcData = this._getSourceImageData().data;

        const maskDataObj = this.maskCtx.getImageData(0, 0, width, height);
        const mData = maskDataObj.data;

        for (let i = 0; i < srcData.length; i += 4) {
            const alpha = this.evalQualifierAlpha(srcData[i], srcData[i + 1], srcData[i + 2]);
            if (alpha > 0) {
                const val = Math.min(255, mData[i + 3] + alpha * 255);
                mData[i] = 255; mData[i + 1] = 255; mData[i + 2] = 255;
                mData[i + 3] = val;
            }
        }

        this.maskCtx.putImageData(maskDataObj, 0, 0);
        this.qualifier.enabled = false;
        this._updateQualifierToggleUI();
        this.drawDisplay();
    }

    // ═══════════════════════════════════════════════════════════════════════════
    //                      COMPOSITING RENDER
    // ═══════════════════════════════════════════════════════════════════════════

    drawDisplay(hoverPos = null) {
        if (!this.imageEl || !this.imageEl.complete) return;

        this.displayCanvas.width = this.imageEl.width;
        this.displayCanvas.height = this.imageEl.height;
        const ctx = this.displayCtx;

        // 1. Base Image
        if (this.viewMode === 'matte') {
            ctx.fillStyle = 'black';
            ctx.fillRect(0, 0, this.displayCanvas.width, this.displayCanvas.height);
        } else {
            ctx.drawImage(this.imageEl, 0, 0);
        }

        // Live Qualifier overlay (dynamic procedural matte)
        if (this.qualifier.enabled) {
            const imgData = ctx.getImageData(0, 0, this.displayCanvas.width, this.displayCanvas.height);
            const data = imgData.data;
            for (let i = 0; i < data.length; i += 4) {
                const alpha = this.evalQualifierAlpha(data[i], data[i + 1], data[i + 2]);
                if (alpha > 0) {
                    if (this.viewMode === 'matte') {
                        data[i] = Math.max(data[i], alpha * 255);
                        data[i + 1] = Math.max(data[i + 1], alpha * 255);
                        data[i + 2] = Math.max(data[i + 2], alpha * 255);
                    } else if (this.viewMode === 'overlay') {
                        data[i] = Math.min(255, data[i] + alpha * 255);
                        data[i + 1] *= (1 - alpha * 0.5);
                        data[i + 2] *= (1 - alpha * 0.5);
                    }
                }
            }
            ctx.putImageData(imgData, 0, 0);
        }

        // 2. Overlay Base Mask (using Combined Mask canvas!)
        const combinedCanvas = this.getCombinedMaskCanvas();
        const combinedCtx = combinedCanvas.getContext('2d');

        ctx.save();
        if (this.viewMode === 'overlay') {
            // Ruby Red overlay common in Nuke
            const tempCanvas = document.createElement("canvas");
            tempCanvas.width = combinedCanvas.width; tempCanvas.height = combinedCanvas.height;
            const tempCtx = tempCanvas.getContext('2d');
            tempCtx.drawImage(combinedCanvas, 0, 0);
            tempCtx.globalCompositeOperation = 'source-in';
            tempCtx.fillStyle = 'rgba(255, 30, 30, 0.45)';
            tempCtx.fillRect(0, 0, tempCanvas.width, tempCanvas.height);
            ctx.drawImage(tempCanvas, 0, 0);

        } else if (this.viewMode === 'matte') {
            ctx.drawImage(combinedCanvas, 0, 0);

        } else if (this.viewMode === 'false_color') {
            // Per-pixel alpha-to-color-ramp (Flame/Nuke-style false color)
            // Maps mask alpha: 0=black, low=blue, mid=green, high=red, 1.0=white
            const maskData = combinedCtx.getImageData(0, 0, combinedCanvas.width, combinedCanvas.height);
            const md = maskData.data;
            const outData = ctx.getImageData(0, 0, this.displayCanvas.width, this.displayCanvas.height);
            const od = outData.data;

            for (let i = 0; i < md.length; i += 4) {
                const a = md[i + 3] / 255; // Normalized alpha 0..1
                if (a <= 0) continue;

                let r, g, b;
                if (a < 0.25) {
                    // Black → Blue
                    const t = a / 0.25;
                    r = 0; g = 0; b = t * 255;
                } else if (a < 0.5) {
                    // Blue → Green
                    const t = (a - 0.25) / 0.25;
                    r = 0; g = t * 255; b = (1 - t) * 255;
                } else if (a < 0.75) {
                    // Green → Red
                    const t = (a - 0.5) / 0.25;
                    r = t * 255; g = (1 - t) * 255; b = 0;
                } else {
                    // Red → White
                    const t = (a - 0.75) / 0.25;
                    r = 255; g = t * 255; b = t * 255;
                }

                od[i] = r;
                od[i + 1] = g;
                od[i + 2] = b;
                od[i + 3] = 255;
            }
            ctx.putImageData(outData, 0, 0);
        }
        ctx.restore();

        // 3. Draw Vector UI Elements (Lines and Points)
        if (this.tool === 'polygon') {
            ctx.save();
            this.polygons.forEach(poly => {
                if (poly.points.length === 0) return;

                ctx.beginPath();
                ctx.strokeStyle = poly.closed ? '#00a8ff' : '#00ffaa';
                ctx.lineWidth = 1.5 / this.zoom;

                poly.points.forEach((p, i) => {
                    if (i === 0) ctx.moveTo(p.x, p.y);
                    else ctx.lineTo(p.x, p.y);
                });

                if (poly.closed) ctx.closePath();
                else if (hoverPos && poly === this.currentPoly) {
                    ctx.lineTo(hoverPos.x, hoverPos.y);
                }
                ctx.stroke();

                // Draw points
                ctx.fillStyle = '#fff';
                poly.points.forEach((p, i) => {
                    ctx.beginPath();
                    ctx.arc(p.x, p.y, 3 / this.zoom, 0, Math.PI * 2);

                    if (!poly.closed && i === 0) {
                        ctx.fillStyle = '#ffaa00';
                        ctx.fill();
                        ctx.fillStyle = '#fff';
                    } else {
                        ctx.fill();
                    }
                });
            });
            ctx.restore();
        }
    }

    // ═══════════════════════════════════════════════════════════════════════════
    //                       LIFECYCLE: LOAD / SHOW / HIDE / SAVE
    // ═══════════════════════════════════════════════════════════════════════════

    showToast(message, tone = "info") {
        const toast = document.createElement("div");
        const color = tone === "error" ? "#ff6b6b" : tone === "success" ? "#4cd964" : "#00a8ff";
        toast.textContent = message;
        toast.style.cssText = `
            position: fixed; left: 50%; bottom: 28px; transform: translateX(-50%);
            z-index: 10020; max-width: 460px; padding: 10px 14px;
            color: #f5f5f7; background: rgba(18,18,24,0.96);
            border: 1px solid ${color}66; border-radius: 8px;
            box-shadow: 0 12px 36px rgba(0,0,0,0.55);
            font: 12px/1.4 -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            pointer-events: none;
        `;
        document.body.appendChild(toast);
        setTimeout(() => toast.remove(), 3200);
    }

    async load(node) {
        this.node = node;
        const imgWidget = node.widgets.find(w => w.name === "image");
        if (!imgWidget || !imgWidget.value) {
            this.showToast("Please upload or select an image first.", "info");
            return;
        }

        const filename = imgWidget.value;
        const pureName = filename.substring(0, filename.lastIndexOf('.')) || filename;
        const compMaskName = `${pureName}_radmask.png`;

        const imgUrl = api.apiURL(`/view?filename=${encodeURIComponent(filename)}&type=input`);
        const maskUrl = api.apiURL(`/view?filename=${encodeURIComponent(compMaskName)}&type=input`);

        // Invalidate source data cache for new image
        this._cachedSourceData = null;

        this.imageEl = new Image();
        this.imageEl.crossOrigin = "Anonymous";
        this.imageEl.onload = () => {
            this.maskCanvas.width = this.imageEl.width;
            this.maskCanvas.height = this.imageEl.height;

            const maskImg = new Image();
            maskImg.crossOrigin = "Anonymous";
            maskImg.onload = () => {
                this.maskCtx.drawImage(maskImg, 0, 0);
                this.finishLoadSetup();
            };
            maskImg.onerror = () => this.finishLoadSetup();
            maskImg.src = maskUrl + "&timestamp=" + Date.now();
        };
        this.imageEl.onerror = () => {
            console.error("[Radiance] Failed to load source image:", filename);
            this.showToast("Failed to load source image.", "error");
        };
        this.imageEl.src = imgUrl;

        this.show();
    }

    show() {
        this.overlay.style.display = "flex";
        this.zoom = 1.0; this.panX = 0; this.panY = 0;
        this.history = []; this.historyIndex = -1;
        this.polygons = []; this.currentPoly = null;
        this._attachActiveListeners();
    }

    finishLoadSetup() {
        const rect = this.canvasContainer.getBoundingClientRect();
        const scaleX = rect.width / this.imageEl.width;
        const scaleY = rect.height / this.imageEl.height;
        this.zoom = Math.min(scaleX, scaleY) * 0.9;
        this.panX = (rect.width - this.imageEl.width * this.zoom) / 2;
        this.panY = (rect.height - this.imageEl.height * this.zoom) / 2;
        this.updateCanvasTransform();
        this.saveUndoState();
        this.drawDisplay();
    }

    hide() {
        this.overlay.style.display = "none";
        this._removeGlobalListeners();
    }

    saveMask() {
        // Close any open polygon before saving
        if (this.currentPoly && !this.currentPoly.closed && this.currentPoly.points.length > 2) {
            this.currentPoly.closed = true;
            // No longer bake directly to raster canvas to keep vector roto non-destructive
            this.currentPoly = null;
        }

        const imgWidget = this.node.widgets.find(w => w.name === "image");
        if (!imgWidget) return;

        const filename = imgWidget.value;
        const pureName = filename.substring(0, filename.lastIndexOf('.')) || filename;
        const compMaskName = `${pureName}_radmask.png`;
        const metaName = `${pureName}_radmask_meta.json`;

        // Save Combined Mask (Dynamic Vector Polygons + Raster Paint Canvas)
        const combinedCanvas = this.getCombinedMaskCanvas();
        combinedCanvas.toBlob(async (blob) => {
            if (!blob) {
                console.error("[Radiance] Failed to create mask blob.");
                return;
            }

            const formData = new FormData();
            formData.append("image", blob, compMaskName);
            formData.append("type", "input");
            formData.append("overwrite", "true");

            try {
                const resp = await fetch(api.apiURL("/upload/image"), { method: "POST", body: formData });
                if (!resp.ok) throw new Error(`Upload failed: HTTP ${resp.status}`);
                console.log("[Radiance] Raster mask saved.");

                // Save Vector/Node Metadata
                const metaBlob = new Blob(
                    [JSON.stringify({ polygons: this.polygons })],
                    { type: "application/json" }
                );
                const metaForm = new FormData();
                metaForm.append("image", metaBlob, metaName);
                metaForm.append("type", "input");
                metaForm.append("overwrite", "true");

                await fetch(api.apiURL("/upload/image"), { method: "POST", body: metaForm });

                if (app.graph) {
                    this.node.setDirtyCanvas(true, true);
                    app.graph.setDirtyCanvas(true, true);
                }
                this.hide();
            } catch (error) {
                console.error("[Radiance] Save error:", error);
                this.showToast("Failed to save mask. Check console for details.", "error");
            }
        }, "image/png");
    }
}

const RADIANCE_MASK_EDITOR = new RadianceMaskEditor();

app.registerExtension({
    name: "FXTD.Radiance.MaskEditorPro",
    nodeCreated(node) {
        if (node.comfyClass === "RadianceLoadImageMask") {
            const editBtn = node.addWidget("button", "◎ Radiance Mask", "edit", () => {
                RADIANCE_MASK_EDITOR.load(node);
            });
            editBtn.serialize = false;

            // Apply standard Radiance header colors so it looks premium
            node.color = "#232330";
            node.bgcolor = "#0f0f14";
        }
    }
});
