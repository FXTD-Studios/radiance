import { api } from "../../scripts/api.js";

(function() {
var _queue = window.__radianceInit = window.__radianceInit || [];
function install(RV) {
    window.RadianceViewer = RV;

    RV.prototype._openDeliveryDialog = function() {
        document.getElementById('radiance-delivery-toast')?.remove();
        
        const toast = document.createElement('div');
        toast.id = 'radiance-delivery-toast';
        toast.style.cssText = `
            position: fixed;
            bottom: 24px;
            right: 24px;
            background: rgba(13,18,27,0.95);
            border: 1px solid rgba(0,168,255,0.4);
            border-radius: 6px;
            padding: 12px 16px;
            color: #e8e8f0;
            font-size: 11px;
            font-family: var(--radiance-font-ui), sans-serif;
            box-shadow: 0 10px 30px rgba(0,0,0,0.5);
            z-index: 100000;
            display: flex;
            align-items: center;
            gap: 12px;
            backdrop-filter: blur(10px);
            animation: radToastIn 0.2s ease-out;
        `;
        
        if (!document.getElementById('radiance-toast-style')) {
            const style = document.createElement('style');
            style.id = 'radiance-toast-style';
            style.textContent = `
                @keyframes radToastIn {
                    from { transform: translateY(20px); opacity: 0; }
                    to { transform: translateY(0); opacity: 1; }
                }
            `;
            document.head.appendChild(style);
        }
        
        const infoIcon = document.createElement('span');
        infoIcon.textContent = 'ⓘ';
        infoIcon.style.cssText = 'color: #00a8ff; font-size: 14px; font-weight: bold;';
        
        const text = document.createElement('span');
        text.textContent = 'Use the Write node in your workflow for video and sequence export. Save PNG exports the current Viewer frame. Viewer adjustments are preview-only unless applied in the workflow.';
        
        const close = document.createElement('span');
        close.textContent = '✕';
        close.style.cssText = 'cursor: pointer; opacity: 0.5; font-size: 10px; margin-left: 8px;';
        close.onclick = () => toast.remove();
        
        toast.append(infoIcon, text, close);
        document.body.appendChild(toast);
        
        setTimeout(() => toast.remove(), 5000);
    };

    RV.prototype.showExportMenu = function(e) {
        if (this.exportMenu) this.exportMenu.remove();

        const menu = document.createElement('div');
        this.exportMenu = menu;
        menu.style.cssText = `
            position: absolute;
            background: rgba(15, 15, 20, 0.95);
            border: 1px solid rgba(100, 110, 150, 0.4);
            border-radius: 6px;
            padding: 4px;
            z-index: 1000;
            display: flex;
            flex-direction: column;
            gap: 2px;
            box-shadow: 0 8px 16px rgba(0,0,0,0.5);
            backdrop-filter: blur(10px);
        `;

        const rect = e.target.getBoundingClientRect();
        menu.style.left = rect.left + 'px';
        menu.style.top = (rect.bottom + 5) + 'px';

        const addOption = (label, icon, onClick, disabled = false) => {
            const opt = document.createElement('div');
            opt.innerHTML = `<span style="margin-right: 8px;">${icon}</span> ${label}`;
            opt.style.cssText = `
                padding: 6px 12px;
                color: ${disabled ? '#555' : '#ccc'};
                font-size: 11px;
                cursor: ${disabled ? 'default' : 'pointer'};
                border-radius: 4px;
                white-space: nowrap;
                transition: 0.2s;
            `;
            if (!disabled) {
                opt.onmouseenter = () => opt.style.background = 'rgba(255,255,255,0.08)';
                opt.onmouseleave = () => opt.style.background = 'transparent';
                opt.onclick = () => {
                    onClick();
                    menu.remove();
                };
            }
            menu.appendChild(opt);
        };

        addOption('Save PNG (Result)', '\u25CE', () => this.exportSnapshot('png'));
        addOption('Video / Sequence Export Help', '\u25CE', () => this._openDeliveryDialog());
        addOption('Export CDL (Grade)', '\u25CE', () => this._exportCDL());
        addOption('Import CDL (Grade)', '\u25CE', () => this._importCDL());
        addOption('Export Grade as .CUBE LUT', '\u25CE', () => this._exportGradeLUT());

        document.body.appendChild(menu);

        const closeMenu = (ev) => {
            if (!menu.contains(ev.target) && ev.target !== e.target) {
                menu.remove();
                document.removeEventListener('mousedown', closeMenu);
            }
        };
        setTimeout(() => document.addEventListener('mousedown', closeMenu), 10);
    };

    // _exportCDL and _exportGradeLUT are not overridden here. This file used
    // to replace the viewer's methods with its own copies of the grade maths,
    // so the File menu and the Inspector button wrote different files. Both
    // now run the viewer's one implementation (js/radiance_grade_export.js).

    RV.prototype._importCDL = function() {
        const input = document.createElement('input');
        input.type = 'file';
        input.accept = '.cdl,.xml';
        input.onchange = (e) => {
            const file = e.target.files[0];
            if (!file) return;
            const reader = new FileReader();
            reader.onload = (ev) => {
                try {
                    const parser = new DOMParser();
                    const doc = parser.parseFromString(ev.target.result, 'text/xml');
                    const slope = doc.querySelector('Slope')?.textContent?.trim().split(/\s+/).map(Number);
                    const offset = doc.querySelector('Offset')?.textContent?.trim().split(/\s+/).map(Number);
                    const power = doc.querySelector('Power')?.textContent?.trim().split(/\s+/).map(Number);
                    const satEl = doc.querySelector('Saturation');
                    const sat = satEl ? parseFloat(satEl.textContent) : 1.0;

                    // FIX-006: inverse of _exportCDL. The Viewer grade is
                    // ((in * 2^exposure + offset) * gain) ^ (1/gamma), so a CDL
                    // maps to gain = slope, offset = Offset / slope, gamma =
                    // 1/power, with exposure and lift at identity. Offset used to
                    // be loaded into the luma-pivoted lift.
                    if (slope && slope.length === 3) {
                        this.gain = slope;
                        if (this.renderer) this.renderer.setGain(...slope);
                        this.exposure = 0.0;
                        if (this.renderer) this.renderer.setExposure(0.0);
                        this.lift = [0, 0, 0];
                        if (this.renderer) this.renderer.setLift(0, 0, 0);
                    }
                    if (offset && offset.length === 3) {
                        const s3 = (slope && slope.length === 3) ? slope : [1, 1, 1];
                        const off = offset.map((o, i) => Math.abs(s3[i]) > 1e-9 ? o / s3[i] : o);
                        this.offset = off;
                        if (this.renderer) this.renderer.setOffset(...off);
                    }
                    if (power && power.length === 3) {
                        const gamma = power.map(p => p > 0 ? 1.0 / p : 1.0);
                        this.gamma = gamma;
                        if (this.renderer) this.renderer.setGamma(...gamma);
                    }
                    this.saturation = sat;
                    if (this.renderer) this.renderer.setSaturation(sat);
                    this.render();
                    var _origLog = window.__radianceOrigLog || console.log;
                    _origLog('[Radiance v3.0] CDL imported:', { slope, offset, power, sat });
                } catch (err) {
                    console.error('[Radiance v3.0] CDL import failed:', err);
                }
            };
            reader.readAsText(file);
        };
        input.click();
    };

    RV.prototype.exportSnapshot = function(format = 'png') {
        if (!this.image) return;

        if (format === 'exr') {
            const imgData = (this.lastResult || []).find(d => d.frame === this.currentFrame && !d.is_compare && !d.is_zdepth);
            if (imgData && imgData.exr_filename) {
                const sub = imgData.exr_subfolder ?? imgData.subfolder ?? '';
                const type = imgData.exr_type ?? imgData.type ?? 'temp';
                const url = api.apiURL(
                    `/view?filename=${encodeURIComponent(imgData.exr_filename)}`
                    + `&subfolder=${encodeURIComponent(sub)}`
                    + `&type=${encodeURIComponent(type)}`
                );
                const link = document.createElement('a');
                link.href = url;
                link.download = imgData.exr_filename;
                link.click();
                this._termLog?.('success', `[Export] Saved Source EXR: ${imgData.exr_filename}`);
            } else {
                this._termLog?.('warn', '[Export] Source EXR not available for this frame. Re-run the node to generate it.');
            }
            return;
        }

        if (format === 'exr32') {
            if (!this.useWebGL || !this.renderer) {
                this._termLog?.('warn', '[Export] EXR 32-bit export requires WebGL renderer.');
                return;
            }
            const result = this.renderer.readPixelsFloat32(
                this.imageWidth, this.imageHeight, this.lutIntensity || 1.0
            );
            if (!result) {
                this._termLog?.('warn', '[Export] Float32 readback failed (WebGL2 required).');
                return;
            }

            const blob = this._encodeEXR32(result.data, result.width, result.height);
            if (!blob) {
                this._termLog?.('warn', '[Export] EXR encoding failed.');
                return;
            }

            const url = URL.createObjectURL(blob);
            const link = document.createElement('a');
            link.download = `radiance_graded_linear_${Date.now()}.exr`;
            link.href = url;
            link.click();
            URL.revokeObjectURL(url);
            this._termLog?.('success', `[Export] Saved 32-bit graded EXR (scene-linear, ${this.sourceTag?.colorspace || 'Linear Rec.709'} primaries): ${result.width}\u00D7${result.height}`);
            return;
        }

        // H17: the result without the viewer-only look or overlays, in a
        // stated colour space (radiance_viewer.js _saveResultPNG).
        this._saveResultPNG();
    };

    // 3.5.0: primaries for the EXR "chromaticities" attribute, by the OCIO
    // colour space the node tagged the source with. The graded EXR carried
    // no colour metadata, so Nuke/Resolve could only guess its gamut.
    const EXR_CHROMA = {
        'Linear Rec.709 (sRGB)': [0.64, 0.33, 0.30, 0.60, 0.15, 0.06, 0.3127, 0.3290],
        'sRGB Encoded Rec.709 (sRGB)': [0.64, 0.33, 0.30, 0.60, 0.15, 0.06, 0.3127, 0.3290],
        'Linear Rec.2020': [0.708, 0.292, 0.170, 0.797, 0.131, 0.046, 0.3127, 0.3290],
        'Linear P3-D65': [0.680, 0.320, 0.265, 0.690, 0.150, 0.060, 0.3127, 0.3290],
        'ACEScg': [0.713, 0.293, 0.165, 0.830, 0.128, 0.044, 0.32168, 0.33767],
        'ACES2065-1': [0.7347, 0.2653, 0.0, 1.0, 0.0001, -0.0770, 0.32168, 0.33767],
    };

    RV.prototype._encodeEXR32 = function(pixels, width, height, colorspace = null) {
        if (!pixels || pixels.length < width * height * 4) return null;

        const nCh = 4;
        const bytesPerPixel = 4;
        const scanlineBytes = nCh * width * bytesPerPixel;

        const encoder = new TextEncoder();
        const encStr = (s) => {
            const b = encoder.encode(s);
            const r = new Uint8Array(b.length + 1);
            r.set(b); r[b.length] = 0;
            return r;
        };

        const headerParts = [];

        const channelNames = ['A', 'B', 'G', 'R'];
        const channelEntries = [];
        for (const ch of channelNames) {
            const nameBytes = encStr(ch);
            const entry = new Uint8Array(nameBytes.length + 16);
            entry.set(nameBytes, 0);
            const dv = new DataView(entry.buffer, entry.byteOffset);
            dv.setInt32(nameBytes.length, 2, true);
            dv.setUint8(nameBytes.length + 4, 0);
            dv.setInt32(nameBytes.length + 8, 1, true);
            dv.setInt32(nameBytes.length + 12, 1, true);
            channelEntries.push(entry);
        }
        const channelsValueLen = channelEntries.reduce((s, e) => s + e.length, 0) + 1;
        const channelsValue = new Uint8Array(channelsValueLen);
        let cp = 0;
        for (const e of channelEntries) { channelsValue.set(e, cp); cp += e.length; }
        channelsValue[cp] = 0;

        const writeAttr = (name, type, valueBytes) => {
            const n = encStr(name);
            const t = encStr(type);
            const sizeBytes = new Uint8Array(4);
            new DataView(sizeBytes.buffer).setInt32(0, valueBytes.length, true);
            headerParts.push(n, t, sizeBytes, valueBytes);
        };

        const chroma = EXR_CHROMA[colorspace || this.sourceTag?.colorspace] || EXR_CHROMA['Linear Rec.709 (sRGB)'];
        const chromaBytes = new Uint8Array(32);
        const chromaView = new DataView(chromaBytes.buffer);
        chroma.forEach((v, i) => chromaView.setFloat32(i * 4, v, true));
        writeAttr('channels', 'chlist', channelsValue);
        writeAttr('chromaticities', 'chromaticities', chromaBytes);
        writeAttr('compression', 'compression', new Uint8Array([0]));

        const dwBytes = new Uint8Array(16);
        const dwView = new DataView(dwBytes.buffer);
        dwView.setInt32(0, 0, true);
        dwView.setInt32(4, 0, true);
        dwView.setInt32(8, width - 1, true);
        dwView.setInt32(12, height - 1, true);
        writeAttr('dataWindow', 'box2i', dwBytes);
        writeAttr('displayWindow', 'box2i', dwBytes);
        writeAttr('lineOrder', 'lineOrder', new Uint8Array([0]));

        const parBytes = new Uint8Array(4);
        new DataView(parBytes.buffer).setFloat32(0, 1.0, true);
        writeAttr('pixelAspectRatio', 'float', parBytes);

        const swcBytes = new Uint8Array(8);
        writeAttr('screenWindowCenter', 'v2f', swcBytes);

        const swwBytes = new Uint8Array(4);
        new DataView(swwBytes.buffer).setFloat32(0, 1.0, true);
        writeAttr('screenWindowWidth', 'float', swwBytes);

        headerParts.push(new Uint8Array([0]));

        const headerSize = headerParts.reduce((s, p) => s + p.length, 0);
        const magicAndVersion = 8;
        const offsetTableSize = height * 8;
        const headerTotalSize = magicAndVersion + headerSize;
        const dataStart = headerTotalSize + offsetTableSize;
        const scanlineBlockSize = 4 + 4 + scanlineBytes;
        const totalSize = dataStart + height * scanlineBlockSize;

        const buffer = new ArrayBuffer(totalSize);
        const out = new Uint8Array(buffer);
        const view = new DataView(buffer);

        view.setUint32(0, 20000630, true);
        view.setUint32(4, 2, true);

        let wp = 8;
        for (const part of headerParts) {
            out.set(part, wp);
            wp += part.length;
        }

        for (let y = 0; y < height; y++) {
            const offset = dataStart + y * scanlineBlockSize;
            view.setUint32(wp, offset, true);
            view.setUint32(wp + 4, 0, true);
            wp += 8;
        }

        const chMap = [3, 2, 1, 0];

        for (let y = 0; y < height; y++) {
            const blockOff = dataStart + y * scanlineBlockSize;
            view.setInt32(blockOff, y, true);
            view.setUint32(blockOff + 4, scanlineBytes, true);

            const pixelOff = blockOff + 8;
            for (let ci = 0; ci < nCh; ci++) {
                const srcCh = chMap[ci];
                const chOff = pixelOff + ci * width * bytesPerPixel;
                for (let x = 0; x < width; x++) {
                    const srcIdx = (y * width + x) * 4 + srcCh;
                    view.setFloat32(chOff + x * 4, pixels[srcIdx], true);
                }
            }
        }

        var _origLog = window.__radianceOrigLog || console.log;
        _origLog(`[Radiance EXR] Encoded ${width}\u00D7${height}\u00D74ch FLOAT (${(totalSize / 1048576).toFixed(1)} MB)`);
        return new Blob([buffer], { type: 'application/octet-stream' });
    };
}
if (typeof window.RadianceViewer !== 'undefined' && window.RadianceViewer) {
    install(window.RadianceViewer);
} else {
    _queue.push(install);
}
})();
