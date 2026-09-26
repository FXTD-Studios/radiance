// ─────────────────────────────────────────────────────────────
// Radiance Viewer — UI Widget Module
// Prototype-extension for RadianceViewer
// Uses deferred setup so the module loads regardless of
// RadianceViewer's module evaluation order.
// ─────────────────────────────────────────────────────────────

(function() {
var R = window.RadianceViewer;
function install(RV) {
R = RV;
window.RadianceViewer = R;

// ══════════════════════════════════════════════════════════════
//                    Tactile SVG Knob Widget
// ══════════════════════════════════════════════════════════════
R.prototype.createKnob = function(label, min, max, initial, step, callback) {
    const container = document.createElement('div');
    container.style.cssText = `
        display: flex;
        flex-direction: column;
        align-items: center;
        gap: 6px;
        min-width: 60px;
        width: 70px;
        flex: 1;
        max-width: 90px;
        position: relative;
        padding: 4px;
        border-radius: 8px;
        background: rgba(255, 255, 255, 0.01);
        border: 1px solid rgba(255, 255, 255, 0.0);
        transition: transform 0.25s cubic-bezier(0.25, 0.8, 0.25, 1), background 0.25s, border-color 0.25s;
    `;

    container.onmouseenter = () => {
        container.style.transform = 'scale(1.05)';
        container.style.background = 'rgba(255, 255, 255, 0.03)';
        container.style.borderColor = 'rgba(255, 255, 255, 0.05)';
    };
    container.onmouseleave = () => {
        container.style.transform = 'scale(1.0)';
        container.style.background = 'rgba(255, 255, 255, 0.01)';
        container.style.borderColor = 'rgba(255, 255, 255, 0.0)';
    };

    const lbl = document.createElement('div');
    lbl.textContent = label;
    lbl.style.cssText = `
        color: ${this.theme.textDim};
        font-size: 8px;
        font-weight: 700;
        font-family: ${this.theme.font};
        letter-spacing: 0.8px;
        text-transform: uppercase;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
        max-width: 100%;
        text-align: center;
        transition: color 0.2s ease;
    `;

    const knobWrapper = document.createElement('div');
    knobWrapper.style.cssText = 'position: relative; width: 50px; height: 50px; display: flex; align-items: center; justify-content: center;';

    const size = 50;
    const strokeWidth = 3.5;
    const radius = (size - strokeWidth) / 2 - 2; // radius is 21px
    const circumference = 2 * Math.PI * radius;
    const arcLength = circumference * 0.75; // 270 degree track

    const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    svg.setAttribute("width", size);
    svg.setAttribute("height", size);
    svg.style.cssText = "cursor: ns-resize; touch-action: none; overflow: visible;";

    // SVG Defs for linear gradient fill of metallic cap
    const defs = document.createElementNS("http://www.w3.org/2000/svg", "defs");
    const metalGrad = document.createElementNS("http://www.w3.org/2000/svg", "linearGradient");
    metalGrad.setAttribute("id", "radiance-knob-metallic");
    metalGrad.setAttribute("x1", "0%"); metalGrad.setAttribute("y1", "0%");
    metalGrad.setAttribute("x2", "100%"); metalGrad.setAttribute("y2", "100%");
    metalGrad.innerHTML = `
        <stop offset="0%" stop-color="#2a2b35"/>
        <stop offset="50%" stop-color="#14151b"/>
        <stop offset="100%" stop-color="#0a0a0f"/>
    `;
    defs.appendChild(metalGrad);
    svg.appendChild(defs);

    // 1. Dark background track (270 degrees)
    const track = document.createElementNS("http://www.w3.org/2000/svg", "circle");
    track.setAttribute("cx", size / 2);
    track.setAttribute("cy", size / 2);
    track.setAttribute("r", radius);
    track.setAttribute("stroke", "rgba(255, 255, 255, 0.06)");
    track.setAttribute("stroke-width", strokeWidth);
    track.setAttribute("fill", "transparent");
    track.setAttribute("stroke-dasharray", `${arcLength} ${circumference}`);
    track.style.transform = "rotate(135deg)";
    track.style.transformOrigin = "center";
    svg.appendChild(track);

    // 2. Active colored progress sweep (270 degrees)
    const progress = document.createElementNS("http://www.w3.org/2000/svg", "circle");
    progress.setAttribute("cx", size / 2);
    progress.setAttribute("cy", size / 2);
    progress.setAttribute("r", radius);
    progress.setAttribute("stroke", this.theme.accent);
    progress.setAttribute("stroke-width", strokeWidth);
    progress.setAttribute("fill", "transparent");
    progress.setAttribute("stroke-dasharray", `${arcLength} ${circumference}`);
    progress.setAttribute("stroke-dashoffset", arcLength);
    progress.style.transform = "rotate(135deg)";
    progress.style.transformOrigin = "center";
    progress.style.transition = "stroke-dashoffset 0.08s ease-out, stroke 0.25s, filter 0.25s";
    svg.appendChild(progress);

    // 3. Central physical Dial Face (Metallic Cap)
    const dialFace = document.createElementNS("http://www.w3.org/2000/svg", "circle");
    dialFace.setAttribute("cx", size / 2);
    dialFace.setAttribute("cy", size / 2);
    dialFace.setAttribute("r", radius - 3.5);
    dialFace.setAttribute("fill", "url(#radiance-knob-metallic)");
    dialFace.setAttribute("stroke", "rgba(255, 255, 255, 0.12)");
    dialFace.setAttribute("stroke-width", "0.75");
    dialFace.style.filter = "drop-shadow(0 3px 6px rgba(0,0,0,0.4))";
    svg.appendChild(dialFace);

    // 4. Rotating tactile indicator needle/tick
    const tick = document.createElementNS("http://www.w3.org/2000/svg", "line");
    tick.setAttribute("x1", size / 2);
    tick.setAttribute("y1", size / 2 - 6); // Starts near center
    tick.setAttribute("x2", size / 2);
    tick.setAttribute("y2", size / 2 - radius + 2.5); // Ends near cap edge
    tick.setAttribute("stroke-linecap", "round");
    tick.style.transformOrigin = "center";
    tick.style.transition = "transform 0.08s ease-out, stroke 0.25s, filter 0.25s";
    svg.appendChild(tick);

    knobWrapper.appendChild(svg);

    // Value Display overlay
    const valDisplay = document.createElement('div');
    valDisplay.textContent = initial.toFixed(2);
    valDisplay.style.cssText = `
        font-family: ${this.theme.mono};
        font-size: 8px;
        font-weight: 700;
        color: ${this.theme.text};
        position: absolute;
        bottom: -2px;
        pointer-events: none;
        text-shadow: 0 1px 3px rgba(0, 0, 0, 0.9);
        z-index: 10;
        background: rgba(10,10,15,0.75);
        padding: 1px 4px;
        border-radius: 4px;
        border: 1px solid rgba(255, 255, 255, 0.05);
        letter-spacing: 0;
        transition: color 0.25s, border-color 0.25s;
    `;
    knobWrapper.appendChild(valDisplay);

    container.appendChild(lbl);
    container.appendChild(knobWrapper);

    let currentValue = initial;
    let startY = 0;
    let startVal = 0;

    // Dynamic color coding based on function and active value
    const updateVisuals = (val) => {
        const t = (val - min) / (max - min);
        const offset = arcLength - (t * arcLength);
        progress.setAttribute("stroke-dashoffset", offset);

        const angle = -135 + (t * 270);
        tick.style.transform = `rotate(${angle}deg)`;

        const isAtLimit = (val <= min + (max - min) * 0.001) || (val >= max - (max - min) * 0.001);
        
        // Define premium active highlights
        let activeColor = this.theme.accent;
        const uLabel = label.toUpperCase();
        if (uLabel === 'TEMP') {
            activeColor = val < 0 ? '#4fa3ff' : (val > 0 ? '#ffb03a' : 'rgba(255,255,255,0.4)');
        } else if (uLabel === 'TINT') {
            activeColor = val < 0 ? '#38ef7d' : (val > 0 ? '#ff3a85' : 'rgba(255,255,255,0.4)');
        } else if (uLabel === 'CONTRAST') {
            activeColor = '#c084fc'; // purple
        } else if (uLabel === 'PIVOT') {
            activeColor = '#fbbf24'; // amber
        } else if (uLabel === 'M.DETAIL') {
            activeColor = '#22d3ee'; // light-cyan
        } else if (uLabel === 'SOFT CLIP') {
            activeColor = '#f43f5e'; // rose
        }

        if (isAtLimit && val !== initial) {
            progress.setAttribute("stroke", "#ff4a4a");
            progress.style.filter = "drop-shadow(0 0 4px #ff4a4a)";
            tick.setAttribute("stroke", "#ff4a4a");
            tick.style.filter = "drop-shadow(0 0 3px #ff4a4a)";
            tick.setAttribute("stroke-width", "2");
            valDisplay.style.color = "#ff4a4a";
            valDisplay.style.borderColor = "rgba(255, 74, 74, 0.3)";
            lbl.style.color = "#ff4a4a";
        } else {
            const isNeutral = val === initial;
            const strokeColor = isNeutral ? "rgba(255,255,255,0.22)" : activeColor;
            
            progress.setAttribute("stroke", strokeColor);
            progress.style.filter = isNeutral ? "none" : `drop-shadow(0 0 4px ${activeColor}70)`;
            
            tick.setAttribute("stroke", strokeColor);
            tick.setAttribute("stroke-width", isNeutral ? "1.5" : "2");
            tick.style.filter = isNeutral ? "none" : `drop-shadow(0 0 3px ${activeColor}55)`;
            
            valDisplay.style.color = isNeutral ? this.theme.textDim : "#fff";
            valDisplay.style.borderColor = isNeutral ? "rgba(255, 255, 255, 0.05)" : `${activeColor}40`;
            lbl.style.color = isNeutral ? this.theme.textDim : activeColor;
        }

        valDisplay.textContent = val.toFixed(2);
    };

    updateVisuals(initial);

    container.updateValue = (val) => {
        currentValue = val;
        updateVisuals(val);
    };

    svg.onpointerdown = (e) => {
        e.preventDefault();
        this._pushUndo();
        startY = e.clientY;
        startVal = currentValue;

        svg.setPointerCapture(e.pointerId);
        container.style.transform = 'scale(1.08)';
        container.style.background = 'rgba(255, 255, 255, 0.05)';

        const onMove = (em) => {
            const delta = startY - em.clientY;
            const range = max - min;
            let sensitivity = range * 0.005;
            if (em.ctrlKey || em.metaKey) sensitivity *= 0.1;
            if (em.shiftKey) sensitivity *= 3.0;

            let newVal = startVal + (delta * sensitivity);
            newVal = Math.max(min, Math.min(newVal, max));

            if (step) newVal = Math.round(newVal / step) * step;

            currentValue = newVal;
            updateVisuals(currentValue);
            callback(currentValue);
        };

        const onUp = () => {
            svg.removeEventListener('pointermove', onMove);
            svg.removeEventListener('pointerup', onUp);
            svg.releasePointerCapture(e.pointerId);
            container.style.transform = 'scale(1.0)';
            container.style.background = 'rgba(255, 255, 255, 0.01)';
        };

        svg.addEventListener('pointermove', onMove);
        svg.addEventListener('pointerup', onUp);
    };

    svg.ondblclick = () => {
        this._pushUndo();
        currentValue = initial;
        updateVisuals(initial);
        callback(initial);
        // Quick visual spring
        container.style.transform = 'scale(0.95)';
        setTimeout(() => { container.style.transform = 'scale(1.0)'; }, 100);
    };

    svg.addEventListener('contextmenu', (e) => {
        e.preventDefault();
        e.stopPropagation();

        const existingInput = container.querySelector('.knob-numeric-input');
        if (existingInput) { existingInput.remove(); return; }

        const input = document.createElement('input');
        input.type = 'text';
        input.className = 'knob-numeric-input';
        input.value = currentValue.toFixed(Math.max(0, -Math.log10(step || 0.01)));
        input.style.cssText = `
            position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%);
            width: 48px; text-align: center; font-size: 9px; font-weight: bold;
            font-family: ${this.theme.mono}; color: #fff;
            background: rgba(8, 8, 12, 0.95);
            backdrop-filter: blur(10px);
            border: 1px solid ${this.theme.accent};
            border-radius: 4px; padding: 2px 4px; outline: none; z-index: 100;
            box-shadow: 0 4px 15px rgba(0,0,0,0.8), 0 0 8px ${this.theme.accent}33;
        `;
        container.appendChild(input);
        input.focus();
        input.select();

        const apply = () => {
            let val = parseFloat(input.value);
            if (!isNaN(val)) {
                this._pushUndo();
                val = Math.max(min, Math.min(val, max));
                if (step) val = Math.round(val / step) * step;
                currentValue = val;
                updateVisuals(currentValue);
                callback(currentValue);
            }
            input.remove();
        };

        input.onkeydown = (ke) => {
            if (ke.key === 'Enter') { ke.preventDefault(); apply(); }
            else if (ke.key === 'Escape') { ke.preventDefault(); input.remove(); }
            ke.stopPropagation();
        };
        input.onblur = () => { setTimeout(() => { if (input.parentNode) apply(); }, 80); };
    });

    return container;
};

// ══════════════════════════════════════════════════════════════
//                    Hollywood Color Wheel Widget
// ══════════════════════════════════════════════════════════════
R.prototype.createColorWheel = function(label, min, max, defaults, step, callback) {
    const container = document.createElement('div');
    container.style.cssText = `
        display: flex;
        flex-direction: column;
        align-items: center;
        gap: 6px;
        min-width: 120px;
        flex: 1;
        position: relative;
        padding: 6px;
        background: rgba(255,255,255,0.01);
        border-radius: 12px;
        border: 1px solid rgba(255,255,255,0);
        transition: transform 0.25s cubic-bezier(0.25, 0.8, 0.25, 1), background 0.25s;
    `;

    container.onmouseenter = () => {
        container.style.transform = 'scale(1.02)';
        container.style.background = 'rgba(255,255,255,0.02)';
    };
    container.onmouseleave = () => {
        container.style.transform = 'scale(1.0)';
        container.style.background = 'rgba(255,255,255,0.01)';
    };

    // Colorscience styling setup based on label
    let wheelAccent = this.theme.accent;
    const uLabel = label.toUpperCase();
    if (uLabel === 'LIFT' || uLabel === 'SHADOW') {
        wheelAccent = '#00f2ff'; // cyber cyan
    } else if (uLabel === 'GAMMA' || uLabel === 'MIDTONE') {
        wheelAccent = '#ffb800'; // warm gold
    } else if (uLabel === 'GAIN' || uLabel === 'HILIGHT') {
        wheelAccent = '#ff00b8'; // cyber magenta
    } else if (uLabel === 'OFFSET') {
        wheelAccent = '#e0f7fa'; // ice blue
    }

    const lbl = document.createElement('div');
    lbl.textContent = label;
    lbl.style.cssText = `
        color: ${wheelAccent};
        font-size: 9px;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        margin-bottom: 2px;
        text-shadow: 0 0 6px ${wheelAccent}44;
        transition: text-shadow 0.25s;
    `;
    container.appendChild(lbl);

    const size = 110;
    const center = size / 2;
    const wheelRadius = 38;
    const ringOuter = 52;
    const ringInner = 46;
    const puckRadius = 6;

    const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    svg.setAttribute("width", size);
    svg.setAttribute("height", size);
    svg.style.cssText = "cursor: default; touch-action: none; filter: drop-shadow(0 6px 16px rgba(0,0,0,0.5)); overflow: visible;";

    // Outer Master Slider Ring background
    const ringBg = document.createElementNS("http://www.w3.org/2000/svg", "circle");
    ringBg.setAttribute("cx", center); ringBg.setAttribute("cy", center);
    ringBg.setAttribute("r", (ringOuter + ringInner) / 2);
    ringBg.setAttribute("fill", "none");
    ringBg.setAttribute("stroke", "rgba(0,0,0,0.5)");
    ringBg.setAttribute("stroke-width", ringOuter - ringInner);
    svg.appendChild(ringBg);

    // Outer Master Slider active sweep
    const ringProgress = document.createElementNS("http://www.w3.org/2000/svg", "circle");
    const ringCirc = 2 * Math.PI * ((ringOuter + ringInner) / 2);
    ringProgress.setAttribute("cx", center); ringProgress.setAttribute("cy", center);
    ringProgress.setAttribute("r", (ringOuter + ringInner) / 2);
    ringProgress.setAttribute("fill", "none");
    ringProgress.setAttribute("stroke", wheelAccent);
    ringProgress.setAttribute("stroke-width", ringOuter - ringInner - 2.5);
    ringProgress.setAttribute("stroke-dasharray", ringCirc);
    ringProgress.setAttribute("stroke-dashoffset", ringCirc);
    ringProgress.setAttribute("stroke-linecap", "round");
    ringProgress.style.transformOrigin = "center";
    ringProgress.style.transform = "rotate(-90deg)";
    ringProgress.style.transition = "stroke-dashoffset 0.08s ease-out, stroke 0.25s, filter 0.25s";
    svg.appendChild(ringProgress);

    // Wheel border ring
    const wheelBorder = document.createElementNS("http://www.w3.org/2000/svg", "circle");
    wheelBorder.setAttribute("cx", center); wheelBorder.setAttribute("cy", center);
    wheelBorder.setAttribute("r", wheelRadius);
    wheelBorder.setAttribute("fill", "#09090d");
    wheelBorder.setAttribute("stroke", "rgba(255,255,255,0.06)");
    wheelBorder.setAttribute("stroke-width", "1");
    svg.appendChild(wheelBorder);

    // ── HIGH-FIDELITY SPECTRUM BACKGROUND ──
    // Embed a modern conic gradient backplane inside the wheel radius using a foreignObject
    const foreign = document.createElementNS("http://www.w3.org/2000/svg", "foreignObject");
    foreign.setAttribute("x", center - wheelRadius);
    foreign.setAttribute("y", center - wheelRadius);
    foreign.setAttribute("width", wheelRadius * 2);
    foreign.setAttribute("height", wheelRadius * 2);
    foreign.style.pointerEvents = "none";

    const spectrumBg = document.createElement('div');
    spectrumBg.style.cssText = `
        width: 100%;
        height: 100%;
        border-radius: 50%;
        background: conic-gradient(from 90deg, #ff4545 0%, #ffff45 17%, #45ff45 33%, #45ffff 50%, #4545ff 67%, #ff45ff 83%, #ff4545 100%);
        mask: radial-gradient(circle, transparent 15%, rgba(0,0,0,0.85) 100%);
        -webkit-mask: radial-gradient(circle, transparent 15%, rgba(0,0,0,0.85) 100%);
        opacity: 0.22;
        filter: blur(1.5px);
        box-shadow: inset 0 0 10px rgba(0,0,0,0.8);
        pointer-events: none;
    `;
    foreign.appendChild(spectrumBg);
    svg.appendChild(foreign);

    // Precision Dotted Holographic Crosshairs matching wheel theme color
    const crossStyle = `stroke: ${wheelAccent}22; stroke-width: 0.75; stroke-dasharray: 2, 3;`;
    const hLine = document.createElementNS("http://www.w3.org/2000/svg", "line");
    hLine.setAttribute("x1", center - wheelRadius); hLine.setAttribute("x2", center + wheelRadius);
    hLine.setAttribute("y1", center); hLine.setAttribute("y2", center); hLine.setAttribute("style", crossStyle);
    svg.appendChild(hLine);
    
    const vLine = document.createElementNS("http://www.w3.org/2000/svg", "line");
    vLine.setAttribute("x1", center); vLine.setAttribute("x2", center);
    vLine.setAttribute("y1", center - wheelRadius); vLine.setAttribute("y2", center + wheelRadius);
    vLine.setAttribute("style", crossStyle);
    svg.appendChild(vLine);

    // Fine inner circle reticle center reference
    const centerRef = document.createElementNS("http://www.w3.org/2000/svg", "circle");
    centerRef.setAttribute("cx", center); centerRef.setAttribute("cy", center);
    centerRef.setAttribute("r", 4);
    centerRef.setAttribute("fill", "none");
    centerRef.setAttribute("stroke", "rgba(255,255,255,0.06)");
    centerRef.setAttribute("stroke-width", "0.5");
    svg.appendChild(centerRef);

    // Hollow design reticle (puck) for coordinate clarity
    const puck = document.createElementNS("http://www.w3.org/2000/svg", "circle");
    puck.setAttribute("r", puckRadius);
    puck.setAttribute("fill", "rgba(0, 0, 0, 0.45)");
    puck.setAttribute("stroke", wheelAccent);
    puck.setAttribute("stroke-width", "1.75");
    puck.style.cursor = "crosshair";
    puck.style.filter = `drop-shadow(0 0 4px ${wheelAccent})`;
    puck.style.transition = "cx 0.05s linear, cy 0.05s linear, stroke-width 0.2s, stroke 0.2s";
    svg.appendChild(puck);

    container.appendChild(svg);

    let wheelR = 0, wheelG = 0, wheelB = 0;
    let masterVal = (defaults[0] + defaults[1] + defaults[2]) / 3.0;

    const updateVisuals = () => {
        const angleR = 0, angleG = 2 * Math.PI / 3, angleB = 4 * Math.PI / 3;
        let dx = wheelR * Math.cos(angleR) + wheelG * Math.cos(angleG) + wheelB * Math.cos(angleB);
        let dy = wheelR * Math.sin(angleR) + wheelG * Math.sin(angleG) + wheelB * Math.sin(angleB);

        const pxSens = wheelRadius / 0.5;
        let px = dx * pxSens;
        let py = dy * pxSens;

        const dist = Math.sqrt(px * px + py * py);
        if (dist > wheelRadius) {
            px = (px / dist) * wheelRadius;
            py = (py / dist) * wheelRadius;
        }

        puck.setAttribute("cx", center + px);
        puck.setAttribute("cy", center + py);

        const range = max - min;
        const t = Math.max(0, Math.min(1, (masterVal - min) / range));
        const offset = ringCirc * (1 - t);
        ringProgress.setAttribute("stroke-dashoffset", offset);

        const isDefaultMaster = Math.abs(masterVal - (defaults[0] + defaults[1] + defaults[2]) / 3) < 0.001;
        const isDefaultCh = Math.abs(wheelR) < 0.001 && Math.abs(wheelG) < 0.001 && Math.abs(wheelB) < 0.001;

        ringProgress.setAttribute("stroke", isDefaultMaster ? "rgba(255,255,255,0.18)" : wheelAccent);
        ringProgress.style.filter = isDefaultMaster ? "none" : `drop-shadow(0 0 5px ${wheelAccent}55)`;
        
        puck.setAttribute("stroke", isDefaultCh ? "rgba(255,255,255,0.4)" : wheelAccent);
        puck.style.filter = isDefaultCh ? "none" : `drop-shadow(0 0 5px ${wheelAccent})`;

        lbl.style.textShadow = isDefaultCh && isDefaultMaster ? 'none' : `0 0 8px ${wheelAccent}66`;
    };

    const initialAvg = (defaults[0] + defaults[1] + defaults[2]) / 3.0;
    wheelR = defaults[0] - initialAvg;
    wheelG = defaults[1] - initialAvg;
    wheelB = defaults[2] - initialAvg;
    masterVal = initialAvg;
    updateVisuals();

    svg.onpointerdown = (e) => {
        e.preventDefault();
        this._pushUndo();
        svg.setPointerCapture(e.pointerId);

        const rect = svg.getBoundingClientRect();
        const startX = e.clientX, startY = e.clientY;
        const distToCenter = Math.sqrt(Math.pow(startX - (rect.left + center), 2) + Math.pow(startY - (rect.top + center), 2));

        const isRingDrag = distToCenter > (wheelRadius + 2);
        const initialMaster = masterVal;
        
        puck.setAttribute("stroke-width", "2.25");
        container.style.transform = 'scale(1.03)';

        const onMove = (em) => {
            if (isRingDrag) {
                const deltaY = startY - em.clientY;
                let sens = (max - min) * 0.005;
                if (em.ctrlKey) sens *= 0.1;
                if (em.shiftKey) sens *= 3.0;
                masterVal = Math.max(min, Math.min(max, initialMaster + deltaY * sens));
            } else {
                let dx = em.clientX - (rect.left + center);
                let dy = em.clientY - (rect.top + center);
                const dist = Math.sqrt(dx * dx + dy * dy);
                if (dist > wheelRadius) {
                    dx = (dx / dist) * wheelRadius;
                    dy = (dy / dist) * wheelRadius;
                }
                const mag = dist / wheelRadius;
                const ang = Math.atan2(dy, dx);
                const strength = mag * 0.5;
                wheelR = Math.cos(ang) * strength;
                wheelG = Math.cos(ang - 2 * Math.PI / 3) * strength;
                wheelB = Math.cos(ang - 4 * Math.PI / 3) * strength;
            }
            updateVisuals();
            callback(wheelR + masterVal, wheelG + masterVal, wheelB + masterVal);
        };

        const onUp = () => {
            svg.removeEventListener('pointermove', onMove);
            svg.removeEventListener('pointerup', onUp);
            svg.releasePointerCapture(e.pointerId);
            puck.setAttribute("stroke-width", "1.75");
            container.style.transform = 'scale(1.0)';
        };

        svg.addEventListener('pointermove', onMove);
        svg.addEventListener('pointerup', onUp);
    };

    svg.ondblclick = (e) => {
        const rect = svg.getBoundingClientRect();
        const distToCenter = Math.sqrt(Math.pow(e.clientX - (rect.left + center), 2) + Math.pow(e.clientY - (rect.top + center), 2));
        this._pushUndo();
        if (distToCenter > (wheelRadius + 2)) {
            masterVal = (defaults[0] + defaults[1] + defaults[2]) / 3.0;
        } else {
            wheelR = 0; wheelG = 0; wheelB = 0;
        }
        updateVisuals();
        callback(wheelR + masterVal, wheelG + masterVal, wheelB + masterVal);
        // Quick visual pop
        container.style.transform = 'scale(0.97)';
        setTimeout(() => { container.style.transform = 'scale(1.0)'; }, 100);
    };

    svg.addEventListener('contextmenu', (e) => {
        e.preventDefault();
        this._pushUndo();
        wheelR = 0; wheelG = 0; wheelB = 0;
        masterVal = (defaults[0] + defaults[1] + defaults[2]) / 3.0;
        updateVisuals();
        callback(wheelR + masterVal, wheelG + masterVal, wheelB + masterVal);
    });

    container.updateValue = (r, g, b) => {
        const m = (r + g + b) / 3.0;
        wheelR = r - m; wheelG = g - m; wheelB = b - m;
        masterVal = m;
        updateVisuals();
    };

    return container;
};

// ══════════════════════════════════════════════════════════════
//                    LEGACY CONTROL ROW (SLIDER)
// ══════════════════════════════════════════════════════════════
R.prototype.createControlRow = function(label, min, max, initial, step, callback) {
    const row = document.createElement('div');
    row.style.cssText = 'display: flex; flex-direction: column; gap: 4px; width: 100%;';

    const metaRow = document.createElement('div');
    metaRow.style.cssText = 'display: flex; justify-content: space-between; align-items: center; width: 100%;';

    const lbl = document.createElement('div');
    lbl.textContent = label;
    lbl.style.cssText = `
        color: ${this.theme.textDim};
        font-size: 10px;
        font-family: ${this.theme.font};
        min-width: 50px;
        font-weight: bold;
        letter-spacing: 0.4px;
    `;

    const value = document.createElement('div');
    value.textContent = initial.toFixed(2);
    value.style.cssText = `
        font-family: ${this.theme.mono};
        font-size: 10px;
        color: ${this.theme.accent};
        min-width: 40px;
        text-align: right;
        font-variant-numeric: tabular-nums;
        font-weight: bold;
    `;

    metaRow.appendChild(lbl);
    metaRow.appendChild(value);

    const sliderContainer = document.createElement('div');
    sliderContainer.style.cssText = 'position: relative; height: 5px; background: rgba(255,255,255,0.06); border-radius: 3px; margin-top: 2px; border: 1px solid rgba(255,255,255,0.03);';

    const sliderFill = document.createElement('div');
    sliderFill.style.cssText = `
        position: absolute; left: 0; top: 0; height: 100%; background: #445;
        width: 50%; pointer-events: none; border-radius: 2px;
        box-shadow: 0 0 6px ${this.theme.accent}33;
    `;

    const sliderInput = document.createElement('input');
    sliderInput.type = 'range';
    sliderInput.min = min; sliderInput.max = max; sliderInput.step = step; sliderInput.value = initial;
    sliderInput.style.cssText = `
        position: absolute; left: 0; top: -6px; width: 100%; height: 16px; opacity: 0; cursor: ew-resize; margin: 0;
    `;

    const updateVisuals = (val) => {
        const pct = ((val - min) / (max - min)) * 100;
        sliderFill.style.width = pct + '%';
        sliderFill.style.background = this.theme.accent;
        value.textContent = parseFloat(val).toFixed(2);
    };
    updateVisuals(initial);

    sliderInput.oninput = (e) => {
        const v = parseFloat(e.target.value);
        updateVisuals(v);
        callback(v);
    };

    sliderInput.ondblclick = () => {
        sliderInput.value = initial;
        updateVisuals(initial);
        callback(initial);
    };

    sliderContainer.appendChild(sliderFill);
    sliderContainer.appendChild(sliderInput);

    row.appendChild(metaRow);
    row.appendChild(sliderContainer);

    return row;
};
}

if (typeof window.RadianceViewer !== 'undefined' && window.RadianceViewer) {
    install(window.RadianceViewer);
} else {
    var _queue = window.__radianceInit = window.__radianceInit || [];
    _queue.push(install);
}
})();
