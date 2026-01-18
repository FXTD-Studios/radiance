import { app } from "../../scripts/app.js";

/**
 * ═══════════════════════════════════════════════════════════════════════════════
 *                         RADIANCE UI THEME
 *                    Premium Styling for Custom Nodes
 * ═══════════════════════════════════════════════════════════════════════════════
 */

// Elegant Color Palette
const COLORS = {
    HEADER: "#232330",       // Dark Slate Blue
    BODY: "#0f0f14",         // Deep Black/Blue
    BORDER: "#4a4a6a",       // Muted Blue-Grey
    TEXT_HEADER: "#ffffff",  // White
    TEXT_BODY: "#a0a0b0",    // Dim White
    ACCENT: "#00a8ff"        // Radiance Blue
};

app.registerExtension({
    name: "FXTD.Radiance.Style",
    
    async nodeCreated(node, app) {
        // Identify Radiance nodes by their class name prefix
        const isRadiance = node.comfyClass && (
            node.comfyClass.startsWith("FXTD") || 
            node.comfyClass.startsWith("Radiance") ||
            node.comfyClass.includes("Radiance")
        );

        if (isRadiance) {
            // Apply Colors
            node.color = COLORS.HEADER;
            node.bgcolor = COLORS.BODY;
            
            // Optional: Set shape to box if not set (standard is 1)
            // node.shape = 1; 

            // If the node has a widget resizing method, we can hook it here if needed, 
            // but for now, color is the main request.
            
            // Force redraw to apply immediate changes if needed
            if (node.graph) {
                node.setDirtyCanvas(true, true);
            }
        }
    }
});
