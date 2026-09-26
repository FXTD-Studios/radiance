import torch
import logging
import json
import os
import xml.etree.ElementTree as std_ET
import defusedxml.ElementTree as ET

from radiance.path_utils import resolve_input_path, resolve_output_path

logger = logging.getLogger("radiance.cdl")

class RadianceCDLTransform:
    CATEGORY = "FXTD STUDIOS/Radiance/◎ Color"
    DESCRIPTION = "Apply an ASC CDL (Slope/Offset/Power/Saturation) colour transform."
    FUNCTION = "apply"
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "cdl_info")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Image to grade. The CDL maths is applied to the values as they arrive (no colour space conversion), so feed it the encoding the CDL was authored in."}),
                "slope_r": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 4.0, "step": 0.01, "tooltip": "Red channel slope (gain). 1.0 = unity."}),
                "slope_g": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 4.0, "step": 0.01, "tooltip": "Green channel slope (gain). 1.0 = unity."}),
                "slope_b": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 4.0, "step": 0.01, "tooltip": "Blue channel slope (gain). 1.0 = unity."}),
                "offset_r": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.001, "tooltip": "Red channel offset. 0.0 = no shift."}),
                "offset_g": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.001, "tooltip": "Green channel offset. 0.0 = no shift."}),
                "offset_b": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.001, "tooltip": "Blue channel offset. 0.0 = no shift."}),
                "power_r": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 4.0, "step": 0.01, "tooltip": "Red channel power (gamma)."}),
                "power_g": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 4.0, "step": 0.01, "tooltip": "Green channel power (gamma)."}),
                "power_b": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 4.0, "step": 0.01, "tooltip": "Blue channel power (gamma)."}),
                "saturation": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 4.0, "step": 0.01, "tooltip": "Global saturation. 1.0 = unity."}),
            },
            "optional": {
                "cdl_data": ("STRING", {"forceInput": True, "tooltip": "JSON CDL data from Radiance CDL Import. When connected, its slope, offset, power and saturation values replace the sliders above."}),
            }
        }

    def apply(self, image: torch.Tensor, slope_r, slope_g, slope_b,
              offset_r, offset_g, offset_b, power_r, power_g, power_b,
              saturation, cdl_data=None):
        if cdl_data:
            try:
                d = json.loads(cdl_data)
                slope_r, slope_g, slope_b = d.get("slope", [slope_r, slope_g, slope_b])
                offset_r, offset_g, offset_b = d.get("offset", [offset_r, offset_g, offset_b])
                power_r, power_g, power_b = d.get("power", [power_r, power_g, power_b])
                saturation = d.get("saturation", saturation)
            except Exception as exc:
                logger.warning("[nodes_cdl] apply: %s", exc)

        device = image.device
        img = image.clone()
        slope = torch.tensor([slope_r, slope_g, slope_b], device=device)
        offset = torch.tensor([offset_r, offset_g, offset_b], device=device)
        power = torch.tensor([power_r, power_g, power_b], device=device)

        img = img * slope + offset
        img = torch.clamp(img, min=0.0)
        img = torch.pow(img, power)

        if saturation != 1.0:
            luma = (0.2126 * img[..., 0] + 0.7152 * img[..., 1] + 0.0722 * img[..., 2]).unsqueeze(-1)
            img = luma + saturation * (img - luma)
            img = torch.clamp(img, min=0.0)

        cdl_info = json.dumps({
            "slope": [slope_r, slope_g, slope_b],
            "offset": [offset_r, offset_g, offset_b],
            "power": [power_r, power_g, power_b],
            "saturation": saturation
        })
        return (img, cdl_info)


class RadianceCDLImport:
    CATEGORY = "FXTD STUDIOS/Radiance/◎ Color"
    DESCRIPTION = "Import an ASC CDL (.cdl / .cc / .ccc) file into pipeline metadata."
    FUNCTION = "load"
    RETURN_TYPES = ("STRING", "FLOAT", "FLOAT", "FLOAT", "FLOAT", "FLOAT", "FLOAT", "FLOAT", "FLOAT", "FLOAT", "FLOAT")
    RETURN_NAMES = ("cdl_data", "slope_r", "slope_g", "slope_b", "offset_r", "offset_g", "offset_b", "power_r", "power_g", "power_b", "saturation")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "file_path": ("STRING", {
                    "default": "grading/shot_01.cdl",
                    "tooltip": (
                        "Path to a .cdl, .cc, or .ccc file. A relative path is looked "
                        "for in ComfyUI's input/ then output/ folder; absolute paths "
                        "are used as given."
                    ),
                }),
            }
        }

    def load(self, file_path):
        file_path = resolve_input_path(file_path)
        if not os.path.isfile(file_path):
            logger.error(f"[CDL Import] File not found: {file_path}")
            return (json.dumps({}), 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0)
        try:
            tree = ET.parse(file_path)
            root = tree.getroot()
            def get_vec3(tag_name):
                node = root.find(f'.//{{*}}{tag_name}')
                if node is not None:
                    return [float(x) for x in node.text.split()]
                return None
            slope = get_vec3('Slope') or [1.0, 1.0, 1.0]
            offset = get_vec3('Offset') or [0.0, 0.0, 0.0]
            power = get_vec3('Power') or [1.0, 1.0, 1.0]
            sat_node = root.find('.//{*}Saturation')
            saturation = float(sat_node.text) if sat_node is not None else 1.0
            data = {"slope": slope, "offset": offset, "power": power, "saturation": saturation}
            return (json.dumps(data), *slope, *offset, *power, saturation)
        except Exception as e:
            logger.error(f"[CDL Import] Failed to parse CDL: {e}")
            return (json.dumps({}), 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0)


class RadianceCDLExport:
    CATEGORY = "FXTD STUDIOS/Radiance/◎ Color"
    DESCRIPTION = "Export current CDL values to an ASC-compliant .cdl or .cc file."
    FUNCTION = "save"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("file_path",)
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "file_path": ("STRING", {
                    "default": "grading/shot_01_output.cdl",
                    "tooltip": (
                        "Destination .cdl path. A relative path is written under "
                        "ComfyUI's output/ folder; absolute paths are used as given."
                    ),
                }),
                "slope_r": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 4.0, "step": 0.001, "tooltip": "Red slope (gain) written to the file. 1.0 = unity."}),
                "slope_g": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 4.0, "step": 0.001, "tooltip": "Green slope (gain) written to the file. 1.0 = unity."}),
                "slope_b": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 4.0, "step": 0.001, "tooltip": "Blue slope (gain) written to the file. 1.0 = unity."}),
                "offset_r": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.0001, "tooltip": "Red offset written to the file. 0.0 = no shift."}),
                "offset_g": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.0001, "tooltip": "Green offset written to the file. 0.0 = no shift."}),
                "offset_b": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.0001, "tooltip": "Blue offset written to the file. 0.0 = no shift."}),
                "power_r": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 4.0, "step": 0.001, "tooltip": "Red power (exponent) written to the file. 1.0 = unity."}),
                "power_g": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 4.0, "step": 0.001, "tooltip": "Green power (exponent) written to the file. 1.0 = unity."}),
                "power_b": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 4.0, "step": 0.001, "tooltip": "Blue power (exponent) written to the file. 1.0 = unity."}),
                "saturation": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 4.0, "step": 0.001, "tooltip": "Saturation written to the file's SaturationNode. 1.0 = unity."}),
            },
            "optional": {
                "cdl_data": ("STRING", {"forceInput": True, "tooltip": "JSON CDL data (from CDL Import or the cdl_info output of CDL Transform). When connected, its values replace the sliders above."}),
            }
        }

    def save(self, file_path, slope_r, slope_g, slope_b, offset_r, offset_g, offset_b,
             power_r, power_g, power_b, saturation, cdl_data=None):
        file_path = resolve_output_path(file_path)
        if cdl_data:
            try:
                d = json.loads(cdl_data)
                slope_r, slope_g, slope_b = d.get("slope", [slope_r, slope_g, slope_b])
                offset_r, offset_g, offset_b = d.get("offset", [offset_r, offset_g, offset_b])
                power_r, power_g, power_b = d.get("power", [power_r, power_g, power_b])
                saturation = d.get("saturation", saturation)
            except Exception as exc:
                logger.warning("[nodes_cdl] save: %s", exc)

        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        root = std_ET.Element("ColorDecisionList", {"xmlns": "urn:ASC:CDL:v1.01"})
        cd = std_ET.SubElement(root, "ColorDecision")
        sop = std_ET.SubElement(cd, "SOPNode")
        std_ET.SubElement(sop, "Slope").text = f"{slope_r:.6f} {slope_g:.6f} {slope_b:.6f}"
        std_ET.SubElement(sop, "Offset").text = f"{offset_r:.6f} {offset_g:.6f} {offset_b:.6f}"
        std_ET.SubElement(sop, "Power").text = f"{power_r:.6f} {power_g:.6f} {power_b:.6f}"
        sat = std_ET.SubElement(cd, "SaturationNode")
        std_ET.SubElement(sat, "Saturation").text = f"{saturation:.6f}"
        tree = std_ET.ElementTree(root)
        tree.write(file_path, xml_declaration=True, encoding='UTF-8')
        return (file_path,)
