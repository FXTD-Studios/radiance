"""
FXTD EXR Saver - Professional EXR Export for ComfyUI
Version: 1.0.0
Author: FXTD Studios

Professional EXR export solution with:
- 16-bit half float and 32-bit float support
- Multiple compression options (ZIP, PIZ, ZIPS, RLE, etc.)
- Multi-layer/multi-channel export
- Full metadata support
- Color space handling
- Batch export with frame numbering
- Alpha channel options
"""

# FIX: OpenMP duplicate library conflict
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'

from typing import Tuple, Dict, Any, Optional, List, Union

import torch
import numpy as np
import struct

# Robust EXR writer using OpenCV
def write_exr_cv2(filepath: str, image: np.ndarray, bit_depth: str = "float32"):
    """
    Write EXR using OpenCV - most reliable cross-platform method.
    
    Args:
        filepath: Output file path
        image: HWC numpy array (RGB or RGBA)
        bit_depth: "float32" or "float16" (float16 will be converted)
    """
    import cv2
    import os
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    
    # Convert to float32
    img = image.astype(np.float32)
    
    # Convert RGB to BGR for OpenCV
    if img.ndim == 3:
        if img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        elif img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGRA)
    
    # Write EXR
    success = cv2.imwrite(filepath, img)
    if success:
        print(f"[FXTD EXR] Saved: {filepath}")
    else:
        raise RuntimeError(f"OpenCV failed to write EXR: {filepath}")
    return success


def write_exr_from_channels(filepath: str, channels: Dict[str, np.ndarray], 
                            pixel_type: str = "FLOAT"):
    """
    Write EXR from separate channel dictionaries using OpenCV.
    
    Args:
        filepath: Output file path
        channels: Dict of channel_name -> 2D numpy array
        pixel_type: "HALF" or "FLOAT"
    """
    import cv2
    import os
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    
    # Reconstruct image from channels
    if 'R' in channels and 'G' in channels and 'B' in channels:
        r = channels['R'].astype(np.float32)
        g = channels['G'].astype(np.float32)
        b = channels['B'].astype(np.float32)
        
        if 'A' in channels:
            a = channels['A'].astype(np.float32)
            img = np.stack([b, g, r, a], axis=-1)  # BGRA for OpenCV
        else:
            img = np.stack([b, g, r], axis=-1)  # BGR for OpenCV
    else:
        # Handle other channel layouts
        first_channel = list(channels.values())[0]
        img = first_channel.astype(np.float32)
    
    # Write EXR
    success = cv2.imwrite(filepath, img)
    if success:
        print(f"[FXTD EXR] Saved: {filepath}")
    else:
        raise RuntimeError(f"OpenCV failed to write EXR: {filepath}")
    return success

import zlib
from datetime import datetime
import json
from pathlib import Path


# =============================================================================
# EXR FORMAT CONSTANTS
# =============================================================================

EXR_COMPRESSION = {
    "None": 0,
    "RLE": 1,
    "ZIPS": 2,
    "ZIP": 3,
    "PIZ": 4,
    "PXR24": 5,
    "B44": 6,
    "B44A": 7,
    "DWAA": 8,
    "DWAB": 9,
}

EXR_PIXEL_TYPE = {
    "UINT": 0,
    "HALF": 1,
    "FLOAT": 2,
}

EXR_LINE_ORDER = {
    "INCREASING_Y": 0,
    "DECREASING_Y": 1,
    "RANDOM_Y": 2,
}


# =============================================================================
# HALF FLOAT CONVERSION
# =============================================================================

def float32_to_float16(arr: np.ndarray) -> np.ndarray:
    """Convert float32 array to float16 (half precision)."""
    return arr.astype(np.float16)


def float16_to_bytes(arr: np.ndarray) -> bytes:
    """Convert float16 array to bytes."""
    return arr.astype(np.float16).tobytes()


def float32_to_bytes(arr: np.ndarray) -> bytes:
    """Convert float32 array to bytes."""
    return arr.astype(np.float32).tobytes()


# =============================================================================
# SIMPLE EXR WRITER (No dependencies)
# =============================================================================

class SimpleEXRWriter:
    """
    Simple EXR writer that doesn't require OpenEXR library.
    Supports basic EXR writing with ZIP compression.
    """
    
    MAGIC = 20000630
    VERSION = 2
    
    def __init__(self):
        pass
    
    def write(self, filepath: str, channels: Dict[str, np.ndarray],
              compression: str = "ZIP", pixel_type: str = "HALF",
              metadata: Dict = None):
        """
        Write EXR file.
        """
        first_channel = list(channels.values())[0]
        height, width = first_channel.shape[:2]
        
        channel_list = sorted(channels.keys())
        
        with open(filepath, 'wb') as f:
            f.write(struct.pack('<I', self.MAGIC))
            version = self.VERSION
            f.write(struct.pack('<I', version))
            
            self._write_header(f, width, height, channel_list, 
                             compression, pixel_type, metadata)
            
            self._write_scanlines(f, channels, channel_list, width, height,
                                 compression, pixel_type)
    
    def _write_header(self, f, width: int, height: int, 
                      channel_list: List[str], compression: str,
                      pixel_type: str, metadata: Dict = None):
        """Write EXR header."""
        
        self._write_attribute(f, "channels", "chlist", 
                             self._encode_channel_list(channel_list, pixel_type))
        
        comp_value = EXR_COMPRESSION.get(compression, 3)
        # FIX: Simple writer only supports single-scanline compression (ZIPS logic)
        # ZIP (3) requires 16-line blocks. If we write single lines labeled as ZIP, it's corrupt.
        # So we remap ZIP -> ZIPS (2) which is zlib-compressed single lines.
        if comp_value == 3:  # ZIP
            comp_value = 2  # ZIPS
        
        self._write_attribute(f, "compression", "compression",
                              struct.pack('<B', comp_value))
        
        self._write_attribute(f, "dataWindow", "box2i",
                             struct.pack('<iiii', 0, 0, width - 1, height - 1))
        
        self._write_attribute(f, "displayWindow", "box2i",
                             struct.pack('<iiii', 0, 0, width - 1, height - 1))
        
        self._write_attribute(f, "lineOrder", "lineOrder",
                             struct.pack('<B', EXR_LINE_ORDER["INCREASING_Y"]))
        
        self._write_attribute(f, "pixelAspectRatio", "float",
                             struct.pack('<f', 1.0))
        
        self._write_attribute(f, "screenWindowCenter", "v2f",
                             struct.pack('<ff', 0.0, 0.0))
        
        self._write_attribute(f, "screenWindowWidth", "float",
                             struct.pack('<f', 1.0))
        
        if metadata:
            for key, value in metadata.items():
                if isinstance(value, str):
                    self._write_attribute(f, key, "string", 
                                         self._encode_string(value))
                elif isinstance(value, float):
                    self._write_attribute(f, key, "float",
                                         struct.pack('<f', value))
                elif isinstance(value, int):
                    self._write_attribute(f, key, "int",
                                         struct.pack('<i', value))
        
        f.write(b'\x00')
    
    def _write_attribute(self, f, name: str, attr_type: str, data: bytes):
        """Write a single header attribute."""
        f.write(name.encode('ascii') + b'\x00')
        f.write(attr_type.encode('ascii') + b'\x00')
        f.write(struct.pack('<I', len(data)))
        f.write(data)
    
    def _encode_channel_list(self, channel_list: List[str], 
                            pixel_type: str) -> bytes:
        """Encode channel list for header."""
        data = b''
        ptype = EXR_PIXEL_TYPE.get(pixel_type, 1)
        
        for name in channel_list:
            data += name.encode('ascii') + b'\x00'
            data += struct.pack('<I', ptype)
            data += struct.pack('<B', 0)
            data += b'\x00\x00\x00'
            data += struct.pack('<i', 1)
            data += struct.pack('<i', 1)
        
        data += b'\x00'
        return data
    
    def _encode_string(self, s: str) -> bytes:
        """Encode string for header."""
        encoded = s.encode('utf-8')
        return struct.pack('<I', len(encoded)) + encoded
    
    def _write_scanlines(self, f, channels: Dict[str, np.ndarray],
                        channel_list: List[str], width: int, height: int,
                        compression: str, pixel_type: str):
        """Write scanline pixel data."""
        
        if pixel_type == "HALF":
            bytes_per_sample = 2
        else:
            bytes_per_sample = 4
        
        num_channels = len(channel_list)
        
        offset_table_pos = f.tell()
        offsets = []
        
        for y in range(height):
            f.write(struct.pack('<Q', 0))
        
        comp_type = EXR_COMPRESSION.get(compression, 3)
        
        for y in range(height):
            scanline_start = f.tell()
            offsets.append(scanline_start)
            
            f.write(struct.pack('<i', y))
            
            scanline_data = b''
            
            for channel_name in channel_list:
                channel = channels[channel_name]
                row = channel[y, :]
                
                if pixel_type == "HALF":
                    row_bytes = float16_to_bytes(row)
                else:
                    row_bytes = float32_to_bytes(row)
                
                scanline_data += row_bytes
            
            if comp_type in [2, 3]:
                compressed = zlib.compress(scanline_data)
                if len(compressed) < len(scanline_data):
                    scanline_data = compressed
            
            f.write(struct.pack('<I', len(scanline_data)))
            f.write(scanline_data)
        
        current_pos = f.tell()
        f.seek(offset_table_pos)
        for offset in offsets:
            f.write(struct.pack('<Q', offset))
        f.seek(current_pos)


# =============================================================================
# ADVANCED EXR WRITER (with OpenEXR if available)
# =============================================================================

def check_openexr_available():
    """Check if OpenEXR library is available and working."""
    # Disabled - OpenEXR library has compatibility issues with Python bindings
    # Using OpenCV or SimpleEXRWriter instead for reliable EXR export
    return False


def write_exr_openexr(filepath: str, channels: Dict[str, np.ndarray],
                      compression: str = "ZIP", pixel_type: str = "HALF",
                      metadata: Dict = None):
    """Write EXR using OpenEXR library."""
    import OpenEXR
    import Imath
    
    first_channel = list(channels.values())[0]
    height, width = first_channel.shape[:2]
    
    header = OpenEXR.Header(width, height)
    
    comp_map = {
        "None": Imath.Compression.NO_COMPRESSION,
        "RLE": Imath.Compression.RLE_COMPRESSION,
        "ZIPS": Imath.Compression.ZIPS_COMPRESSION,
        "ZIP": Imath.Compression.ZIP_COMPRESSION,
        "PIZ": Imath.Compression.PIZ_COMPRESSION,
        "PXR24": Imath.Compression.PXR24_COMPRESSION,
        "B44": Imath.Compression.B44_COMPRESSION,
        "B44A": Imath.Compression.B44A_COMPRESSION,
        "DWAA": Imath.Compression.DWAA_COMPRESSION,
        "DWAB": Imath.Compression.DWAB_COMPRESSION,
    }
    header['compression'] = comp_map.get(compression, Imath.Compression.ZIP_COMPRESSION)
    
    if pixel_type == "HALF":
        ptype = Imath.PixelType(Imath.PixelType.HALF)
    else:
        ptype = Imath.PixelType(Imath.PixelType.FLOAT)
    
    channel_defs = {}
    for name in channels.keys():
        channel_defs[name] = Imath.Channel(ptype)
    header['channels'] = channel_defs
    
    # Skip custom metadata - OpenEXR 3.x doesn't support arbitrary attributes easily
    # The essential data (channels, compression, pixel type) is already set above
    # Custom metadata like 'software', 'created' would require SMPTE-defined attributes
    pass
    
    out = OpenEXR.OutputFile(filepath, header)
    
    channel_data = {}
    for name, data in channels.items():
        if pixel_type == "HALF":
            channel_data[name] = data.astype(np.float16).tobytes()
        else:
            channel_data[name] = data.astype(np.float32).tobytes()
    
    out.writePixels(channel_data)
    out.close()


def write_exr_imageio(filepath: str, image: np.ndarray, pixel_type: str = "HALF"):
    """Write EXR using imageio library (fallback)."""
    try:
        import imageio
        import os
        # Ensure directory exists
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        image = image.astype(np.float32)
        imageio.imwrite(filepath, image, format='EXR-FI')
        return True
    except Exception as e:
        print(f"imageio EXR write failed: {e}")
        return False


# =============================================================================
# COLOR SPACE UTILITIES
# =============================================================================

def linear_to_srgb(img: np.ndarray) -> np.ndarray:
    """Convert linear to sRGB."""
    return np.where(
        img <= 0.0031308,
        img * 12.92,
        1.055 * np.power(np.maximum(img, 0), 1/2.4) - 0.055
    ).astype(np.float32)


def srgb_to_linear(img: np.ndarray) -> np.ndarray:
    """Convert sRGB to linear."""
    return np.where(
        img <= 0.04045,
        img / 12.92,
        np.power((np.maximum(img, 0) + 0.055) / 1.055, 2.4)
    ).astype(np.float32)


# =============================================================================
# COMFYUI NODES
# =============================================================================

class FXTDSaveEXR:
    """
    Professional EXR Saver Node
    """
    
    def __init__(self):
        self.output_dir = "output"
        self.type = "output"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "filename_prefix": ("STRING", {"default": "FXTD"}),
                "bit_depth": (["16-bit Half Float", "32-bit Float"],),
                "compression": ([
                    "ZIP", "ZIPS", "PIZ", "RLE", "None",
                    "PXR24", "B44", "B44A", "DWAA", "DWAB"
                ],),
            },
            "optional": {
                "input_color_space": (["sRGB", "Linear", "Raw"],),
                "output_color_space": (["Linear", "sRGB", "Same as Input"],),
                "alpha_mode": (["None", "From Image", "Solid White", "Solid Black"],),
                "premultiply_alpha": ("BOOLEAN", {"default": False}),
                "subfolder": ("STRING", {"default": ""}),
                "start_frame": ("INT", {"default": 1, "min": 0, "max": 999999}),
                "frame_padding": ("INT", {"default": 4, "min": 1, "max": 8}),
                "add_metadata": ("BOOLEAN", {"default": True}),
                "custom_metadata": ("STRING", {
                    "default": "",
                    "multiline": True,
                }),
                "channel_format": (["RGB", "BGR", "RGBA", "BGRA", "ACEScg"],),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO"
            },
        }
    
    RETURN_TYPES = ("STRING", "STRING", "INT")
    RETURN_NAMES = ("file_paths", "folder_path", "frame_count")
    FUNCTION = "save_exr"
    OUTPUT_NODE = True
    CATEGORY = "FXTD Studios/Radiance/IO"
    DESCRIPTION = "Save images as EXR files with full HDR and metadata support."
    
    def save_exr(self, images: torch.Tensor, filename_prefix: str,
                 bit_depth: str, compression: str,
                 input_color_space: str = "sRGB",
                 output_color_space: str = "Linear",
                 alpha_mode: str = "None",
                 premultiply_alpha: bool = False,
                 subfolder: str = "",
                 start_frame: int = 1,
                 frame_padding: int = 4,
                 add_metadata: bool = True,
                 custom_metadata: str = "",
                 channel_format: str = "RGB",
                 prompt=None, extra_pnginfo=None):
        
        output_dir = self.output_dir
        if subfolder:
            output_dir = os.path.join(output_dir, subfolder)
        os.makedirs(output_dir, exist_ok=True)
        
        pixel_type = "HALF" if "16" in bit_depth else "FLOAT"
        
        metadata = {}
        if add_metadata:
            metadata["software"] = "FXTD Studio - ComfyUI"
            metadata["created"] = datetime.now().isoformat()
            metadata["compression"] = compression
            metadata["bitDepth"] = bit_depth
            metadata["colorSpace"] = output_color_space
        
        if custom_metadata:
            for line in custom_metadata.strip().split('\n'):
                if '=' in line:
                    key, value = line.split('=', 1)
                    metadata[key.strip()] = value.strip()
        
        batch_size = images.shape[0]
        saved_paths = []
        
        use_openexr = check_openexr_available()
        writer = SimpleEXRWriter()
        
        for i in range(batch_size):
            img = images[i].cpu().numpy().astype(np.float32)
            
            if input_color_space == "sRGB" and output_color_space == "Linear":
                img_rgb = srgb_to_linear(img[..., :3])
            elif input_color_space == "Linear" and output_color_space == "sRGB":
                img_rgb = linear_to_srgb(img[..., :3])
            elif output_color_space == "Same as Input":
                img_rgb = img[..., :3]
            else:
                img_rgb = img[..., :3]
            
            if alpha_mode == "From Image" and img.shape[-1] == 4:
                alpha = img[..., 3]
            elif alpha_mode == "Solid White":
                alpha = np.ones(img_rgb.shape[:2], dtype=np.float32)
            elif alpha_mode == "Solid Black":
                alpha = np.zeros(img_rgb.shape[:2], dtype=np.float32)
            else:
                alpha = None
            
            if premultiply_alpha and alpha is not None:
                for c in range(3):
                    img_rgb[..., c] *= alpha
            
            channels = {}
            
            if channel_format in ["RGB", "RGBA", "ACEScg"]:
                channels["R"] = img_rgb[..., 0]
                channels["G"] = img_rgb[..., 1]
                channels["B"] = img_rgb[..., 2]
            elif channel_format in ["BGR", "BGRA"]:
                channels["B"] = img_rgb[..., 2]
                channels["G"] = img_rgb[..., 1]
                channels["R"] = img_rgb[..., 0]
            
            if alpha_mode != "None" and alpha is not None:
                channels["A"] = alpha
            
            frame_num = start_frame + i
            frame_str = str(frame_num).zfill(frame_padding)
            filename = f"{filename_prefix}_{frame_str}.exr"
            filepath = os.path.join(output_dir, filename)
            
            frame_metadata = metadata.copy()
            frame_metadata["frame"] = frame_num
            
            try:
                if use_openexr:
                    write_exr_openexr(filepath, channels, compression, 
                                     pixel_type, frame_metadata)
                else:
                    writer.write(filepath, channels, compression,
                                pixel_type, frame_metadata)
                
                saved_paths.append(filepath)
            except Exception as e:
                print(f"Error writing EXR {filepath}: {e}")
                try:
                    if alpha is not None:
                        full_img = np.concatenate([img_rgb, alpha[..., np.newaxis]], axis=-1)
                    else:
                        full_img = img_rgb
                    write_exr_imageio(filepath, full_img, pixel_type)
                    saved_paths.append(filepath)
                except Exception as e2:
                    print(f"Fallback also failed: {e2}")
        
        paths_str = "\n".join(saved_paths)
        
        # No UI images - the node is a saver, not a viewer
        return {"ui": {}, 
                "result": (paths_str, output_dir, batch_size)}


class FXTDSaveEXRMultiLayer:
    """
    Multi-Layer EXR Saver
    """
    
    def __init__(self):
        self.output_dir = "output"
        self.type = "output"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "filename": ("STRING", {"default": "multilayer.exr"}),
                "bit_depth": (["16-bit Half Float", "32-bit Float"],),
                "compression": (["ZIP", "PIZ", "ZIPS", "None"],),
            },
            "optional": {
                "layer_beauty": ("IMAGE",),
                "layer_diffuse": ("IMAGE",),
                "layer_specular": ("IMAGE",),
                "layer_emission": ("IMAGE",),
                "layer_normal": ("IMAGE",),
                "layer_depth": ("IMAGE",),
                "layer_alpha": ("IMAGE",),
                "layer_ao": ("IMAGE",),
                "layer_shadow": ("IMAGE",),
                "layer_uv": ("IMAGE",),
                "layer_motion": ("IMAGE",),
                "layer_crypto": ("IMAGE",),
                "layer_custom1": ("IMAGE",),
                "layer_custom2": ("IMAGE",),
                "layer_custom3": ("IMAGE",),
                "custom1_name": ("STRING", {"default": "custom1"}),
                "custom2_name": ("STRING", {"default": "custom2"}),
                "custom3_name": ("STRING", {"default": "custom3"}),
                "subfolder": ("STRING", {"default": ""}),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("file_path",)
    FUNCTION = "save_multilayer"
    OUTPUT_NODE = True
    CATEGORY = "FXTD Studios/Radiance/IO"
    DESCRIPTION = "Save multiple layers into a single multi-layer EXR file."
    
    def save_multilayer(self, filename: str, bit_depth: str, compression: str,
                        layer_beauty=None, layer_diffuse=None, layer_specular=None,
                        layer_emission=None, layer_normal=None, layer_depth=None,
                        layer_alpha=None, layer_ao=None, layer_shadow=None,
                        layer_uv=None, layer_motion=None, layer_crypto=None,
                        layer_custom1=None, layer_custom2=None, layer_custom3=None,
                        custom1_name: str = "custom1", custom2_name: str = "custom2",
                        custom3_name: str = "custom3", subfolder: str = ""):
        
        output_dir = self.output_dir
        if subfolder:
            output_dir = os.path.join(output_dir, subfolder)
        os.makedirs(output_dir, exist_ok=True)
        
        if not filename.lower().endswith('.exr'):
            filename += '.exr'
        
        filepath = os.path.join(output_dir, filename)
        
        layers = {}
        
        layer_map = {
            "beauty": layer_beauty,
            "diffuse": layer_diffuse,
            "specular": layer_specular,
            "emission": layer_emission,
            "normal": layer_normal,
            "depth": layer_depth,
            "alpha": layer_alpha,
            "ao": layer_ao,
            "shadow": layer_shadow,
            "uv": layer_uv,
            "motion": layer_motion,
            "crypto": layer_crypto,
            custom1_name: layer_custom1,
            custom2_name: layer_custom2,
            custom3_name: layer_custom3,
        }
        
        for layer_name, layer_data in layer_map.items():
            if layer_data is not None:
                img = layer_data[0].cpu().numpy().astype(np.float32)
                layers[layer_name] = img
        
        if not layers:
            return ("No layers provided",)
        
        channels = {}
        
        for layer_name, img in layers.items():
            if img.shape[-1] >= 3:
                channels[f"{layer_name}.R"] = img[..., 0]
                channels[f"{layer_name}.G"] = img[..., 1]
                channels[f"{layer_name}.B"] = img[..., 2]
            if img.shape[-1] == 4:
                channels[f"{layer_name}.A"] = img[..., 3]
            elif img.shape[-1] == 1 or len(img.shape) == 2:
                ch_data = img[..., 0] if len(img.shape) == 3 else img
                channels[f"{layer_name}.Y"] = ch_data
        
        pixel_type = "HALF" if "16" in bit_depth else "FLOAT"
        
        metadata = {
            "software": "FXTD Studio - ComfyUI",
            "created": datetime.now().isoformat(),
            "layers": ",".join(layers.keys()),
        }
        
        writer = SimpleEXRWriter()
        
        try:
            # Use OpenCV-based writer (most reliable)
            write_exr_from_channels(filepath, channels, pixel_type)
        except Exception as e:
            return (f"Error: {e}",)
        
        return (filepath,)


class FXTDSaveEXRSequence:
    """
    EXR Sequence Saver - VFX Standard Naming
    """
    
    def __init__(self):
        self.output_dir = "output"
        self.type = "output"
        self.frame_counter = {}
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "sequence_name": ("STRING", {"default": "sequence"}),
                "bit_depth": (["16-bit Half Float", "32-bit Float"],),
                "compression": (["ZIP", "PIZ", "ZIPS", "None"],),
            },
            "optional": {
                "output_folder": ("STRING", {"default": "exr_sequences"}),
                "start_frame": ("INT", {"default": 1001, "min": 0, "max": 999999}),
                "frame_padding": ("INT", {"default": 4, "min": 1, "max": 8}),
                "auto_increment": ("BOOLEAN", {"default": True}),
                "reset_counter": ("BOOLEAN", {"default": False}),
                "input_color_space": (["sRGB", "Linear"],),
                "add_alpha": ("BOOLEAN", {"default": False}),
            }
        }
    
    RETURN_TYPES = ("STRING", "INT", "INT")
    RETURN_NAMES = ("output_path", "first_frame", "last_frame")
    FUNCTION = "save_sequence"
    OUTPUT_NODE = True
    CATEGORY = "FXTD Studios/Radiance/IO"
    DESCRIPTION = "Save image sequence as EXR files with VFX-standard naming."
    
    def save_sequence(self, images: torch.Tensor, sequence_name: str,
                      bit_depth: str, compression: str,
                      output_folder: str = "exr_sequences",
                      start_frame: int = 1001, frame_padding: int = 4,
                      auto_increment: bool = True,
                      reset_counter: bool = False,
                      input_color_space: str = "sRGB",
                      add_alpha: bool = False):
        
        output_dir = os.path.join(self.output_dir, output_folder, sequence_name)
        os.makedirs(output_dir, exist_ok=True)
        
        if reset_counter and sequence_name in self.frame_counter:
            del self.frame_counter[sequence_name]
        
        if auto_increment:
            if sequence_name not in self.frame_counter:
                self.frame_counter[sequence_name] = start_frame
            current_frame = self.frame_counter[sequence_name]
        else:
            current_frame = start_frame
        
        batch_size = images.shape[0]
        pixel_type = "HALF" if "16" in bit_depth else "FLOAT"
        
        writer = SimpleEXRWriter()
        first_frame = current_frame
        
        for i in range(batch_size):
            img = images[i].cpu().numpy().astype(np.float32)
            
            if input_color_space == "sRGB":
                img_rgb = srgb_to_linear(img[..., :3])
            else:
                img_rgb = img[..., :3]
            
            channels = {
                "R": img_rgb[..., 0],
                "G": img_rgb[..., 1],
                "B": img_rgb[..., 2],
            }
            
            if add_alpha:
                if img.shape[-1] == 4:
                    channels["A"] = img[..., 3]
                else:
                    channels["A"] = np.ones(img_rgb.shape[:2], dtype=np.float32)
            
            frame_str = str(current_frame).zfill(frame_padding)
            filename = f"{sequence_name}.{frame_str}.exr"
            filepath = os.path.join(output_dir, filename)
            
            metadata = {
                "software": "FXTD Studio",
                "sequence": sequence_name,
                "frame": current_frame,
            }
            
            try:
                write_exr_from_channels(filepath, channels, pixel_type)
            except Exception as e:
                print(f"Error writing {filepath}: {e}")
            
            current_frame += 1
        
        if auto_increment:
            self.frame_counter[sequence_name] = current_frame
        
        last_frame = current_frame - 1
        
        return (output_dir, first_frame, last_frame)


class FXTDEXRChannelMerge:
    """
    Merge separate channel images into EXR channels.
    """
    
    def __init__(self):
        self.output_dir = "output"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "filename": ("STRING", {"default": "merged.exr"}),
                "bit_depth": (["16-bit Half Float", "32-bit Float"],),
                "compression": (["ZIP", "PIZ", "None"],),
            },
            "optional": {
                "red_channel": ("IMAGE",),
                "green_channel": ("IMAGE",),
                "blue_channel": ("IMAGE",),
                "alpha_channel": ("IMAGE",),
                "depth_channel": ("IMAGE",),
                "normal_channel": ("IMAGE",),
                "custom_channel_1": ("IMAGE",),
                "custom_channel_1_name": ("STRING", {"default": "custom1"}),
                "custom_channel_2": ("IMAGE",),
                "custom_channel_2_name": ("STRING", {"default": "custom2"}),
                "subfolder": ("STRING", {"default": ""}),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("file_path",)
    FUNCTION = "merge_channels"
    OUTPUT_NODE = True
    CATEGORY = "FXTD Studios/Radiance/IO"
    DESCRIPTION = "Merge separate images into EXR channels."
    
    def merge_channels(self, filename: str, bit_depth: str, compression: str,
                       red_channel=None, green_channel=None, blue_channel=None,
                       alpha_channel=None, depth_channel=None, normal_channel=None,
                       custom_channel_1=None, custom_channel_1_name: str = "custom1",
                       custom_channel_2=None, custom_channel_2_name: str = "custom2",
                       subfolder: str = ""):
        
        output_dir = self.output_dir
        if subfolder:
            output_dir = os.path.join(output_dir, subfolder)
        os.makedirs(output_dir, exist_ok=True)
        
        if not filename.lower().endswith('.exr'):
            filename += '.exr'
        filepath = os.path.join(output_dir, filename)
        
        channels = {}
        
        def get_channel(img_tensor, idx=0):
            if img_tensor is None:
                return None
            img = img_tensor[0].cpu().numpy().astype(np.float32)
            if len(img.shape) == 3:
                if img.shape[-1] >= 3 and idx < 0:
                    return 0.2126 * img[..., 0] + 0.7152 * img[..., 1] + 0.0722 * img[..., 2]
                return img[..., min(idx, img.shape[-1]-1)]
            return img
        
        if red_channel is not None:
            channels["R"] = get_channel(red_channel, 0)
        
        if green_channel is not None:
            channels["G"] = get_channel(green_channel, 1)
        
        if blue_channel is not None:
            channels["B"] = get_channel(blue_channel, 2)
        
        if alpha_channel is not None:
            channels["A"] = get_channel(alpha_channel, -1)
        
        if depth_channel is not None:
            channels["Z"] = get_channel(depth_channel, -1)
        
        if normal_channel is not None:
            n = normal_channel[0].cpu().numpy().astype(np.float32)
            if n.shape[-1] >= 3:
                channels["normal.X"] = n[..., 0]
                channels["normal.Y"] = n[..., 1]
                channels["normal.Z"] = n[..., 2]
        
        if custom_channel_1 is not None:
            channels[custom_channel_1_name] = get_channel(custom_channel_1, -1)
        
        if custom_channel_2 is not None:
            channels[custom_channel_2_name] = get_channel(custom_channel_2, -1)
        
        if not channels:
            return ("No channels provided",)
        
        pixel_type = "HALF" if "16" in bit_depth else "FLOAT"
        writer = SimpleEXRWriter()
        
        try:
            write_exr_from_channels(filepath, channels, pixel_type)
        except Exception as e:
            return (f"Error: {e}",)
        
        return (filepath,)


class FXTDSaveEXRCryptomatte:
    """
    Save Cryptomatte-compatible EXR for compositing.
    """
    
    def __init__(self):
        self.output_dir = "output"
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "filename": ("STRING", {"default": "crypto.exr"}),
                "compression": (["ZIP", "PIZ", "ZIPS"],),
            },
            "optional": {
                "crypto_object": ("IMAGE",),
                "crypto_material": ("IMAGE",),
                "crypto_asset": ("IMAGE",),
                "depth": ("IMAGE",),
                "subfolder": ("STRING", {"default": ""}),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("file_path",)
    FUNCTION = "save_crypto"
    OUTPUT_NODE = True
    CATEGORY = "FXTD Studios/Radiance/IO"
    DESCRIPTION = "Save Cryptomatte-compatible EXR for Nuke/Fusion/AE."
    
    def save_crypto(self, image: torch.Tensor, filename: str, compression: str,
                    crypto_object=None, crypto_material=None, crypto_asset=None,
                    depth=None, subfolder: str = ""):
        
        output_dir = self.output_dir
        if subfolder:
            output_dir = os.path.join(output_dir, subfolder)
        os.makedirs(output_dir, exist_ok=True)
        
        if not filename.lower().endswith('.exr'):
            filename += '.exr'
        filepath = os.path.join(output_dir, filename)
        
        img = image[0].cpu().numpy().astype(np.float32)
        
        channels = {
            "R": img[..., 0],
            "G": img[..., 1],
            "B": img[..., 2],
        }
        
        if img.shape[-1] == 4:
            channels["A"] = img[..., 3]
        
        def add_crypto_layer(name, tensor):
            if tensor is None:
                return
            t = tensor[0].cpu().numpy().astype(np.float32)
            if t.shape[-1] >= 3:
                channels[f"{name}00.R"] = t[..., 0]
                channels[f"{name}00.G"] = t[..., 1]
                channels[f"{name}00.B"] = t[..., 2]
                if t.shape[-1] == 4:
                    channels[f"{name}00.A"] = t[..., 3]
        
        add_crypto_layer("cryptoObject", crypto_object)
        add_crypto_layer("cryptoMaterial", crypto_material)
        add_crypto_layer("cryptoAsset", crypto_asset)
        
        if depth is not None:
            d = depth[0].cpu().numpy().astype(np.float32)
            channels["depth.Z"] = d[..., 0] if len(d.shape) == 3 else d
        
        writer = SimpleEXRWriter()
        
        metadata = {
            "software": "FXTD Studio",
            "cryptomatte": "1.0",
        }
        
        try:
            write_exr_from_channels(filepath, channels, "FLOAT")
        except Exception as e:
            return (f"Error: {e}",)
        
        return (filepath,)


# =============================================================================
# NODE MAPPINGS
# =============================================================================

NODE_CLASS_MAPPINGS = {
    "FXTDSaveEXR": FXTDSaveEXR,
    "FXTDSaveEXRMultiLayer": FXTDSaveEXRMultiLayer,
    "FXTDSaveEXRSequence": FXTDSaveEXRSequence,
    "FXTDEXRChannelMerge": FXTDEXRChannelMerge,
    "FXTDSaveEXRCryptomatte": FXTDSaveEXRCryptomatte,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FXTDSaveEXR": "💾 Radiance Save EXR",
    "FXTDSaveEXRMultiLayer": "📚 Radiance Save EXR Multi-Layer",
    "FXTDSaveEXRSequence": "🎬 Radiance Save EXR Sequence",
    "FXTDEXRChannelMerge": "🔀 Radiance EXR Channel Merge",
    "FXTDSaveEXRCryptomatte": "🎭 Radiance Save EXR Cryptomatte",
}
