import os
import hashlib
import logging
logger = logging.getLogger("radiance")
# io, base64, json, traceback removed — were imported but never used in code

from PIL import Image, ImageOps, ImageSequence
import numpy as np
import torch

import folder_paths
import node_helpers

class RadianceLoadImageMask:
    """
    Advanced Image Loader + Mask Editor for Radiance.
    
    This node loads an image, but also checks for a companion `_radmask.png` file.
    If the mask file exists, it overrides the default alpha channel.
    The companion frontend extension provides a WebGL-based soft brush mask editor
    that saves the non-destructive mask via the ComfyUI API.
    """
    @classmethod
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_input_directory()
        # Guard against a missing/invalid input directory (fresh ComfyUI install,
        # or a non-existent stub path on a different OS) — INPUT_TYPES must never
        # raise, or the node fails to register.
        try:
            files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        except (FileNotFoundError, NotADirectoryError, OSError):
            files = []
        files = folder_paths.filter_files_content_types(files, ["image"])
        # Same widget signature as default LoadImage
        return {"required":
                    {"image": (sorted(files), {"image_upload": True, "tooltip": "Image file from the ComfyUI input folder, loaded as 8-bit sRGB (0-1). If '<name>_radmask.png' sits beside it, that file's alpha is the mask as-is; otherwise the image's own alpha is inverted, as in Load Image."})},
                }

    CATEGORY = "FXTD STUDIOS/Radiance/◎ VFX"
    SEARCH_ALIASES = ["◎ Radiance image", "◎ Radiance mask", "◎ Radiance mask editor", "load image mask"]

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "mask")
    FUNCTION = "load_image"
    DESCRIPTION = (
        "Load an image with optional non-destructive mask override. "
        "If a companion '_radmask.png' file exists alongside the source image, "
        "its alpha channel is used as the mask instead of the image's own alpha. "
        "The Radiance mask editor frontend saves masks in this companion format."
    )

    def load_image(self, image):
        # 1. Load the primary image
        image_path = folder_paths.get_annotated_filepath(image)
        img = node_helpers.pillow(Image.open, image_path)

        # 2. Check for RadMask companion file
        #    Example: 'input/myimage.png' -> companion: 'input/myimage_radmask.png'
        filename, ext = os.path.splitext(image_path)
        companion_mask_path = f"{filename}_radmask.png"
        
        has_companion_mask = os.path.exists(companion_mask_path)
        if has_companion_mask:
            companion_img = node_helpers.pillow(Image.open, companion_mask_path)
            companion_img = node_helpers.pillow(ImageOps.exif_transpose, companion_img)

        output_images = []
        output_masks = []
        w, h = None, None

        for i in ImageSequence.Iterator(img):
            i = node_helpers.pillow(ImageOps.exif_transpose, i)

            if i.mode == 'I':
                i = i.point(lambda i: i * (1 / 255))
            image_rgb = i.convert("RGB")

            if len(output_images) == 0:
                w = image_rgb.size[0]
                h = image_rgb.size[1]

            if image_rgb.size[0] != w or image_rgb.size[1] != h:
                continue

            image_tensor = np.array(image_rgb).astype(np.float32) / 255.0
            image_tensor = torch.from_numpy(image_tensor)[None,]
            
            # --- Mask Resolution ---
            if has_companion_mask:
                # Use the companion _radmask.png's alpha channel (or grayscale if no alpha)
                c_img = companion_img.convert("RGBA")
                if c_img.size[0] != w or c_img.size[1] != h:
                    c_img = c_img.resize((w, h), resample=Image.BILINEAR)
                mask_np = np.array(c_img.getchannel('A')).astype(np.float32) / 255.0
                mask_tensor = torch.from_numpy(mask_np)
            else:
                # Fallback to default LoadImage behavior
                if 'A' in i.getbands():
                    mask_np = np.array(i.getchannel('A')).astype(np.float32) / 255.0
                    mask_tensor = 1. - torch.from_numpy(mask_np) # Default LoadImage inverts the alpha mask
                elif i.mode == 'P' and 'transparency' in i.info:
                    mask_np = np.array(i.convert('RGBA').getchannel('A')).astype(np.float32) / 255.0
                    mask_tensor = 1. - torch.from_numpy(mask_np)
                else:
                    # Previously torch.zeros((64,64)) — hardcoded size mismatched every
                    # real image and crashed torch.cat() on multi-frame inputs.
                    # 'I' (32-bit int) mode images have no alpha band; they also
                    # need a correctly-sized zeros mask so the shape is always (H, W).
                    mask_tensor = torch.zeros((h, w), dtype=torch.float32)
                    
            output_images.append(image_tensor)
            output_masks.append(mask_tensor.unsqueeze(0))

            if img.format == "MPO":
                break  # ignore all frames except the first one for MPO format

        if len(output_images) > 1:
            output_image = torch.cat(output_images, dim=0)
            output_mask = torch.cat(output_masks, dim=0)
        elif len(output_images) == 1:
            output_image = output_images[0]
            output_mask = output_masks[0]
        else:
            # MPO edge case: if the first (and only) frame was skipped due to a size
            # mismatch, output_images is empty. Previously this caused IndexError on
            # output_images[0]. Fall back to a 1×1 black image + zeros mask so the
            # node returns a valid tensor rather than crashing the entire queue.
            logger.warning(
                f"[RadianceLoadImageMask] No valid frames found in '{image}' — "
                "returning 1×1 fallback. Check image format or file integrity."
            )
            output_image = torch.zeros((1, 1, 1, 3), dtype=torch.float32)
            output_mask  = torch.zeros((1, 1, 1),    dtype=torch.float32)

        return (output_image, output_mask)

    @classmethod
    def IS_CHANGED(s, image):
        # Trigger change if originally image, raster mask, or vector metadata changes
        image_path = folder_paths.get_annotated_filepath(image)
        filename, _ = os.path.splitext(image_path)
        companion_mask_path = f"{filename}_radmask.png"
        companion_meta_path = f"{filename}_radmask_meta.json"

        # FIX 3: unguarded open() crashed the ComfyUI queue on missing/race-condition
        # paths. Return math.nan on any error so ComfyUI treats the node as always-changed
        # (safe fallback) instead of propagating an unhandled exception.
        import math
        try:
            m = hashlib.sha256()
            with open(image_path, 'rb') as f:
                m.update(f.read())
            if os.path.exists(companion_mask_path):
                with open(companion_mask_path, 'rb') as f:
                    m.update(f.read())
            if os.path.exists(companion_meta_path):
                with open(companion_meta_path, 'rb') as f:
                    m.update(f.read())
            return m.digest().hex()
        except Exception as e:
            logger.warning(f"[RadianceLoadImageMask] IS_CHANGED hash failed: {e} — treating as changed.")
            return math.nan

    @classmethod
    def VALIDATE_INPUTS(s, image):
        if not folder_paths.exists_annotated_filepath(image):
            return "Invalid image file: {}".format(image)
        return True

# FIX 1: Plain ASCII key — ◎ belongs only in DISPLAY_NAME_MAPPINGS.
NODE_CLASS_MAPPINGS = {
    "RadianceLoadImageMask": RadianceLoadImageMask
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RadianceLoadImageMask": "◎ Radiance Load Image"
}
