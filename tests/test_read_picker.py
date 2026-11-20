"""Read's file list shows images and videos, and uploads both (4.0).

Until 4.0 the list was marked "image_upload", so ComfyUI's node UI showed
images only and its upload took PNG, JPG and WebP: an MP4 or MOV in the input
folder never appeared.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from radiance.nodes.io import write as W

ROOT = Path(__file__).resolve().parents[1]


def test_the_list_is_not_an_image_only_upload():
    spec = W.RadianceRead.INPUT_TYPES()["required"]["browse"][1]
    assert "image_upload" not in spec and "video_upload" not in spec
    assert spec.get("radiance_upload") is True


def test_the_list_holds_images_and_videos(monkeypatch, tmp_path):
    for name in ("plate.mp4", "plate.MOV", "clip.mxf", "still.png", "master.exr", "scan.dpx", "notes.txt"):
        (tmp_path / name).write_bytes(b"x")
    monkeypatch.setattr(W, "_HAS_FOLDER_PATHS", True)
    monkeypatch.setattr(W._folder_paths, "get_input_directory", lambda: str(tmp_path), raising=False)
    files = W._get_input_files()
    assert files[0] == ""
    assert {"plate.mp4", "plate.MOV", "clip.mxf", "still.png", "master.exr", "scan.dpx"} <= set(files)
    assert "notes.txt" not in files


def test_the_upload_button_takes_images_and_video():
    js = (ROOT / "js" / "radiance_io.js").read_text(encoding="utf-8")
    assert 'READ_UPLOAD_ACCEPT = "image/*,video/*,' in js
    assert '"/upload/image"' in js                 # ComfyUI's upload stores any file in input/
    assert "onDragDrop" in js                      # dropping a file on the node uploads it


@pytest.mark.real_torch
def test_a_linear_exr_gets_a_tone_mapped_preview(tmp_path):
    OpenEXR = pytest.importorskip("OpenEXR")
    rgb = np.zeros((32, 48, 3), np.float32)
    rgb[..., 0] = np.linspace(0.0, 8.0, 48)[None, :]
    path = tmp_path / "hdr.exr"
    OpenEXR.File({"compression": OpenEXR.ZIP_COMPRESSION, "type": OpenEXR.scanlineimage}, {"RGB": rgb}).write(str(path))
    png = W._still_poster_png(str(path))
    assert png and png[:8] == b"\x89PNG\r\n\x1a\n"
    from io import BytesIO
    from PIL import Image
    img = np.asarray(Image.open(BytesIO(png)))
    assert img.shape[:2] == (32, 48)
    assert img[0, -1, 0] < 255 and img[0, -1, 0] > img[0, 24, 0]   # 8.0 is not clipped flat
