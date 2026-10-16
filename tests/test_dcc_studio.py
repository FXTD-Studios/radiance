"""
Tests for ComfyUI DCC/MCP integrations and Nuke/Resolve sending nodes.

Covers:
  - NukeConnector wire protocol framing (RCMD magic header, version, length).
  - NukeConnector security sanitization (identifier validation, string escaping).
  - NukeConnector high-level APIs (ping, load_exr, set_frame, get_info).
  - RadianceNukeSend (single frames, image sequences, Nuke push integration).
  - RadianceDaVinciSend (8bit PNG, 16bit TIFF, EXR folder drop exports).
  - RadianceMCP (Export mode, source loading, path resolution regex fix).
"""
import sys
import os
import json
import struct
import re
from pathlib import Path
from unittest.mock import MagicMock, patch, ANY
import pytest
import torch
import types

# Ensure folder_paths is robustly stubbed to prevent mock leakage crashes
if "folder_paths" not in sys.modules or isinstance(sys.modules["folder_paths"], MagicMock):
    fp = types.ModuleType("folder_paths")
    fp.get_output_directory = lambda: "/tmp/comfy_output"
    fp.get_input_directory = lambda: "/tmp/comfy_input"
    fp.get_temp_directory = lambda: "/tmp/comfy_temp"
    fp.get_annotated_filepath = lambda name: name
    fp.exists_annotated_filepath = lambda name: True
    sys.modules["folder_paths"] = fp
else:
    fp = sys.modules["folder_paths"]
    if not hasattr(fp, "get_output_directory"): fp.get_output_directory = lambda: "/tmp/comfy_output"
    if not hasattr(fp, "get_input_directory"): fp.get_input_directory = lambda: "/tmp/comfy_input"
    if not hasattr(fp, "get_temp_directory"): fp.get_temp_directory = lambda: "/tmp/comfy_temp"
    if not hasattr(fp, "get_annotated_filepath"): fp.get_annotated_filepath = lambda name: name
    if not hasattr(fp, "exists_annotated_filepath"): fp.exists_annotated_filepath = lambda name: True

def _fake_writer(*args, **kwargs):
    """Stands in for _save_exr / _save_pil_image: the real ones write the file they are given.

    The nodes now write to a hidden temporary sibling and replace the final
    name only if the file exists, so a mock that writes nothing no longer
    passes for a writer.
    """
    Path(args[1]).write_bytes(b"x")


from radiance.tools.nuke_connector import NukeConnector, validate_nuke_identifier, _sanitize_nuke_string
from radiance.nodes.pipeline.dcc import RadianceMCP, _push_to_nuke
from radiance.nodes.pipeline.studio_integrations import RadianceNukeSend, RadianceDaVinciSend

# Only the tests marked @pytest.mark.real_torch below need genuine tensors; the
# rest run fine against conftest's stub. Opt out of the automatic module-level
# skip so they keep running on the no-torch CI matrix.
RADIANCE_TORCH_GATED = True


@pytest.fixture(autouse=True)
def _token_home(tmp_path, monkeypatch):
    """The shared DCC token file lives in ~/.radiance; keep it out of the real home."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))


# ─────────────────────────────────────────────────────────────────────────────
#  1. NukeConnector & Security Sanitization Tests
# ─────────────────────────────────────────────────────────────────────────────

def test_sanitize_nuke_string():
    """Verify that potentially dangerous characters are escaped or removed."""
    dirty_path = "D:\\renders\\shot_01'\"\\\n\r\t\x00_test"
    clean_path = _sanitize_nuke_string(dirty_path)
    # Backslashes become forward slashes, single/double quotes and control chars removed
    assert clean_path == "D:/renders/shot_01/_test"

def test_validate_nuke_identifier_valid():
    """Safe identifiers should pass unchanged."""
    assert validate_nuke_identifier("Read1") == "Read1"
    assert validate_nuke_identifier("Radiance_Stream-AOV") == "Radiance_Stream-AOV"

def test_validate_nuke_identifier_invalid():
    """Unsafe identifiers must raise ValueError."""
    with pytest.raises(ValueError, match="Invalid identifier"):
        validate_nuke_identifier("Read1; import os; os.system('calc')")
    
    with pytest.raises(ValueError, match="Empty identifier"):
        validate_nuke_identifier("   ")
        
    with pytest.raises(ValueError, match="too long"):
        validate_nuke_identifier("a" * 130)

@patch("socket.socket")
def test_nuke_connector_ping_success(mock_socket_cls):
    """Verify ping returns True when Nuke responds with PONG."""
    mock_sock = MagicMock()
    mock_socket_cls.return_value = mock_sock
    
    # Simulate a successful wire protocol response
    end_marker = b"\n__RADIANCE_END__\n"
    mock_sock.recv.side_effect = [b"RADIANCE_PONG" + end_marker, b""]
    
    conn = NukeConnector(host="127.0.0.1", port=1986)
    success, reply = conn.ping()
    
    assert success is True
    assert "PONG" in reply

@patch("socket.socket")
def test_nuke_connector_load_exr(mock_socket_cls):
    """Verify correct python command formatting and framing for EXR loading."""
    mock_sock = MagicMock()
    mock_socket_cls.return_value = mock_sock
    
    end_marker = b"\n__RADIANCE_END__\n"
    mock_sock.recv.side_effect = [b"OK" + end_marker, b""]
    
    conn = NukeConnector(host="127.0.0.1", port=1986)
    success, reply = conn.load_exr(
        filepath="D:/renders/frame_####.exr",
        node_name="MyReadNode",
        first_frame=1001,
        last_frame=1010,
        current_frame=1001,
    )
    
    assert success is True
    # Verify magic bytes and headers in sent data
    args, kwargs = mock_sock.sendall.call_args
    sent_bytes = args[0]
    assert sent_bytes.startswith(b"RCMD")  # Magic
    assert struct.unpack("<B", sent_bytes[4:5])[0] == 3  # protocol 3, signed with the shared token

    # Verify the structured command carries the parameters
    payload = sent_bytes[65:].decode("utf-8")
    assert "MyReadNode" in payload
    assert "D:/renders/frame_####.exr" in payload
    assert "1001" in payload
    assert "1010" in payload

# ─────────────────────────────────────────────────────────────────────────────
#  2. RadianceNukeSend Node Tests
# ─────────────────────────────────────────────────────────────────────────────

@patch("radiance.nodes.pipeline.studio_integrations._save_exr", side_effect=_fake_writer)
@patch("radiance.tools.nuke_connector.NukeConnector.load_exr")
@pytest.mark.real_torch
def test_nuke_send_single_frame(mock_load_exr, mock_save_exr, tmp_path):
    """Verify RadianceNukeSend single-frame export and .nk snippet creation."""
    mock_load_exr.return_value = (True, "loaded")
    
    # Setup node and mock inputs
    node = RadianceNukeSend()
    fake_image = torch.ones(1, 64, 64, 3)
    folder = str(tmp_path / "nuke_out")
    
    status, render_path = node.run(
        image=fake_image,
        nuke_folder=folder,
        filename="single_test",
        frame_start=1001,
        push_to_nuke=True,
    )
    
    # Assert output directories and files are recorded
    assert "EXR frame(s) + .nk" in status
    assert Path(folder).exists()
    assert (Path(folder) / "single_test.nk").exists()
    
    # Verify .nk file contents
    nk_content = (Path(folder) / "single_test.nk").read_text()
    assert "Read {" in nk_content
    assert "single_test.exr" in nk_content
    assert "first 1001" in nk_content
    
    # Verify save and push functions called
    mock_save_exr.assert_called_once()
    mock_load_exr.assert_called_once()

# ─────────────────────────────────────────────────────────────────────────────
#  3. RadianceDaVinciSend Node Tests
# ─────────────────────────────────────────────────────────────────────────────

@patch("radiance.nodes.pipeline.studio_integrations._save_exr", side_effect=_fake_writer)
@patch("radiance.nodes.pipeline.studio_integrations._save_pil_image", side_effect=_fake_writer)
@pytest.mark.real_torch
def test_davinci_send_formats(mock_save_pil, mock_save_exr, tmp_path):
    """Verify RadianceDaVinciSend successfully exports 8-bit PNG, 16-bit TIFF, and EXR."""
    node = RadianceDaVinciSend()
    fake_image = torch.ones(1, 64, 64, 3)
    folder = str(tmp_path / "resolve_out")
    
    # 1. Test 8-bit PNG
    node.run(fake_image, folder, "test_png", "8bit", 1001)
    assert mock_save_pil.call_args.args[1].suffix == ".png"
    assert mock_save_pil.call_args.args[2] == "8-bit PNG"
    assert (Path(folder) / "test_png.png").is_file()
    
    # 2. Test 16-bit TIFF
    node.run(fake_image, folder, "test_tif", "16bit", 1001)
    assert mock_save_pil.call_args.args[1].suffix == ".tif"
    assert mock_save_pil.call_args.args[2] == "16-bit TIFF"
    assert (Path(folder) / "test_tif.tif").is_file()
    
    # 3. Test EXR
    node.run(fake_image, folder, "test_exr", "EXR", 1001)
    assert mock_save_exr.call_args.args[1].suffix == ".exr"
    assert mock_save_exr.call_args.kwargs == {"half": True}
    assert (Path(folder) / "test_exr.exr").is_file()

# ─────────────────────────────────────────────────────────────────────────────
#  4. DCC Bridge (RadianceMCP) & Sequence Path Pattern Regex Fix Tests
# ─────────────────────────────────────────────────────────────────────────────

def test_mcp_push_to_nuke_sequence_regex_fix():
    """Verify both dot and underscore separated sequence paths resolve to Nuke's '####' format."""
    # Underscore style sequence (DCC Bridge node default)
    written_underscore = ["C:/renders/frame_1001.exr", "C:/renders/frame_1002.exr"]
    with patch("radiance.tools.nuke_connector.NukeConnector.load_exr") as mock_load:
        mock_load.return_value = (True, "loaded")
        _push_to_nuke(written_underscore, "frame", 1001, 1002)
        
        args, kwargs = mock_load.call_args
        assert kwargs["filepath"] == "C:/renders/frame_####.exr"
        
    # Dot style sequence (RadianceNukeSend default)
    written_dot = ["C:/renders/frame.1001.exr", "C:/renders/frame.1002.exr"]
    with patch("radiance.tools.nuke_connector.NukeConnector.load_exr") as mock_load:
        mock_load.return_value = (True, "loaded")
        _push_to_nuke(written_dot, "frame", 1001, 1002)
        
        args, kwargs = mock_load.call_args
        assert kwargs["filepath"] == "C:/renders/frame.####.exr"

@patch("radiance.nodes.pipeline.dcc._save_exr", side_effect=_fake_writer)
@patch("radiance.tools.nuke_connector.NukeConnector.load_exr")
@pytest.mark.real_torch
def test_mcp_export_frames_flow(mock_load_exr, mock_save_exr, tmp_path):
    """Verify full frame export flow of RadianceMCP."""
    mock_load_exr.return_value = (True, "loaded")
    node = RadianceMCP()
    
    fake_images = torch.ones(3, 32, 32, 3)
    out_path = str(tmp_path / "mcp_render")
    
    status, render_path = node.run(
        mode="Export Frames",
        source="Images",
        target="Nuke",
        output_path=out_path,
        format="EXR (16-bit half)",
        images=fake_images,
        frame_start=1001,
        filename_prefix="myframe",
    )
    
    assert "3 EXR frames" in status
    assert "nuke: OK" in status
    assert mock_save_exr.call_count == 3
    mock_load_exr.assert_called_once()


@patch("socket.socket")
def test_nuke_connector_v3_signature(mock_socket_cls, monkeypatch):
    """NukeConnector sends protocol 3: HMAC over timestamp, nonce and payload (E-11)."""
    monkeypatch.setenv("RADIANCE_DCC_AUTH_TOKEN", "super-secret-key-123")

    mock_sock = MagicMock()
    mock_socket_cls.return_value = mock_sock

    end_marker = b"\n__RADIANCE_END__\n"
    mock_sock.recv.side_effect = [b"OK" + end_marker, b""]

    conn = NukeConnector(host="127.0.0.1", port=1986)

    cmd = "import nuke; print('hello')"
    success, reply = conn.send_command(cmd)

    assert success is True
    sent_bytes = mock_sock.sendall.call_args[0][0]

    assert sent_bytes[:5] == b"RCMD\x03"
    sig, ts_bytes, nonce = sent_bytes[5:37], sent_bytes[37:45], sent_bytes[45:61]
    import hmac
    import hashlib
    import time
    expected_sig = hmac.new(b"super-secret-key-123", b"RCMD3" + ts_bytes + nonce + cmd.encode(), hashlib.sha256).digest()
    assert sig == expected_sig
    # The protocol-2 construction (command only, replayable) must not match.
    assert sig != hmac.new(b"super-secret-key-123", cmd.encode("utf-8"), hashlib.sha256).digest()
    assert abs(struct.unpack("<Q", ts_bytes)[0] / 1000 - time.time()) < 30
    assert len(set(nonce)) > 1
    cmd_len = struct.unpack("<I", sent_bytes[61:65])[0]
    assert cmd_len == len(cmd)
    assert sent_bytes[65:].decode("utf-8") == cmd


@patch("socket.socket")
def test_nuke_connector_signs_with_the_shared_token_file(mock_socket_cls, monkeypatch, tmp_path):
    """3.5.0: with no RADIANCE_DCC_AUTH_TOKEN both sides use ~/.radiance/dcc_token,
    so commands are signed out of the box. Unsigned (v1) commands were refused by
    the listener, which made push_to_nuke fail on every default install."""
    import hmac
    import hashlib
    monkeypatch.delenv("RADIANCE_DCC_AUTH_TOKEN", raising=False)

    mock_sock = MagicMock()
    mock_socket_cls.return_value = mock_sock
    mock_sock.recv.side_effect = [b"OK\n__RADIANCE_END__\n", b""]

    conn = NukeConnector(host="127.0.0.1", port=1986)
    cmd = "print('hello')"
    success, reply = conn.send_command(cmd)
    assert success is True

    token = (tmp_path / ".radiance" / "dcc_token").read_text().strip()
    assert len(token) == 64
    sent_bytes = mock_sock.sendall.call_args[0][0]
    assert sent_bytes[:5] == b"RCMD\x03"
    ts_bytes, nonce = sent_bytes[37:45], sent_bytes[45:61]
    assert sent_bytes[5:37] == hmac.new(token.encode(), b"RCMD3" + ts_bytes + nonce + cmd.encode(), hashlib.sha256).digest()
    assert struct.unpack("<I", sent_bytes[61:65])[0] == len(cmd)


@pytest.mark.real_torch
def test_dcc_bridge_exports_an_image_sequence(tmp_path):
    # The sequence source passed the colour space "Linear (none)", which the
    # reader does not know, so source=Sequence always ended in
    # "Error: failed to read sequence". It now reads the files untransformed.
    import numpy as np
    from PIL import Image
    from radiance.nodes.pipeline.dcc import RadianceMCP
    src = tmp_path / "src"
    src.mkdir()
    for f in (1001, 1002, 1003):
        Image.fromarray(np.full((8, 8, 3), 60 * (f - 1000), np.uint8)).save(src / f"plate.{f}.png")
    out = tmp_path / "out"
    status, _ = RadianceMCP().run(mode="Export Frames", source="Sequence", output_path=str(out),
                                  sequence_path=str(src / "plate.%04d.png"), frame_start=1001)
    assert not status.startswith("Error"), status
    assert len(list(out.glob("*.exr"))) == 3
