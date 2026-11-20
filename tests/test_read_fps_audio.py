"""Read hands Write the clip's frame rate and sound.

Read used to return image, mask and info only. Write's fps 0 means auto, but
an IMAGE tensor carries no rate, so auto fell back to 24: a 25 or 30 fps clip
through Read -> Write came out at 24 fps, longer than the source and silent.
Read now outputs `fps` and `audio`; the shipped workflows wire both to Write.
"""
import json
import pathlib
import subprocess

import pytest

torch = pytest.importorskip("torch")

from radiance.core.ffmpeg import ffmpeg_exe  # noqa: E402

needs_ffmpeg = pytest.mark.skipif(not ffmpeg_exe(), reason="ffmpeg is required to encode the fixtures")

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _clip(dst: pathlib.Path, fps: int, seconds: float, audio: bool) -> pathlib.Path:
    cmd = [ffmpeg_exe(), "-y", "-v", "error", "-f", "lavfi", "-i",
           f"testsrc=size=64x48:rate={fps}:duration={seconds}"]
    if audio:
        cmd += ["-f", "lavfi", "-i", f"sine=frequency=440:sample_rate=44100:duration={seconds}",
                "-c:a", "aac", "-shortest"]
    cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p", str(dst)]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0 or not dst.exists():
        pytest.skip("this ffmpeg cannot make the fixture: " + proc.stderr.decode("utf-8", "replace")[-200:])
    return dst


def _probe(path: pathlib.Path) -> str:
    return subprocess.run([ffmpeg_exe(), "-hide_banner", "-i", str(path)],
                          capture_output=True, text=True).stderr


@needs_ffmpeg
def test_a_25fps_clip_reads_as_25fps_with_its_sound(tmp_path):
    from radiance.nodes.io.write import RadianceRead

    src = _clip(tmp_path / "plate_25.mp4", 25, 2.0, audio=True)
    image, _mask, info, fps, audio = RadianceRead().read(path=str(src), start_frame=0)
    assert fps == pytest.approx(25.0)
    assert json.loads(info)["fps"] == pytest.approx(25.0)
    assert audio is not None and audio["sample_rate"] == 48000
    wav = audio["waveform"]
    assert wav.ndim == 3 and wav.shape[:2] == (1, 2)
    # the sound under the frames that were read: n frames at 25 fps
    seconds = image.shape[0] / 25.0
    assert abs(wav.shape[-1] / 48000 - seconds) < 0.1
    assert float(wav.abs().max()) > 0.05          # a tone (ffmpeg sine is 1/8 full scale), not silence


@needs_ffmpeg
def test_frame_step_keeps_the_duration(tmp_path):
    from radiance.nodes.io.write import RadianceRead

    src = _clip(tmp_path / "plate_30.mp4", 30, 1.0, audio=False)
    image, _m, _i, fps, audio = RadianceRead().read(path=str(src), start_frame=0, frame_step=2)
    assert fps == pytest.approx(15.0)
    assert audio is None                           # a silent clip has no audio output
    assert image.shape[0] == 15


def test_stills_and_empty_reads_say_24(tmp_path):
    from radiance.nodes.io.write import RadianceRead
    from PIL import Image

    Image.new("RGB", (8, 8), (128, 64, 32)).save(tmp_path / "still.png")
    *_, fps, audio = RadianceRead().read(path=str(tmp_path / "still.png"))
    assert fps == 24.0 and audio is None
    *_, fps, audio = RadianceRead().read(path="")
    assert fps == 24.0 and audio is None


@needs_ffmpeg
def test_read_to_write_keeps_the_rate_and_the_sound(tmp_path):
    from radiance.nodes.io.write import RadianceRead, RadianceWrite

    src = _clip(tmp_path / "plate_25.mp4", 25, 2.0, audio=True)
    image, mask, _info, fps, audio = RadianceRead().read(path=str(src), start_frame=0, color_space="sRGB")
    RadianceWrite().write(image=image, output_path=str(tmp_path / "out"), format="VID │ MP4 (H.264)",
                          filename="shot", color_space="Rec.709 (BT.1886)", fps=fps, audio=audio)
    out = next((tmp_path / "out").glob("shot_v0001*.mp4"))
    desc = _probe(out)
    assert "25 fps" in desc
    assert "Audio:" in desc


def test_the_shipped_workflows_wire_fps_and_audio_from_read_to_write():
    """Every Write fed by a Read in a shipped workflow takes its fps and audio
    from that Read, in the JSON and in the library card."""
    import io
    import zipfile

    def wired(graph):
        nodes = {n["id"]: n for n in graph["nodes"]}
        links = {lk[0]: lk for lk in graph["links"]}
        found = []
        for n in graph["nodes"]:
            if n.get("type") != "RadianceWrite" or n.get("mode", 0) == 4:
                continue
            ins = {i["name"]: i for i in n.get("inputs", [])}
            src_fps = links.get(ins.get("fps", {}).get("link"))
            src_audio = links.get(ins.get("audio", {}).get("link"))
            if src_fps or src_audio:
                assert src_fps and src_audio, f"Write {n['id']}: fps and audio come together"
                assert nodes[src_fps[1]]["type"] == "RadianceRead" and src_fps[2] == 3
                assert nodes[src_audio[1]]["type"] == "RadianceRead" and src_audio[2] == 4
                found.append(n["id"])
        return found

    checked = 0
    for name in ("sdr_to_hdr", "sdr_remaster_ai_fill", "hdr_delivery", "shot_look_match"):
        graph = json.loads((ROOT / "workflows" / f"{name}.json").read_text(encoding="utf-8"))
        assert wired(graph), f"{name}: no Write takes fps and audio from Read"
        card = (ROOT / "workflows" / "official" / "Local" / f"{name}.rad").read_bytes()
        with zipfile.ZipFile(io.BytesIO(card[4:])) as z:
            assert json.loads(z.read("workflow.json")) == graph, f"{name}: the library card is stale"
        checked += 1
    assert checked == 4
