"""Release audit F2 and F3 (20 Nov 2026).

F2: DSINE's torch.hub code is pinned to one commit; unpinned, torch.hub ran
whatever hugoycj/DSINE-hub's main held on the day of the first run.

F3: RUDRA's lock runs on the CUDA device ComfyUI runs on (not always cuda:0),
never on MPS (no float64), and finishes on the CPU when the GPU is out of
memory instead of failing the frame.
"""
import ast
import re
import sys
import types
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]


def test_dsine_hub_is_pinned_to_a_commit():
    from radiance.nodes.vfx.multipass import core
    assert re.fullmatch(r"[0-9a-f]{40}", core.DSINE_HUB_REF)
    src = (ROOT / "nodes" / "vfx" / "multipass" / "core.py").read_text(encoding="utf-8")
    calls = [n for n in ast.walk(ast.parse(src)) if isinstance(n, ast.Call)
             and ast.unparse(n.func) == "torch.hub.load"]
    assert calls, "no torch.hub.load call found"
    for c in calls:
        first = ast.unparse(c.args[0])
        assert "DSINE_HUB_REF" in first, f"torch.hub.load({first}) is not pinned"
    assert "hugoycj_DSINE-hub_main" not in src


@pytest.fixture
def fake_comfy(monkeypatch):
    def install(device):
        mm = types.ModuleType("comfy.model_management")
        mm.get_torch_device = lambda: device
        comfy = sys.modules.get("comfy") or types.ModuleType("comfy")
        monkeypatch.setitem(sys.modules, "comfy", comfy)
        monkeypatch.setitem(sys.modules, "comfy.model_management", mm)
        monkeypatch.setattr(comfy, "model_management", mm, raising=False)
    return install


def test_the_lock_uses_comfyuis_cuda_device(fake_comfy):
    from radiance.model import rudra_aces2 as r
    fake_comfy(torch.device("cuda", 1))
    assert r._lock_device() == torch.device("cuda", 1)


def test_the_lock_never_uses_mps(fake_comfy, monkeypatch):
    from radiance.model import rudra_aces2 as r
    fake_comfy(torch.device("mps"))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert r._lock_device() == torch.device("cpu")


def test_an_out_of_memory_gpu_finishes_on_the_cpu(monkeypatch):
    from radiance.model import rudra_aces2 as r
    rng = np.random.default_rng(0)
    codes = rng.uniform(0.4, 0.6, (12, 10, 3))
    lo, hi = codes - 0.01, codes + 0.01
    expected = r._smoothest_in_bins(codes, lo, hi, 20)        # CPU here

    real_to = torch.Tensor.to

    def to(self, *a, **k):
        dev = a[0] if a else k.get("device")
        if isinstance(dev, torch.device) and dev.type == "cuda":
            raise RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")
        return real_to(self, *a, **k)

    monkeypatch.setattr(r, "_lock_device", lambda: torch.device("cuda", 0))
    monkeypatch.setattr(torch.Tensor, "to", to)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    got = r._smoothest_in_bins(codes, lo, hi, 20)
    np.testing.assert_allclose(got, expected)


def test_other_gpu_errors_are_not_swallowed(monkeypatch):
    from radiance.model import rudra_aces2 as r
    codes = np.full((4, 4, 3), 0.5)

    def to(self, *a, **k):
        raise RuntimeError("CUDA error: an illegal memory access was encountered")

    monkeypatch.setattr(r, "_lock_device", lambda: torch.device("cuda", 0))
    monkeypatch.setattr(torch.Tensor, "to", to)
    with pytest.raises(RuntimeError, match="illegal memory access"):
        r._smoothest_in_bins(codes, codes - 0.01, codes + 0.01, 5)
