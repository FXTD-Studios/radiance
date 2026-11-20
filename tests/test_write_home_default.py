"""Write's default folder is "~/radiance_output" (4.0), expanded per user.

Before 4.0 the default was the author's absolute home path, saved into every
workflow made with a new Write node, so a shared workflow wrote to a folder
named after someone else's account.
"""
from __future__ import annotations

import glob
import os

import torch

from radiance.core.system.path_utils import resolve_output_path
from radiance.nodes.io.write import RadianceDigitalCinemaWrite, RadianceWrite


def test_the_default_names_no_one_s_home():
    for cls in (RadianceWrite, RadianceDigitalCinemaWrite):
        spec = cls.INPUT_TYPES()["required"]["output_path"]
        assert spec[1]["default"] == "~/radiance_output"


def test_tilde_is_expanded_and_other_paths_are_unchanged(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    assert resolve_output_path("~/radiance_output") == os.path.normpath(str(tmp_path / "radiance_output"))
    assert resolve_output_path("~") == os.path.normpath(str(tmp_path))
    assert resolve_output_path(str(tmp_path / "abs")) == os.path.normpath(str(tmp_path / "abs"))
    assert "~" in resolve_output_path("renders/~draft")        # only a leading ~ is a home folder


def test_a_render_with_the_default_lands_in_this_user_s_home(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    RadianceWrite().write(image=torch.rand(1, 16, 16, 3), output_path="~/radiance_output",
                          format="IMG │ PNG (8-bit)", filename="shot", overwrite=True)
    assert glob.glob(str(tmp_path / "radiance_output" / "shot*.png"))
    assert not os.path.isdir("~")
