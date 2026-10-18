"""A list of IMAGE batches writes as one sequence, in order (package audit R5)."""

import numpy as np
import pytest
import torch
from PIL import Image

from radiance.io.writer import coerce_to_frames, write_frames


def test_a_list_of_batches_joins_along_the_frame_axis():
    arr, _ = coerce_to_frames([torch.full((1, 4, 4, 3), 0.1), torch.full((2, 4, 4, 3), 0.5)])
    assert arr.shape == (3, 4, 4, 3)
    assert np.allclose(arr[:, 0, 0, 0], [0.1, 0.5, 0.5])


def test_a_list_of_single_frames_gets_one_batch_axis():
    arr, _ = coerce_to_frames([torch.zeros(4, 4, 3), torch.ones(4, 4, 3)])
    assert arr.shape == (2, 4, 4, 3)


@pytest.mark.parametrize("data, match", [
    ([torch.zeros(1, 4, 4, 3), torch.zeros(1, 8, 4, 3)], "share a size"),
    ([torch.zeros(1, 4, 4, 3), torch.zeros(1, 4, 4, 4)], "share a size and channel count"),
    ([torch.zeros(4, 4)], "expected an IMAGE"),
    ([], "empty list"),
])
def test_lists_that_cannot_be_one_sequence_say_why(data, match):
    with pytest.raises(ValueError, match=match):
        coerce_to_frames(data)


def test_a_list_of_batches_writes_every_frame_in_order(tmp_path):
    batches = [torch.full((1, 4, 4, 3), 0.1), torch.full((2, 4, 4, 3), 0.8)]
    _, count = write_frames(image=batches, output_path=str(tmp_path), format="SEQ │ PNG (8-bit)",
                            filename="L", version=1, color_space="Linear (pass-through)")
    assert count == 3
    frames = sorted(tmp_path.rglob("*.png"))
    assert [int(np.asarray(Image.open(f))[0, 0, 0]) for f in frames] == [26, 204, 204]
