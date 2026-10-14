"""Safe frame writes for the DCC hand-off nodes.

The DCC Bridge, Nuke Send and DaVinci Send nodes write numbered frames straight
into a user folder. Two things went wrong there:

* a frame was written in place, so a cancelled or failed write truncated the
  frame that was already on disk;
* a shorter second export left the higher-numbered frames of the first one in
  the folder, and a ``name.####.exr`` or ``name.%04d.exr`` import then read them
  as part of the new sequence.

``write_atomic`` fixes the first. ``stale_sequence_frames`` finds the second so
the node can say so in its status line. It never deletes anything.
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Callable, Iterable, List


def write_atomic(path: "str | os.PathLike", write_fn: Callable[[Path], Any]) -> Any:
    """Run ``write_fn(tmp_path)`` and move the result over ``path`` only on success.

    The temporary file keeps the real extension (writers such as OpenCV choose
    the format from it) and is hidden, so a sequence glob never matches it.
    Whatever ``write_fn`` returns is returned (``_save_exr`` returns a note when
    it had to fall back to OpenCV).
    """
    final = Path(path)
    tmp = final.with_name(f".{final.stem}.partial-{os.getpid()}{final.suffix}")
    try:
        result = write_fn(tmp)
        if not tmp.is_file() or tmp.stat().st_size == 0:
            raise RuntimeError(f"the writer produced no data for {final.name}")
        os.replace(tmp, final)
        return result
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass


def stale_sequence_frames(out_dir: "str | os.PathLike", prefix: str, sep: str,
                          ext: str, keep: Iterable[int]) -> List[int]:
    """Frame numbers of ``prefix<sep>NNNN<ext>`` files in ``out_dir`` that are not in ``keep``."""
    wanted = set(int(k) for k in keep)
    pat = re.compile(re.escape(prefix) + re.escape(sep) + r"(\d+)" + re.escape(ext) + r"$")
    found = []
    try:
        names = os.listdir(out_dir)
    except OSError:
        return []
    for name in names:
        m = pat.match(name)
        if m and int(m.group(1)) not in wanted:
            found.append(int(m.group(1)))
    return sorted(found)


def stale_warning(stale: List[int], prefix: str) -> str:
    """One sentence for a status line, or ``""`` when there is nothing stale."""
    if not stale:
        return ""
    return (f"WARNING: {len(stale)} frame(s) of an earlier '{prefix}' export remain in the folder "
            f"({stale[0]}-{stale[-1]}); a sequence import will read them as part of this one.")
