"""Join EIT recordings read from several files into one recording."""

from __future__ import annotations

import os
import warnings
from collections.abc import Sequence
from functools import reduce
from typing import Any

import numpy as np

#: A pause between two files longer than this many frame intervals is
#: reported as a gap.
_GAP_TOLERANCE_FRAMES = 0.5


def eit_file_paths(path: Any) -> list[str]:
    """The file paths in `path`: one path, or a list or tuple of paths."""

    if isinstance(path, list | tuple):
        if not path:
            raise ValueError("Give at least one EIT file to load.")
        return [os.fspath(item) for item in path]
    return [os.fspath(path)]


def join_eit_sequences(sequences: Sequence[Any], paths: Sequence[str]) -> Any:
    """Join eitprocessing Sequences read from consecutive files, in the given order.

    The frame times of each file must come after those of the file before it.
    A pause between two files is kept in the time axis and reported with a
    warning; filters and breath detection then run across the pause.

    Args:
        sequences: One eitprocessing Sequence per file, in recording order.
        paths: The file of each Sequence, for messages.

    Returns:
        One eitprocessing Sequence holding all frames.

    Raises:
        ValueError: If a file starts before the one before it ends (files out
            of order or overlapping), or if two files cannot be joined
            (different device, image size or sampling rate).
    """

    for (before, before_path), (after, after_path) in zip(
        zip(sequences, paths), zip(sequences[1:], paths[1:]), strict=False
    ):
        before_time = np.asarray(before.eit_data["raw"].time, dtype=float)
        after_time = np.asarray(after.eit_data["raw"].time, dtype=float)
        before_end = float(before_time[-1])
        after_start = float(after_time[0])
        if after_start <= before_end:
            raise ValueError(
                f"EIT file {after_path!r} starts (t = {after_start:.3f} s) "
                f"before {before_path!r} ends (t = {before_end:.3f} s). List "
                "the files in recording order, without overlap."
            )
        frame_interval = (
            float(np.median(np.diff(before_time))) if before_time.size > 1 else 0.0
        )
        gap = after_start - before_end - frame_interval
        if frame_interval > 0 and gap > _GAP_TOLERANCE_FRAMES * frame_interval:
            warnings.warn(
                f"There is a gap of {gap:.3f} s between EIT files "
                f"{before_path!r} and {after_path!r}. The time axis keeps the "
                "gap; filters and breath detection run across it.",
                UserWarning,
                stacklevel=3,
            )

    from eitprocessing.datahandling.mixins.equality import EquivalenceError

    try:
        return reduce(lambda joined, nxt: joined.concatenate(nxt), sequences)
    except EquivalenceError as exc:
        raise ValueError(
            "These EIT files cannot be joined into one recording: they differ "
            f"in device, image size or sampling rate ({exc})."
        ) from exc
