"""Read one EIT file with the reader of its device maker.

---------------------------------------------------------------------------
Provenance
----------
Portions of this module are derived from eitprocessing.

    Source:     https://github.com/EIT-ALIVE/eitprocessing
    Revision:   m3resp-integration (6d6da6a825a08201ad01c0a7b68f4de07d4501f3),
                version 1.8.7
    Original:   eitprocessing/datahandling/loading/__init__.py::
                load_eit_data, _check_first_frame, _ensure_vendor
    Copyright:  Copyright (c) Netherlands eScience Center and Erasmus MC
    License:    Apache License, Version 2.0

Modified for M3RESP:
    - `read_eit_file` holds the part of `load_eit_data` for one file: the
      vendor check, the `first_frame` check, the check that the file
      exists, and the call to the reader of the vendor. It returns an
      `EITFileData`. Several files are joined by the caller.
    - The `first_frame` check raises only for a negative value. A check
      that `max_frames` is at least 1 is added, with the same error for
      all three vendors.
    - The vendor is a text: ``"draeger"``, ``"sentec"`` or ``"timpel"``
      (or an object whose ``value`` is one of these, such as the
      eitprocessing ``Vendor``).
    - Bug fix: ``"simulated"`` gives the NotImplementedError of an unknown
      vendor. eitprocessing 1.8.7 accepts the name and then fails with a
      KeyError, because it has no reader for simulated data.
    - Docstrings are reworded.

The original copyright and license notices are retained per Apache-2.0 §4.
Full attribution notice: see top-level NOTICE.md.
---------------------------------------------------------------------------
"""

from __future__ import annotations

import errno
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

from m3resp.eit.io.draeger import read_draeger
from m3resp.eit.io.file_data import EITFileData
from m3resp.eit.io.sentec import read_sentec
from m3resp.eit.io.timpel import read_timpel

#: The reader of each device maker.
VENDOR_READERS: dict[str, Callable[..., EITFileData]] = {
    "draeger": read_draeger,
    "timpel": read_timpel,
    "sentec": read_sentec,
}


def read_eit_file(
    path: str | os.PathLike[str],
    vendor: Any,
    *,
    sample_frequency: float | None = None,
    first_frame: int | None = 0,
    max_frames: int | None = None,
) -> EITFileData:
    """Read one EIT file of a Draeger, Sentec or Timpel device.

    Args:
        path (str | os.PathLike[str]): The file.
        vendor (Any): The device maker: ``"draeger"``, ``"sentec"`` or
            ``"timpel"``, in lower case.
        sample_frequency (float | None): The sample frequency in Hz.
            ``None`` uses the value of the reader: the estimate from the
            frame times for Draeger, the value in the file (or 50.2 Hz) for
            Sentec, and 50 Hz for Timpel.
        first_frame (int | None): The index of the first frame to read,
            counted from 0. ``None`` means 0.
        max_frames (int | None): The largest number of frames to read.
            ``None`` reads up to the end of the file.

    Returns:
        EITFileData: The data of the file, as read by the vendor reader.

    Raises:
        NotImplementedError: If there is no reader for ``vendor``.
        ValueError: If ``first_frame`` is negative, or ``max_frames`` is 0
            or negative. The vendor readers raise more errors, see
            ``read_draeger``, ``read_sentec`` and ``read_timpel``.
        FileNotFoundError: If the file does not exist.
    """
    reader = _reader_for(vendor)
    first_frame = _check_first_frame(first_frame)
    _check_max_frames(max_frames)
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), str(path))
    return reader(
        path,
        sample_frequency=sample_frequency,
        first_frame=first_frame,
        max_frames=max_frames,
    )


def _check_first_frame(first_frame: int | None) -> int:
    """Return ``first_frame``; ``None`` gives 0.

    Raises:
        ValueError: If ``first_frame`` is negative.
    """
    if first_frame is None:
        return 0
    if first_frame < 0:
        msg = f"`first_frame` can not be negative, but was given as {first_frame}"
        raise ValueError(msg)
    return first_frame


def _check_max_frames(max_frames: int | None) -> None:
    """Check that ``max_frames`` is ``None`` or at least 1.

    Raises:
        ValueError: If ``max_frames`` is 0 or negative.
    """
    if max_frames is not None and max_frames < 1:
        msg = f"`max_frames` must be at least 1, but was given as {max_frames}"
        raise ValueError(msg)


def _reader_for(vendor: Any) -> Callable[..., EITFileData]:
    """Return the reader of ``vendor``.

    Raises:
        NotImplementedError: If there is no reader for ``vendor``.
    """
    name = getattr(vendor, "value", vendor)
    try:
        return VENDOR_READERS[name]
    except (KeyError, TypeError) as e:
        msg = f"No loading method for {vendor} exists."
        raise NotImplementedError(msg) from e
