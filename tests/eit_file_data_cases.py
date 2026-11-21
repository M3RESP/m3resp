"""Small EIT recordings for tests that load EIT data without a real file.

``small_eit_file_data`` gives the data of a made-up EIT file, and
``stand_in_eit_loader`` is a custom EIT loader that returns it for any path:
``M3Session(eit_loader=stand_in_eit_loader)``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from m3resp.data import Signal
from m3resp.eit.io import EITFileData


def small_eit_file_data(
    path: Any = "subject.eit",
    vendor: str = "sentec",
    *,
    n_frames: int = 3,
    sample_frequency: float = 1.0,
    signals: dict[str, Signal] | None = None,
) -> EITFileData:
    """Return the data of a made-up EIT file with one pixel.

    Frame ``i`` holds the value ``i + 1``. Time starts at 0 s.

    Args:
        path (Any): The file path the data names.
        vendor (str): The device maker.
        n_frames (int): Number of frames.
        sample_frequency (float): Frames per second, in Hz.
        signals (dict[str, Signal] | None): Waveforms of the device, keyed
            by their labels.

    Returns:
        EITFileData: The data.
    """

    time = np.arange(n_frames) / sample_frequency
    return EITFileData(
        vendor=vendor,
        paths=(Path(str(path)),),
        sample_frequency=sample_frequency,
        pixel_impedance=Signal(
            values=np.arange(1.0, n_frames + 1.0)[:, None, None],
            time=time,
            sample_frequency=sample_frequency,
            name="raw",
            modality="eit",
            category="impedance",
            channel="pixel_impedance",
            processing_state="raw",
            source=vendor,
        ),
        signals=dict(signals or {}),
    )


def stand_in_eit_loader(
    path: Any, vendor: str | None = None, **kwargs: Any
) -> EITFileData:
    """A custom EIT loader that reads no file and returns ``small_eit_file_data``.

    Args:
        path (Any): The path given to the loader; the first path of a list.
        vendor (str | None): The device maker; None gives ``"sentec"``.
        **kwargs (Any): Not used.

    Returns:
        EITFileData: The data of a made-up file at ``path``.
    """

    first = path[0] if isinstance(path, list | tuple) else path
    return small_eit_file_data(first, vendor or "sentec")
