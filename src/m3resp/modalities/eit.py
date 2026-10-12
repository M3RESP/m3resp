"""EIT modality containers."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

from m3resp.modalities.time_window import TimeWindow


@dataclass
class EITRecording:
    """Loaded EIT recording with source metadata.

    ``path`` is the (first) file read; ``paths`` lists every file when
    several files were joined into this recording, in recording order.
    """

    data: Any
    path: Path
    vendor: str | None = None
    raw: Any = None
    global_impedance: Any = None
    paths: tuple[Path, ...] = ()

    def __post_init__(self) -> None:
        if not self.paths:
            self.paths = (self.path,)


EITLoaderChoice = Literal["auto", "eitprocessing", "custom"]


def load(
    path: str | Path | Sequence[str | Path],
    vendor: str | None = None,
    *,
    adapter: Any = None,
    loader: EITLoaderChoice = "auto",
    **kwargs: Any,
) -> EITRecording:
    """Load an EIT recording through the EIT adapter.

    ``loader`` picks the reader: ``"eitprocessing"`` reads a Draeger, Sentec
    or Timpel file (``vendor`` required), ``"custom"`` uses the loader given
    to the adapter, and ``"auto"`` uses the custom loader when there is one,
    else eitprocessing.

    ``path`` may be a list of files recorded one after the other; they are
    read as one recording. A custom loader receives the list as given.
    """

    from m3resp.adapters.eitprocessing_adapter import EITProcessingAdapter

    eit_adapter = adapter or EITProcessingAdapter()
    if isinstance(path, str | Path):
        paths: tuple[Path, ...] = (Path(path),)
        source: Any = str(path)
    else:
        if not path:
            raise ValueError("Give at least one EIT file to load.")
        paths = tuple(Path(item) for item in path)
        source = [str(item) for item in paths]
    if loader == "eitprocessing":
        sequence = eit_adapter.load_with_eitprocessing(source, vendor=vendor, **kwargs)
    elif loader == "custom":
        if vendor is not None:
            raise ValueError(
                "A custom EIT loader takes no vendor; pass reading options "
                "for the loader as keyword arguments."
            )
        sequence = eit_adapter.load_with_custom_loader(source, **kwargs)
    elif loader == "auto":
        sequence = eit_adapter.load(source, vendor=vendor, **kwargs)
    else:
        raise ValueError(
            f"loader must be 'auto', 'eitprocessing' or 'custom'; got {loader!r}."
        )
    raw = None
    global_impedance = None
    if hasattr(sequence, "eit_data"):
        raw = eit_adapter.get_raw_eit(sequence)
    if hasattr(sequence, "continuous_data") and raw is not None:
        global_impedance = eit_adapter.get_global_impedance(sequence)
    return EITRecording(
        data=sequence,
        path=paths[0],
        vendor=vendor,
        raw=raw,
        global_impedance=global_impedance,
        paths=paths,
    )


def frame_window(
    recording: EITRecording, start_seconds: float, end_seconds: float | None
) -> TimeWindow:
    """The frames of an EIT recording between two times.

    Times are in seconds from the recording's first frame, not the time of
    day stored in the file; ``end_seconds`` None means up to the end. The
    window runs from the first frame at or after the start up to, but not
    including, the first frame at or after the end. Frames are chosen by
    their time stamps, which in Draeger files are not perfectly even. Raises
    ValueError when the window does not lie inside the recording or holds
    no frame.
    """

    time = np.asarray(recording.data.time, dtype=float)
    if time.size < 2:
        raise ValueError("slice_eit needs an EIT recording with at least two frames.")
    frame_interval = float(np.median(np.diff(time)))
    duration = float(time[-1] - time[0]) + frame_interval
    end = duration if end_seconds is None else float(end_seconds)
    start = float(start_seconds)
    if not 0.0 <= start < end <= duration:
        raise ValueError(
            f"slice_eit: need 0 <= start < end <= {duration:.3f} s (the "
            f"recording's length); got start={start_seconds!r}, "
            f"end={end_seconds!r}."
        )

    start_index = int(np.searchsorted(time, time[0] + start, side="left"))
    end_index = (
        time.size
        if end_seconds is None
        else int(np.searchsorted(time, time[0] + end, side="left"))
    )
    if end_index <= start_index:
        raise ValueError(f"slice_eit: no EIT frame lies between {start} s and {end} s.")
    return TimeWindow(
        start_seconds=start,
        end_seconds=end,
        start_index=start_index,
        end_index=end_index,
        n_samples=time.size,
        shift_seconds=float(time[start_index] - time[0]),
    )


def keep_frames(
    recording: EITRecording, start_index: int, end_index: int, *, adapter: Any
) -> None:
    """Keep frames `start_index` up to (not including) `end_index` of an EIT
    recording, in place.

    The cutting is done by `eitprocessing` (`adapter.slice_sequence`, see
    `EITProcessingAdapter`): pixel data, global impedance, the pressure/flow
    channels and the vendor's markers are all cut to the same frames. Frame
    times keep their original values.
    """

    sequence = adapter.slice_sequence(recording.data, start_index, end_index)
    recording.data = sequence
    recording.raw = adapter.get_raw_eit(sequence)
    recording.global_impedance = adapter.get_global_impedance(sequence)
