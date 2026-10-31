"""EIT modality containers."""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np

from m3resp.data.event_data import EventData, IntervalData
from m3resp.data.signals import Signal
from m3resp.eit.io.file_data import EITFileData
from m3resp.modalities.time_window import TimeWindow


@dataclass
class EITRecording:
    """Loaded EIT recording with source metadata.

    ``paths`` lists the files read, in recording order: one file, or several
    files joined into this recording.

    The recording holds its data twice. ``data`` is the eitprocessing
    ``Sequence`` that the EIT steps read; ``raw`` and ``global_impedance``
    are its raw images and raw global impedance. The m3resp fields hold the
    same data in m3resp types: ``pixel_impedance`` the images (frames, rows,
    columns), ``signals`` the global impedance and the waveforms of the
    device, ``events`` the marks of the device and ``intervals`` the breaths
    found by the device (Timpel). Their keys are the eitprocessing labels.
    They are empty for a custom loader whose result has no raw EIT data.
    """

    data: Any
    paths: tuple[Path, ...]
    vendor: str | None = None
    raw: Any = None
    global_impedance: Any = None
    pixel_impedance: Signal | None = None
    signals: dict[str, Signal] = field(default_factory=dict)
    events: dict[str, EventData] = field(default_factory=dict)
    intervals: dict[str, IntervalData] = field(default_factory=dict)
    sample_frequency: float | None = None
    _image_sum: tuple[Signal, Signal] | None = field(
        default=None, init=False, repr=False, compare=False
    )

    def set_native_data(self, file_data: EITFileData | None) -> None:
        """Fill the m3resp fields from ``file_data``; None empties them."""

        self.pixel_impedance = None if file_data is None else file_data.pixel_impedance
        self.signals = {} if file_data is None else dict(file_data.signals)
        self.events = {} if file_data is None else dict(file_data.events)
        self.intervals = {} if file_data is None else dict(file_data.intervals)
        self.sample_frequency = (
            None if file_data is None else file_data.sample_frequency
        )

    @property
    def raw_global_impedance(self) -> Signal | None:
        """The raw global impedance: one value per frame.

        This is the waveform the file gives (``signals["global_impedance_(raw)"]``,
        Draeger and Timpel). For a file without one (Sentec), it is the sum
        of all pixels of each raw image. The sum is computed once and kept
        until ``pixel_impedance`` is replaced by another Signal.

        Returns:
            Signal | None: The waveform, or None when the recording has no
                raw images and no recorded waveform.
        """

        signal = self.signals.get("global_impedance_(raw)")
        if signal is not None:
            return signal
        if self.pixel_impedance is None:
            return None
        if self._image_sum is None or self._image_sum[0] is not self.pixel_impedance:
            from m3resp.eit.processing import global_impedance

            summed = global_impedance(
                self.pixel_impedance, name="Global impedance (raw)"
            )
            self._image_sum = (self.pixel_impedance, summed)
        return self._image_sum[1]


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
    read as one recording. A custom loader receives the list as given. It
    may return an eitprocessing ``Sequence`` or an `EITFileData`.
    """

    from m3resp.adapters.eitprocessing_adapter import (
        EITProcessingAdapter,
        file_data_to_sequence,
        sequence_to_file_data,
    )

    eit_adapter = adapter or EITProcessingAdapter()
    if isinstance(path, str | Path):
        paths: tuple[Path, ...] = (Path(path),)
        source: Any = str(path)
    else:
        if not path:
            raise ValueError("Give at least one EIT file to load.")
        paths = tuple(Path(item) for item in path)
        source = [str(item) for item in paths]
    if loader not in ("auto", "eitprocessing", "custom"):
        raise ValueError(
            f"loader must be 'auto', 'eitprocessing' or 'custom'; got {loader!r}."
        )
    if loader == "custom" and vendor is not None:
        raise ValueError(
            "A custom EIT loader takes no vendor; pass reading options "
            "for the loader as keyword arguments."
        )
    use_custom = loader == "custom" or (
        loader == "auto" and getattr(eit_adapter, "has_custom_loader", False)
    )

    native: EITFileData | None
    if use_custom:
        if loader == "custom":
            loaded = eit_adapter.load_with_custom_loader(source, **kwargs)
        else:
            loaded = eit_adapter.load(source, vendor=vendor, **kwargs)
        if isinstance(loaded, EITFileData):
            native, sequence = loaded, file_data_to_sequence(loaded)
            vendor = vendor or loaded.vendor
        else:
            native, sequence = sequence_to_file_data(loaded), loaded
    elif hasattr(eit_adapter, "load_with_native_data"):
        sequence, native = eit_adapter.load_with_native_data(
            source, vendor=vendor, **kwargs
        )
    else:
        sequence = eit_adapter.load_with_eitprocessing(source, vendor=vendor, **kwargs)
        native = sequence_to_file_data(sequence)

    raw = None
    global_impedance = None
    if hasattr(sequence, "eit_data"):
        raw = eit_adapter.get_raw_eit(sequence)
    if hasattr(sequence, "continuous_data") and raw is not None:
        if (
            native is not None
            and "global_impedance_(raw)" not in sequence.continuous_data
        ):
            from m3resp.adapters.eitprocessing_adapter import (
                EITPROCESSING_FIELDS,
                add_to_collection,
                signal_to_continuous_data,
            )
            from m3resp.eit.processing import global_impedance as sum_pixels

            summed = sum_pixels(native.pixel_impedance, name="Global impedance (raw)")
            summed.metadata[EITPROCESSING_FIELDS] = {
                "label": "global_impedance_(raw)",
                "unit": "AU",
                "description": "Global impedance calculated from EIT pixel data.",
            }
            global_impedance = signal_to_continuous_data(summed)
            add_to_collection(sequence.continuous_data, global_impedance)
        else:
            global_impedance = eit_adapter.get_global_impedance(sequence)
    recording = EITRecording(
        data=sequence,
        paths=paths,
        vendor=vendor,
        raw=raw,
        global_impedance=global_impedance,
    )
    recording.set_native_data(native)
    return recording


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

    if recording.pixel_impedance is None:
        raise ValueError("slice_eit needs native EIT images and their time axis.")
    time = np.asarray(recording.pixel_impedance.time, dtype=float)
    from m3resp.processing.slicing import time_slice_indices

    # Raises a ValueError when the frame times are not finite or not in
    # increasing order.
    time_slice_indices(time)
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


def keep_frames(recording: EITRecording, start_index: int, end_index: int) -> None:
    """Keep frames `start_index` up to (not including) `end_index` of an EIT
    recording, in place.

    Images and waveforms are cut by frame index. Device marks and breaths
    are selected by the times of the kept frames. The eitprocessing Sequence
    is rebuilt for the processing steps that use it. Original times and
    collection keys are kept.

    Args:
        recording (EITRecording): The recording to cut. It changes in place.
        start_index (int): Index of the first frame to keep.
        end_index (int): Index after the last frame to keep.

    Raises:
        ValueError: If the recording has no raw images, no frames per second,
            no device maker (``vendor`` or the ``source`` of the images) or
            no file path, or if no frame is kept.
    """

    from m3resp.adapters.eitprocessing_adapter._slicing import sequence_from_sliced_data
    from m3resp.eit.processing.slicing import slice_file_data

    if recording.pixel_impedance is None or recording.sample_frequency is None:
        raise ValueError(
            "slice_eit needs native EIT images and their sampling frequency."
        )
    vendor = recording.vendor or recording.pixel_impedance.source
    if not vendor:
        raise ValueError(
            "slice_eit needs the EIT device maker: set EITRecording.vendor or "
            "the source of EITRecording.pixel_impedance."
        )
    if not recording.paths:
        raise ValueError(
            "slice_eit needs the EIT file path: EITRecording.paths is empty."
        )
    original = EITFileData(
        vendor=vendor,
        path=os.fspath(recording.paths[0]),
        sample_frequency=recording.sample_frequency,
        pixel_impedance=recording.pixel_impedance,
        signals=recording.signals,
        events=recording.events,
        intervals=recording.intervals,
    )
    native = slice_file_data(original, start_index, end_index)
    sequence = sequence_from_sliced_data(
        recording.data, original, native, start_index, end_index
    )
    recording.data = sequence
    recording.raw = sequence.eit_data["raw"]
    recording.global_impedance = sequence.continuous_data["global_impedance_(raw)"]
    recording.set_native_data(native)
