"""EIT recordings: loading, and cutting to a time window."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np

from m3resp.data.event_data import EventData, IntervalData
from m3resp.data.signals import Signal
from m3resp.eit.io.file_data import EITFileData
from m3resp.eit.io.join import join_eit_file_data
from m3resp.eit.io.read_file import read_eit_file
from m3resp.modalities.time_window import TimeWindow

#: A custom EIT loader: called as ``loader(path, vendor=..., **options)``,
#: it returns the data of the recording as an ``EITFileData``.
EITLoader = Callable[..., EITFileData]


@dataclass
class EITRecording:
    """A loaded EIT recording: images, waveforms, marks and breaths of the device.

    ``paths`` lists the files read, in recording order: one file, or several
    files joined into this recording. ``pixel_impedance`` holds the images
    (frames, rows, columns), ``signals`` the waveforms of the device (one
    value per frame), ``events`` the marks of the device and ``intervals``
    the breaths found by the device (Timpel). The keys are the labels of the
    file readers, for example ``"global_impedance_(raw)"``.
    """

    paths: tuple[Path, ...]
    vendor: str
    sample_frequency: float
    pixel_impedance: Signal
    signals: dict[str, Signal] = field(default_factory=dict)
    events: dict[str, EventData] = field(default_factory=dict)
    intervals: dict[str, IntervalData] = field(default_factory=dict)
    _image_sum: tuple[Signal, Signal] | None = field(
        default=None, init=False, repr=False, compare=False
    )

    @classmethod
    def from_file_data(cls, file_data: EITFileData) -> EITRecording:
        """Return a recording that holds the data of ``file_data``.

        Args:
            file_data (EITFileData): What the file reader read.

        Returns:
            EITRecording: The recording. Its dictionaries are copies; the
                Signals and event data are the same objects.
        """

        recording = cls(
            paths=tuple(file_data.paths),
            vendor=file_data.vendor,
            sample_frequency=file_data.sample_frequency,
            pixel_impedance=file_data.pixel_impedance,
        )
        recording.set_file_data(file_data)
        return recording

    def set_file_data(self, file_data: EITFileData) -> None:
        """Replace the images, waveforms, marks and breaths with those of ``file_data``.

        Args:
            file_data (EITFileData): The new data, for example a time window
                of this recording.
        """

        self.pixel_impedance = file_data.pixel_impedance
        self.signals = dict(file_data.signals)
        self.events = dict(file_data.events)
        self.intervals = dict(file_data.intervals)
        self.sample_frequency = file_data.sample_frequency

    def to_file_data(self) -> EITFileData:
        """Return the data of this recording as an ``EITFileData``.

        Returns:
            EITFileData: The same Signals and event data, in new dictionaries.
        """

        return EITFileData(
            vendor=self.vendor,
            paths=self.paths,
            sample_frequency=self.sample_frequency,
            pixel_impedance=self.pixel_impedance,
            signals=dict(self.signals),
            events=dict(self.events),
            intervals=dict(self.intervals),
        )

    @property
    def raw_global_impedance(self) -> Signal:
        """The raw global impedance: one value per frame.

        This is the waveform the file gives (``signals["global_impedance_(raw)"]``,
        Draeger and Timpel). For a file without one (Sentec), it is the sum
        of all pixels of each raw image. The sum is computed once and kept
        until ``pixel_impedance`` is replaced by another Signal.

        Returns:
            Signal: The waveform.
        """

        signal = self.signals.get("global_impedance_(raw)")
        if signal is not None:
            return signal
        if self._image_sum is None or self._image_sum[0] is not self.pixel_impedance:
            from m3resp.eit.processing import global_impedance

            summed = global_impedance(
                self.pixel_impedance, name="Global impedance (raw)"
            )
            self._image_sum = (self.pixel_impedance, summed)
        return self._image_sum[1]


EITLoaderChoice = Literal["auto", "vendor", "custom"]

#: The reading options of the vendor readers (``m3resp.eit.io.read_eit_file``).
_READ_OPTIONS = ("sample_frequency", "first_frame", "max_frames")


def _read_vendor_files(
    paths: tuple[Path, ...], vendor: str | None, **options: Any
) -> EITFileData:
    """Read Draeger, Sentec or Timpel files and join them into one recording.

    Args:
        paths (tuple[Path, ...]): The files, in recording order.
        vendor (str | None): ``"draeger"``, ``"sentec"`` or ``"timpel"``.
        **options (Any): ``sample_frequency``, ``first_frame`` and
            ``max_frames`` for ``m3resp.eit.io.read_eit_file``.

    Returns:
        EITFileData: The data of all files, joined in the given order.

    Raises:
        ValueError: If ``vendor`` is None, if ``first_frame`` or
            ``max_frames`` is set with more than one file (they would apply
            to each file), or if the files cannot be joined.
        TypeError: If an option is unknown.
    """

    if vendor is None:
        raise ValueError(
            "EIT loading needs the device maker: give vendor='draeger', "
            "'sentec' or 'timpel', or set a custom loader."
        )
    unknown = set(options) - set(_READ_OPTIONS)
    if unknown:
        raise TypeError(
            f"Unknown EIT loading option(s) {sorted(unknown)}. Known: "
            f"{', '.join(_READ_OPTIONS)}."
        )
    if len(paths) > 1 and (
        options.get("first_frame") or options.get("max_frames") is not None
    ):
        raise ValueError(
            "first_frame and max_frames apply to each file, so they cannot "
            "be used when loading several EIT files at once. Load the "
            "files, then cut the recording with eit.slice_recording "
            "(session.slice_eit)."
        )
    files = [read_eit_file(path, vendor, **options) for path in paths]
    return join_eit_file_data(files)


def load(
    path: str | Path | Sequence[str | Path],
    vendor: str | None = None,
    *,
    custom_loader: EITLoader | None = None,
    loader: EITLoaderChoice = "auto",
    **kwargs: Any,
) -> EITRecording:
    """Load an EIT recording from one file or from several files in a row.

    ``loader`` picks the reader: ``"vendor"`` reads a Draeger, Sentec or
    Timpel file with the m3resp reader of ``vendor``
    (``m3resp.eit.io.read_eit_file``), ``"custom"`` uses ``custom_loader``,
    and ``"auto"`` uses ``custom_loader`` when there is one, else the vendor
    reader.

    Args:
        path (str | Path | Sequence[str | Path]): The file, or a list of
            files recorded one after the other. The vendor reader joins the
            files in the given order (``m3resp.eit.io.join_eit_file_data``);
            a pause between two files stays in the time axis and gives a
            warning. A custom loader receives the list as given.
        vendor (str | None): ``"draeger"``, ``"sentec"`` or ``"timpel"``.
            Needed for the vendor reader.
        custom_loader (EITLoader | None): A function called as
            ``custom_loader(path, vendor=..., **kwargs)`` that returns an
            ``EITFileData``. With ``loader="custom"``, ``vendor`` is None.
        loader (EITLoaderChoice): ``"auto"``, ``"vendor"`` or ``"custom"``.
        **kwargs (Any): For the vendor reader: ``sample_frequency`` (Hz),
            ``first_frame`` and ``max_frames``. For a custom loader: its
            own options.

    Returns:
        EITRecording: The recording.

    Raises:
        ValueError: If no file is given, if ``loader`` is unknown, if
            ``vendor`` is given with ``loader="custom"`` or missing for the
            vendor reader, or if the files cannot be joined.
        TypeError: If a custom loader returns something other than an
            ``EITFileData``, or a reading option is unknown.
        UnsupportedWorkflowError: If ``loader="custom"`` and no custom
            loader is set.
    """

    if isinstance(path, str | Path):
        paths: tuple[Path, ...] = (Path(path),)
        source: Any = str(path)
    else:
        if not path:
            raise ValueError("Give at least one EIT file to load.")
        paths = tuple(Path(item) for item in path)
        source = [str(item) for item in paths]
    if loader not in ("auto", "vendor", "custom"):
        raise ValueError(
            f"loader must be 'auto', 'vendor' or 'custom'; got {loader!r}."
        )
    if loader == "custom" and vendor is not None:
        raise ValueError(
            "A custom EIT loader takes no vendor; pass reading options "
            "for the loader as keyword arguments."
        )

    if loader == "custom" or (loader == "auto" and custom_loader is not None):
        if custom_loader is None:
            from m3resp.core.exceptions import UnsupportedWorkflowError

            raise UnsupportedWorkflowError(
                "No custom EIT loader is set. Create the session with "
                "M3Session(eit_loader=my_loader), or pass the loader as "
                "run_workflow(..., eit_loader=my_loader). Use 'eit.load' with "
                "a vendor for Draeger, Sentec or Timpel files."
            )
        file_data = custom_loader(source, vendor=vendor, **kwargs)
        if not isinstance(file_data, EITFileData):
            raise TypeError(
                "A custom EIT loader must return an m3resp.eit.io.EITFileData, "
                f"got {type(file_data).__name__}."
            )
    else:
        file_data = _read_vendor_files(paths, vendor, **kwargs)
    return EITRecording.from_file_data(file_data)


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
    are selected by the times of the kept frames. Original times and
    collection keys are kept.

    Args:
        recording (EITRecording): The recording to cut. It changes in place.
        start_index (int): Index of the first frame to keep.
        end_index (int): Index after the last frame to keep.

    Raises:
        ValueError: If no frame is kept.
    """

    from m3resp.eit.processing.slicing import slice_file_data

    recording.set_file_data(
        slice_file_data(recording.to_file_data(), start_index, end_index)
    )
