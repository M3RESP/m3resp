"""``EITFileData``: everything one EIT reader reads from one file."""

from __future__ import annotations

from dataclasses import dataclass, field

from m3resp.data.event_data import EventData, IntervalData
from m3resp.data.signals import Signal


@dataclass
class EITFileData:
    """The data of one EIT file, as read by a vendor reader.

    The keys of ``signals``, ``events`` and ``intervals`` are the labels
    eitprocessing 1.8.7 gives the same data, for example
    ``"global_impedance_(raw)"``, ``"airway pressure"`` or
    ``"events_(draeger)"``.

    Attributes:
        vendor (str): The device maker: ``"draeger"``, ``"sentec"`` or
            ``"timpel"``.
        path (str): The file that was read.
        sample_frequency (float): Frames per second, in Hz.
        pixel_impedance (Signal): The EIT images, with values of shape
            (frames, rows, columns) and time in seconds.
        signals (dict[str, Signal]): Waveforms with one value per frame: the
            global impedance and the waveforms the device stores next to the
            images (airway pressure, flow, volume, ...).
        events (dict[str, EventData]): Marks at single times, such as the
            breath minima and maxima found by the device and the event
            markers set by the user.
        intervals (dict[str, IntervalData]): Stretches of time, such as the
            breaths found by the device.
    """

    vendor: str
    path: str
    sample_frequency: float
    pixel_impedance: Signal
    signals: dict[str, Signal] = field(default_factory=dict)
    events: dict[str, EventData] = field(default_factory=dict)
    intervals: dict[str, IntervalData] = field(default_factory=dict)
