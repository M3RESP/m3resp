"""``PixelBreaths``: the breaths of each pixel of an EIT image.

Each pixel of an EIT image has its own impedance waveform, so each pixel has
its own breaths. The breaths of a pixel are close in time to the breaths of
the global impedance, but can start and end a little earlier or later.

A :class:`PixelBreaths` holds the global breaths, and for each global breath
the start, the extremum (end of inspiration) and the end time of the breath
of every pixel. A pixel with no breath in a global breath has NaN for its
three times.

The times are three arrays of shape (breath, row, column). For one waveform,
the same times are a list of ``BreathEvent``: one time per breath. TIV and
EELI read the samples at these times in the same way for both forms.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from m3resp.data.event_data import IntervalData
from m3resp.data.events import BreathEvent, Interval

#: The three breath times of each pixel, in the order of the last axis of the
#: grids in ``PixelBreaths.to_interval_data``.
PIXEL_BREATH_POINTS = ("start_time", "extremum_time", "end_time")
#: Metadata key of ``PixelBreaths.to_interval_data`` with the number of rows
#: and columns of the image, also when there are no breaths.
GRID_SHAPE_KEY = "grid_shape"


@dataclass(eq=False)
class PixelBreaths:
    """The start, extremum and end time of the breath of each pixel.

    Attributes:
        name: What the breaths are, e.g. ``'pixel_breaths'``.
        breaths: The global breaths, in time order. The pixel breaths of
            global breath ``i`` are at position ``i`` of the time arrays.
        start_time: Start of each pixel breath, in seconds on the time axis
            of ``breaths``. Shape (breath, row, column).
        extremum_time: End of inspiration of each pixel breath, in seconds.
            Same shape.
        end_time: End of each pixel breath, in seconds. Same shape.
        modality: The device the breaths were found in.
        method: Name of the method that found the pixel breaths.
        metadata: Optional extra information.

    A pixel with no breath in a global breath has NaN for all three times.

    Raises:
        TypeError: If an item of ``breaths`` is not a ``BreathEvent``, or the
            times are not numbers.
        ValueError: If the three time arrays are not 3D with the same shape,
            their first axis does not have one entry per global breath, a
            pixel breath has NaN for only some of its times, or its times
            are not in the order start, extremum, end.
    """

    name: str
    breaths: list[BreathEvent]
    start_time: np.ndarray
    extremum_time: np.ndarray
    end_time: np.ndarray
    modality: str = "eit"
    method: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Check the global breaths and the three time arrays."""

        self.breaths = list(self.breaths)
        for breath in self.breaths:
            if not isinstance(breath, BreathEvent):
                raise TypeError(
                    "PixelBreaths.breaths must be BreathEvent objects, got "
                    f"{type(breath).__name__}."
                )
        for point in PIXEL_BREATH_POINTS:
            setattr(self, point, _as_time_array(getattr(self, point), point))
        shapes = {getattr(self, point).shape for point in PIXEL_BREATH_POINTS}
        if len(shapes) != 1:
            raise ValueError(
                "PixelBreaths: start_time, extremum_time and end_time must have "
                f"the same shape, got {sorted(shapes)}."
            )
        if self.start_time.shape[0] != len(self.breaths):
            raise ValueError(
                f"PixelBreaths: the times have {self.start_time.shape[0]} "
                f"breath(s) on their first axis, for {len(self.breaths)} "
                "global breath(s)."
            )

        start, extremum, end = (getattr(self, point) for point in PIXEL_BREATH_POINTS)
        found = ~np.isnan(start)
        if not (
            np.array_equal(found, ~np.isnan(extremum))
            and np.array_equal(found, ~np.isnan(end))
        ):
            raise ValueError(
                "PixelBreaths: a pixel breath must have all three times, or "
                "NaN for all three."
            )
        if np.any(extremum[found] < start[found]) or np.any(
            end[found] < extremum[found]
        ):
            raise ValueError(
                "PixelBreaths: each pixel breath needs start_time <= "
                "extremum_time <= end_time."
            )

    @property
    def shape(self) -> tuple[int, int]:
        """Return the number of rows and columns of the image, in that order."""

        _, rows, columns = self.start_time.shape
        return rows, columns

    @property
    def has_breath(self) -> np.ndarray:
        """Return True where a pixel has a breath, shape (breath, row, column)."""

        return ~np.isnan(self.start_time)

    def pixel(self, row: int, column: int) -> list[BreathEvent]:
        """Return the breaths of one pixel as ``BreathEvent`` objects.

        Analyses written for global breaths (durations, timing checks,
        linking) can then run on one pixel too.

        Args:
            row (int): Row of the pixel, counted from 0.
            column (int): Column of the pixel, counted from 0.

        Returns:
            list[BreathEvent]: One breath per global breath in which the
                pixel has a breath, in time order. ``start_time``,
                ``extremum_time`` and ``end_time`` are the times of the pixel,
                in seconds. A global breath in which the pixel has no breath
                gives no entry. ``modality`` and ``source`` are ``modality``
                and ``method`` of this object. ``metadata`` holds ``row``,
                ``column`` and ``global_breath_id``, the ``id`` of the global
                breath.

        Raises:
            ValueError: If ``row`` or ``column`` is outside the grid.
        """

        rows, columns = self.shape
        if not (0 <= row < rows and 0 <= column < columns):
            raise ValueError(
                f"Pixel ({row}, {column}) is outside the {rows} x {columns} grid."
            )
        events: list[BreathEvent] = []
        for index, breath in enumerate(self.breaths):
            if np.isnan(self.start_time[index, row, column]):
                continue
            events.append(
                BreathEvent(
                    self.modality,
                    float(self.start_time[index, row, column]),
                    float(self.end_time[index, row, column]),
                    extremum_time=float(self.extremum_time[index, row, column]),
                    source=self.method,
                    metadata={
                        "row": row,
                        "column": column,
                        "global_breath_id": breath.id,
                    },
                )
            )
        return events

    def to_interval_data(self) -> IntervalData:
        """Return the pixel breaths as one grid of times per global breath.

        Returns:
            IntervalData: The global breaths, with one (row, column, 3) grid
                per breath: the start, extremum and end time of each pixel,
                in seconds (NaN for no breath). ``name``, ``modality``,
                ``method`` and a copy of ``metadata`` come from this object;
                the unit is ``'s'``. ``metadata["grid_shape"]`` holds the
                number of rows and columns.
        """

        grids = np.stack(
            [getattr(self, point) for point in PIXEL_BREATH_POINTS], axis=-1
        )
        return IntervalData(
            name=self.name,
            modality=self.modality,
            intervals=list[Interval](self.breaths),
            values=list(grids),
            unit="s",
            method=self.method,
            metadata={**self.metadata, GRID_SHAPE_KEY: list(self.shape)},
        )

    @classmethod
    def from_interval_data(cls, data: IntervalData) -> PixelBreaths:
        """Return the pixel breaths held in an ``IntervalData`` of time grids.

        Args:
            data (IntervalData): The global breaths, with one (row, column, 3)
                grid per breath: the start, extremum and end time of each
                pixel, in seconds, NaN for no breath. This is the form of
                ``pixel_breath_timing_result`` of ``eit.pixel_breaths`` and of
                :meth:`to_interval_data`.

        Returns:
            PixelBreaths: The same breaths and times. ``name``, ``modality``,
                ``method`` and a copy of ``metadata`` come from ``data``,
                without ``"grid_shape"``. With no breaths, the grid has the
                rows and columns of ``metadata["grid_shape"]`` (0 x 0 when it
                is missing).

        Raises:
            TypeError: If an interval of ``data`` is not a ``BreathEvent``.
            ValueError: If ``data`` holds no values, the values are not
                (row, column, 3) grids of the same shape, or
                ``metadata["grid_shape"]`` differs from the grids.
        """

        if data.values is None:
            raise ValueError(
                f"'{data.name}' holds no grids of start, middle and end times."
            )
        grids = [np.asarray(grid, dtype=float) for grid in data.values]
        shapes = {grid.shape for grid in grids}
        for shape in shapes:
            if len(shape) != 3 or shape[-1] != 3:
                raise ValueError(
                    f"'{data.name}' must hold (row, column, 3) grids of start, "
                    f"middle and end times, got shape {shape}."
                )
        if len(shapes) > 1:
            raise ValueError(
                f"'{data.name}' holds grids of different shapes: {sorted(shapes)}."
            )
        metadata = dict(data.metadata)
        grid_shape = metadata.pop(GRID_SHAPE_KEY, None)
        if grids:
            times = np.stack(grids)
            if grid_shape is not None and list(grid_shape) != list(times.shape[1:3]):
                raise ValueError(
                    f"'{data.name}' holds {times.shape[1]} x {times.shape[2]} "
                    f"grids, but its metadata gives the grid shape {grid_shape}."
                )
        else:
            rows, columns = grid_shape if grid_shape is not None else (0, 0)
            times = np.empty((0, rows, columns, 3))
        breaths: list[BreathEvent] = []
        for interval in data.intervals:
            if not isinstance(interval, BreathEvent):
                raise TypeError(
                    f"The intervals of '{data.name}' must be BreathEvent "
                    f"objects, got {type(interval).__name__}."
                )
            breaths.append(interval)
        return cls(
            name=data.name,
            breaths=breaths,
            start_time=times[..., 0],
            extremum_time=times[..., 1],
            end_time=times[..., 2],
            modality=data.modality,
            method=data.method,
            metadata=metadata,
        )

    def __eq__(self, other: object) -> bool:
        """Return True if all fields and times are equal.

        Two NaN times at the same position count as equal.

        Args:
            other (object): The object to compare with.

        Returns:
            bool: True when ``other`` is a ``PixelBreaths`` with the same
                fields and times. ``NotImplemented`` for another type.
        """

        if type(other) is not type(self):
            return NotImplemented
        assert isinstance(other, PixelBreaths)
        return (
            self.name == other.name
            and self.breaths == other.breaths
            and all(
                np.array_equal(
                    getattr(self, point), getattr(other, point), equal_nan=True
                )
                for point in PIXEL_BREATH_POINTS
            )
            and self.modality == other.modality
            and self.method == other.method
            and self.metadata == other.metadata
        )


def _as_time_array(values: Any, point: str) -> np.ndarray:
    """Return one time array of ``PixelBreaths`` as a 3D float array.

    Args:
        values (Any): The times, in seconds, with shape (breath, row,
            column).
        point (str): Name of the time array, for the messages.

    Returns:
        np.ndarray: A copy of the times as floats.

    Raises:
        TypeError: If the values are not numbers.
        ValueError: If the array is not 3D.
    """

    try:
        times = np.array(values, dtype=float)
    except (TypeError, ValueError) as error:
        raise TypeError(f"PixelBreaths.{point} must hold numbers.") from error
    if times.ndim != 3:
        raise ValueError(
            f"PixelBreaths.{point} must have shape (breath, row, column), got "
            f"{times.ndim} dimension(s)."
        )
    return times
