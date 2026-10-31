"""`PixelBreaths`: the start, extremum and end time of the breath of each pixel."""

from __future__ import annotations

import os
import warnings

import numpy as np
import pytest

from m3resp.data import (
    BreathEvent,
    Interval,
    IntervalData,
    PixelBreaths,
    pixel_breath_events,
)
from m3resp.data.signals import Signal
from m3resp.eit.processing._breath_samples import (
    NO_SAMPLE,
    breath_samples,
    nearest_samples,
)

GOLDEN_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "regression", "golden", "eit"
)
DATASETS = (
    "draeger_synthetic_20Hz",
    "sentec_healthy_volunteer_1b",
    "timpel_healthy_volunteer_1_60-120s",
)
POINTS = ("start_time", "extremum_time", "end_time")


def _global_breaths() -> list[BreathEvent]:
    return [
        BreathEvent("eit", 0.0, 4.0, extremum_time=1.5),
        BreathEvent("eit", 4.0, 8.0, extremum_time=5.5),
    ]


def _pixel_breaths() -> PixelBreaths:
    """Two global breaths on a 1 x 2 image. Pixel (0, 1) has no breath in
    the second global breath."""

    return PixelBreaths(
        name="pixel_breaths",
        breaths=_global_breaths(),
        start_time=np.array([[[0.1, 0.3]], [[4.1, np.nan]]]),
        extremum_time=np.array([[[1.6, 1.8]], [[5.6, np.nan]]]),
        end_time=np.array([[[4.1, 4.2]], [[8.1, np.nan]]]),
        method="test",
    )


def _pixel_signal(time: np.ndarray, shape: tuple[int, int]) -> Signal:
    return Signal(
        values=np.zeros((time.size, *shape)),
        time=time,
        sample_frequency=1 / float(np.median(np.diff(time))),
        name="pixels",
        modality="eit",
    )


def _golden(dataset: str) -> tuple[list[BreathEvent], np.ndarray, np.ndarray]:
    """Return the global breaths, the pixel landmarks and the time axis of
    one golden test file."""

    with np.load(os.path.join(GOLDEN_DIR, dataset, "pixel.npz")) as data:
        landmarks = data["breaths/landmarks"]
    with np.load(os.path.join(GOLDEN_DIR, dataset, "breaths.npz")) as data:
        global_times = data["breaths"]
    with np.load(os.path.join(GOLDEN_DIR, dataset, "reader.npz")) as data:
        time = data["time"]
    breaths = [
        BreathEvent("eit", float(start), float(end), extremum_time=float(middle))
        for start, middle, end in global_times
    ]
    return breaths, landmarks, time


def _golden_pixel_breaths(dataset: str) -> PixelBreaths:
    breaths, landmarks, _ = _golden(dataset)
    return PixelBreaths.from_interval_data(
        IntervalData(
            name="pixel_breaths",
            modality="eit",
            intervals=breaths,
            values=list(landmarks),
            unit="s",
            method="eitprocessing.PixelBreath",
        )
    )


class TestPixelBreaths:
    def test_times_are_float_arrays_of_breath_row_column(self):
        pixel_breaths = _pixel_breaths()

        assert pixel_breaths.start_time.dtype == float
        assert pixel_breaths.start_time.shape == (2, 1, 2)
        assert pixel_breaths.shape == (1, 2)
        assert pixel_breaths.has_breath.tolist() == [[[True, True]], [[True, False]]]

    def test_times_of_different_shapes_are_refused(self):
        with pytest.raises(ValueError, match="same shape"):
            PixelBreaths(
                name="pixel_breaths",
                breaths=_global_breaths(),
                start_time=np.zeros((2, 1, 2)),
                extremum_time=np.zeros((2, 1, 2)),
                end_time=np.zeros((2, 2, 2)),
            )

    def test_one_entry_per_global_breath_is_needed(self):
        with pytest.raises(ValueError, match="for 2 global breath"):
            PixelBreaths(
                name="pixel_breaths",
                breaths=_global_breaths(),
                start_time=np.zeros((3, 1, 2)),
                extremum_time=np.zeros((3, 1, 2)),
                end_time=np.zeros((3, 1, 2)),
            )

    def test_times_that_are_not_3d_are_refused(self):
        with pytest.raises(ValueError, match="breath, row, column"):
            PixelBreaths(
                name="pixel_breaths",
                breaths=_global_breaths(),
                start_time=np.zeros((2, 2)),
                extremum_time=np.zeros((2, 2)),
                end_time=np.zeros((2, 2)),
            )

    def test_times_that_are_not_numbers_are_refused(self):
        with pytest.raises(TypeError, match="must hold numbers"):
            PixelBreaths(
                name="pixel_breaths",
                breaths=_global_breaths()[:1],
                start_time=np.array([[["a"]]]),
                extremum_time=np.array([[[1.0]]]),
                end_time=np.array([[[2.0]]]),
            )

    def test_a_pixel_breath_with_only_some_times_is_refused(self):
        with pytest.raises(ValueError, match="all three times"):
            PixelBreaths(
                name="pixel_breaths",
                breaths=_global_breaths()[:1],
                start_time=np.array([[[0.0]]]),
                extremum_time=np.array([[[np.nan]]]),
                end_time=np.array([[[2.0]]]),
            )

    @pytest.mark.parametrize(
        ("start", "extremum", "end"), [(1.0, 0.5, 2.0), (0.0, 2.5, 2.0)]
    )
    def test_times_out_of_order_are_refused(self, start, extremum, end):
        with pytest.raises(ValueError, match="start_time <= extremum_time"):
            PixelBreaths(
                name="pixel_breaths",
                breaths=_global_breaths()[:1],
                start_time=np.array([[[start]]]),
                extremum_time=np.array([[[extremum]]]),
                end_time=np.array([[[end]]]),
            )

    def test_global_breaths_must_be_breath_events(self):
        with pytest.raises(TypeError, match="BreathEvent"):
            PixelBreaths(
                name="pixel_breaths",
                breaths=[Interval("eit", 0.0, 4.0, name="noise")],  # type: ignore[list-item]
                start_time=np.zeros((1, 1, 1)),
                extremum_time=np.zeros((1, 1, 1)),
                end_time=np.zeros((1, 1, 1)),
            )

    def test_equal_objects_compare_equal_with_nan_times(self):
        assert _pixel_breaths() == _pixel_breaths()
        other = _pixel_breaths()
        other.end_time[0, 0, 0] = 4.0
        assert other != _pixel_breaths()


class TestPixel:
    def test_one_breath_event_per_global_breath_with_the_pixel_times(self):
        pixel_breaths = _pixel_breaths()

        events = pixel_breaths.pixel(0, 0)

        assert [(e.start_time, e.extremum_time, e.end_time) for e in events] == [
            (0.1, 1.6, 4.1),
            (4.1, 5.6, 8.1),
        ]
        assert all(e.modality == "eit" and e.source == "test" for e in events)
        assert [e.metadata["global_breath_id"] for e in events] == [
            b.id for b in pixel_breaths.breaths
        ]
        assert events[0].metadata["row"] == 0
        assert events[0].metadata["column"] == 0

    def test_a_global_breath_without_a_pixel_breath_gives_no_entry(self):
        events = _pixel_breaths().pixel(0, 1)

        assert [(e.start_time, e.end_time) for e in events] == [(0.3, 4.2)]

    def test_a_pixel_outside_the_grid_is_refused(self):
        with pytest.raises(ValueError, match="outside the 1 x 2 grid"):
            _pixel_breaths().pixel(1, 0)

    def test_pixel_breath_events_gives_the_same_breaths(self):
        pixel_breaths = _pixel_breaths()
        data = pixel_breaths.to_interval_data()

        for column in range(2):
            assert pixel_breath_events(data, 0, column) == pixel_breaths.pixel(
                0, column
            )


class TestIntervalDataForm:
    def test_round_trip(self):
        pixel_breaths = _pixel_breaths()

        data = pixel_breaths.to_interval_data()

        assert data.unit == "s"
        assert data.intervals == pixel_breaths.breaths
        np.testing.assert_array_equal(
            data.values[0], [[[0.1, 1.6, 4.1], [0.3, 1.8, 4.2]]]
        )
        assert PixelBreaths.from_interval_data(data) == pixel_breaths

    def test_no_global_breaths(self):
        data = IntervalData(
            name="pixel_breaths", modality="eit", intervals=[], values=[]
        )

        assert PixelBreaths.from_interval_data(data).breaths == []
        assert pixel_breath_events(data, row=0, column=0) == []

    def test_round_trip_with_no_breaths_keeps_the_grid(self):
        empty = np.empty((0, 32, 32))
        pixel_breaths = PixelBreaths(
            name="pixel_breaths",
            breaths=[],
            start_time=empty,
            extremum_time=empty,
            end_time=empty,
        )

        data = pixel_breaths.to_interval_data()
        back = PixelBreaths.from_interval_data(data)

        assert data.metadata["grid_shape"] == [32, 32]
        assert back.shape == (32, 32)
        assert back == pixel_breaths

    def test_a_grid_shape_that_differs_from_the_grids_is_refused(self):
        data = _pixel_breaths().to_interval_data()
        data.metadata["grid_shape"] = [2, 2]

        with pytest.raises(ValueError, match="gives the grid shape"):
            PixelBreaths.from_interval_data(data)

    def test_no_values_are_refused(self):
        data = IntervalData(
            name="pixel_breaths", modality="eit", intervals=_global_breaths()
        )
        with pytest.raises(ValueError, match="holds no grids"):
            PixelBreaths.from_interval_data(data)

    def test_grids_without_three_times_are_refused(self):
        data = IntervalData(
            name="pixel_tivs",
            modality="eit",
            intervals=_global_breaths()[:1],
            values=[np.ones((1, 2))],
        )
        with pytest.raises(ValueError, match="start, middle and end"):
            PixelBreaths.from_interval_data(data)

    def test_grids_of_different_shapes_are_refused(self):
        data = IntervalData(
            name="pixel_breaths",
            modality="eit",
            intervals=_global_breaths(),
            values=[np.zeros((1, 2, 3)), np.zeros((2, 2, 3))],
        )
        with pytest.raises(ValueError, match="different shapes"):
            PixelBreaths.from_interval_data(data)

    def test_intervals_that_are_not_breaths_are_refused(self):
        data = IntervalData(
            name="pixel_breaths",
            modality="eit",
            intervals=[Interval("eit", 0.0, 4.0, name="noise")],
            values=[np.zeros((1, 2, 3))],
        )
        with pytest.raises(TypeError, match="BreathEvent"):
            PixelBreaths.from_interval_data(data)


# The three golden test files


@pytest.mark.parametrize("dataset", DATASETS)
def test_round_trip_keeps_the_golden_landmarks(dataset: str) -> None:
    breaths, landmarks, _ = _golden(dataset)

    pixel_breaths = _golden_pixel_breaths(dataset)
    data = pixel_breaths.to_interval_data()

    np.testing.assert_array_equal(np.stack(data.values), landmarks)
    assert data.intervals == breaths
    for index, point in enumerate(POINTS):
        np.testing.assert_array_equal(
            getattr(pixel_breaths, point), landmarks[..., index]
        )


@pytest.mark.parametrize("dataset", DATASETS)
def test_pixel_gives_the_golden_times_of_every_pixel(dataset: str) -> None:
    _, landmarks, _ = _golden(dataset)
    pixel_breaths = _golden_pixel_breaths(dataset)
    data = pixel_breaths.to_interval_data()

    rows, columns = pixel_breaths.shape
    for row in range(rows):
        for column in range(columns):
            pixel = landmarks[:, row, column]
            expected = [tuple(times) for times in pixel if not np.isnan(times).any()]
            events = pixel_breaths.pixel(row, column)
            assert [
                (e.start_time, e.extremum_time, e.end_time) for e in events
            ] == expected
            assert pixel_breath_events(data, row, column) == events


@pytest.mark.parametrize("dataset", DATASETS)
def test_sample_lookup_on_the_grid_equals_the_lookup_per_pixel(dataset: str) -> None:
    _, landmarks, time = _golden(dataset)

    grid = nearest_samples(time, landmarks)

    assert grid.shape == landmarks.shape
    for row in range(landmarks.shape[1]):
        for column in range(landmarks.shape[2]):
            for index in range(3):
                np.testing.assert_array_equal(
                    grid[:, row, column, index],
                    nearest_samples(time, list(landmarks[:, row, column, index])),
                )
    # The pixel times of eitprocessing are sample times.
    found = grid != NO_SAMPLE
    np.testing.assert_array_equal(time[grid[found]], landmarks[found])


@pytest.mark.parametrize("dataset", DATASETS)
def test_pixels_with_no_breath_give_no_sample_and_no_warning(dataset: str) -> None:
    _, landmarks, time = _golden(dataset)
    pixel_breaths = _golden_pixel_breaths(dataset)
    signal = _pixel_signal(time, pixel_breaths.shape)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        samples = breath_samples(signal, pixel_breaths, points=POINTS, function="test")

    no_breath = np.isnan(landmarks[..., 0])
    # The first and the last global breath have no pixel breaths.
    assert no_breath[0].all() and no_breath[-1].all()
    for index, point in enumerate(POINTS):
        assert samples[point].shape == no_breath.shape
        assert (samples[point][no_breath] == NO_SAMPLE).all()
        np.testing.assert_array_equal(
            time[samples[point][~no_breath]], landmarks[..., index][~no_breath]
        )


def test_pixel_breaths_outside_the_data_warn_once_and_have_no_sample() -> None:
    time = np.arange(0.0, 8.0, 0.5)
    signal = _pixel_signal(time, (1, 2))
    pixel_breaths = PixelBreaths(
        name="pixel_breaths",
        breaths=_global_breaths(),
        start_time=np.array([[[0.0, 0.5]], [[4.0, 7.0]]]),
        extremum_time=np.array([[[1.5, 2.0]], [[5.5, 7.5]]]),
        end_time=np.array([[[4.0, 4.5]], [[7.5, 9.0]]]),
    )

    with pytest.warns(UserWarning, match="1 of 4 pixel breaths lie partly outside"):
        samples = breath_samples(signal, pixel_breaths, points=POINTS, function="test")

    for point in POINTS:
        assert samples[point][1, 0, 1] == NO_SAMPLE
        assert (samples[point][[0, 0, 1], 0, [0, 1, 0]] != NO_SAMPLE).all()
    assert samples["end_time"][1, 0, 0] == 15


def test_nearest_samples_keeps_the_shape_of_the_times() -> None:
    time = np.array([10.0, 10.5, 11.0, 11.5])
    times = np.array([[10.2, np.nan], [11.76, 10.3]])

    indices = nearest_samples(time, times)

    assert indices.tolist() == [[0, NO_SAMPLE], [NO_SAMPLE, 1]]
