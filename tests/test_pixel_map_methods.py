"""Methods of `PixelMap`, `PixelMask` and `PixelMaskCollection`.

Each method must give the values, warnings and errors of eitprocessing
1.8.7 on seeded random grids with NaN pixels, zeros, negative values and
weights, on grids that are NaN everywhere and on grids of zeros. The
warnings are compared by topic, because the messages are reworded. The
differences that the Provenance block of `m3resp/data/pixel_maps.py` names
(division warnings, no all-NaN warning for a map, the negative warning in
copies, NumPy numbers in ``normalize``) have their own tests.
"""

from __future__ import annotations

import dataclasses
import operator
import os
import warnings
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

import m3resp
from m3resp.data import PixelMap, PixelMask, PixelMaskCollection, Signal
from m3resp.data.pixel_maps import PIXEL_MAP_KINDS

upstream_pixelmap = pytest.importorskip("eitprocessing.datahandling.pixelmap")
upstream_roi = pytest.importorskip("eitprocessing.roi")
upstream_collection = pytest.importorskip("eitprocessing.roi.pixelmaskcollection")

UPSTREAM_CLASSES = {
    None: upstream_pixelmap.PixelMap,
    "tiv": upstream_pixelmap.TIVMap,
    "amplitude": upstream_pixelmap.AmplitudeMap,
    "odcl": upstream_pixelmap.ODCLMap,
    "difference": upstream_pixelmap.DifferenceMap,
    "perfusion": upstream_pixelmap.PerfusionMap,
    "pendelluft": upstream_pixelmap.PendelluftMap,
    "signed_pendelluft": upstream_pixelmap.SignedPendelluftMap,
    "integer": upstream_pixelmap.IntegerMap,
}
KIND_OF_CLASS = {cls: kind for kind, cls in UPSTREAM_CLASSES.items()}
FLOAT_KINDS = tuple(kind for kind in UPSTREAM_CLASSES if kind != "integer")

#: Warning topics: (upstream text, m3resp text, topic).
_TOPICS = (
    ("initialized with negative values", "holds negative values", "negative"),
    ("Dividing by 0", "Dividing by 0", "divide by 0"),
    ("Normalization by a negative", "divides by a negative", "negative reference"),
    ("Mask contains 0 values", "holds 0 values", "zero to NaN"),
    ("Mask contains only NaN", "NaN everywhere", "mask all NaN"),
)
#: Upstream warnings with no m3resp counterpart (see the module docstring).
_UPSTREAM_ONLY = (
    "initialized with all NaN values",
    "'where' used without 'out'",
    # `IntegerMap` casts NaN and infinity to int before it raises TypeError.
    "invalid value encountered in cast",
)


def _grids() -> dict[str, np.ndarray]:
    """Seeded 6 x 5 grids: NaN pixels, zeros, negative values, and so on."""

    rng = np.random.default_rng(20261030)
    mixed = rng.normal(size=(6, 5))
    mixed[rng.random((6, 5)) < 0.2] = np.nan
    mixed[0, 0] = 0.0
    positive = rng.uniform(0.1, 3.0, size=(6, 5))
    positive[1, 2] = np.nan
    negative = -rng.uniform(0.1, 3.0, size=(6, 5))
    whole = rng.integers(-3, 4, size=(6, 5)).astype(float)
    return {
        "mixed": mixed,
        "positive": positive,
        "negative": negative,
        "whole": whole,
        "zeros": np.zeros((6, 5)),
        "all_nan": np.full((6, 5), np.nan),
    }


GRIDS = _grids()


def _masks() -> dict[str, tuple[np.ndarray, bool]]:
    """Seeded mask grids, and whether to keep their zeros as weight 0."""

    rng = np.random.default_rng(31)
    binary_a = (rng.random((6, 5)) < 0.6).astype(float)
    binary_b = (rng.random((6, 5)) < 0.5).astype(float)
    weights = rng.uniform(0.0, 1.0, size=(6, 5)).round(2)
    weights[rng.random((6, 5)) < 0.3] = np.nan
    weights[2, 2] = 0.0
    return {
        "binary_a": (binary_a, False),
        "binary_b": (binary_b, False),
        "weights": (weights, True),
        "ones": (np.ones((6, 5)), False),
        "all_nan": (np.full((6, 5), np.nan), False),
    }


MASKS = _masks()


def _topic(message: str, side: int) -> str | None:
    if side == 0 and any(text in message for text in _UPSTREAM_ONLY):
        return None
    for patterns in _TOPICS:
        if patterns[side] in message:
            return patterns[2]
    return message


def _run(function: Callable[[], Any], side: int) -> tuple[Any, type | None, set]:
    """Return the result, the error type and the warning topics of a call."""

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            result, error = function(), None
        except Exception as exc:  # noqa: BLE001 - the error type is compared
            result, error = None, type(exc)
    topics = {_topic(str(item.message), side) for item in caught} - {None}
    return result, error, topics


def _values(result: Any) -> Any:
    if isinstance(result, upstream_roi.PixelMask):
        return result.mask
    if isinstance(result, (PixelMap, PixelMask, upstream_pixelmap.PixelMap)):
        return result.values
    return result


def _assert_same(
    upstream: Callable[[], Any],
    ours: Callable[[], Any],
    *,
    extra_ours: frozenset[str] = frozenset(),
) -> tuple[Any, Any]:
    """Run both calls and check the values, the error type and the warnings."""

    up_result, up_error, up_topics = _run(upstream, 0)
    our_result, our_error, our_topics = _run(ours, 1)
    assert our_error is up_error
    assert our_topics == up_topics | set(extra_ours)
    if up_error is None:
        np.testing.assert_array_equal(_values(our_result), _values(up_result))
    return up_result, our_result


def _upstream_map(grid: np.ndarray, kind: str | None = None) -> Any:
    return UPSTREAM_CLASSES[kind](grid)


def _our_map(grid: np.ndarray, kind: str | None = None, name: str = "map") -> PixelMap:
    return PixelMap(name=name, values=grid, kind=kind)


def _quiet(function: Callable[[], Any]) -> Any:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return function()


# --- kinds ------------------------------------------------------------------


@pytest.mark.parametrize("kind", FLOAT_KINDS)
@pytest.mark.parametrize("grid_name", sorted(GRIDS))
def test_each_kind_gives_the_warnings_of_its_upstream_subclass(kind, grid_name):
    grid = GRIDS[grid_name]
    _assert_same(lambda: _upstream_map(grid, kind), lambda: _our_map(grid, kind))


@pytest.mark.parametrize("kind", ["amplitude", "perfusion", "pendelluft"])
def test_negative_warning_can_be_switched_off_as_upstream(kind):
    grid = GRIDS["mixed"]
    _assert_same(
        lambda: UPSTREAM_CLASSES[kind](grid, suppress_negative_warning=True),
        lambda: PixelMap(
            name="map", values=grid, kind=kind, suppress_negative_warning=True
        ),
    )


@pytest.mark.parametrize("grid_name", sorted(GRIDS))
def test_integer_kind_takes_whole_numbers_only_as_upstream(grid_name):
    grid = GRIDS[grid_name]
    _assert_same(
        lambda: _upstream_map(grid, "integer"), lambda: _our_map(grid, "integer")
    )


@pytest.mark.parametrize(
    "grid",
    [
        [[1.0, np.nan], [2.0, 2.0]],
        [[1.0, np.inf], [2.0, 2.0]],
        [[1.0, -np.inf], [2.0, 2.0]],
        [[1.0, 2.5], [2.0, 2.0]],
    ],
    ids=["nan", "infinity", "minus infinity", "not whole"],
)
def test_integer_kind_refuses_nan_infinity_and_fractions_as_upstream(grid):
    grid = np.array(grid)
    _assert_same(
        lambda: _upstream_map(grid, "integer"), lambda: _our_map(grid, "integer")
    )
    with pytest.raises(TypeError, match="whole number in every pixel"):
        _our_map(grid, "integer")


def test_integer_kind_keeps_whole_numbers_and_gives_them_as_integers():
    region_numbers = _our_map(np.array([[1.0, 0.0], [-2.0, 2.0]]), "integer")

    np.testing.assert_array_equal(region_numbers.values, [[1.0, 0.0], [-2.0, 2.0]])
    np.testing.assert_array_equal(region_numbers.to_integer_array(), [[1, 0], [-2, 2]])


@pytest.mark.parametrize("symbol", ["+", "-", "*"])
def test_integer_map_with_a_nan_grid_is_refused_as_upstream(symbol):
    grid = GRIDS["whole"]
    other = np.full(grid.shape, np.nan)
    function = {"+": operator.add, "-": operator.sub, "*": operator.mul}[symbol]
    _assert_same(
        lambda: function(_upstream_map(grid, "integer"), other),
        lambda: function(_our_map(grid, "integer"), other),
    )


def test_unknown_kind_is_refused():
    with pytest.raises(ValueError, match="kind"):
        _our_map(GRIDS["positive"], "volume")


def test_kinds_are_the_eight_upstream_subclasses():
    assert set(PIXEL_MAP_KINDS) == set(UPSTREAM_CLASSES) - {None}


def test_a_map_nan_everywhere_gives_no_warning():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _our_map(GRIDS["all_nan"], "tiv")


@pytest.mark.parametrize("target", list(UPSTREAM_CLASSES))
@pytest.mark.parametrize("grid_name", ["mixed", "whole", "positive"])
def test_convert_to_gives_the_upstream_subclass(target, grid_name):
    grid = GRIDS[grid_name]
    up_map = _quiet(lambda: _upstream_map(grid))
    our_map = _our_map(grid)
    _, ours = _assert_same(
        lambda: up_map.convert_to(UPSTREAM_CLASSES[target]),
        lambda: our_map.convert_to(target),
    )
    if ours is not None:
        assert ours.kind == target
        assert ours.name == "map"


# --- normalize --------------------------------------------------------------

_NORMALIZE_CASES = [
    {"mode": "zero-based"},
    {"mode": "symmetric"},
    {"mode": "maximum"},
    {"mode": "reference", "reference": 2.0},
    {"mode": "reference", "reference": -3},
    {"mode": "reference", "reference": 0},
    {"mode": "reference", "reference": float("nan")},
    {"mode": "reference", "reference": "2"},
    {"mode": "reference"},
    {"mode": "maximum", "reference": 2.0},
    {"mode": "middle"},
]


@pytest.mark.parametrize("kind", [None, "amplitude", "tiv"])
@pytest.mark.parametrize("case", _NORMALIZE_CASES, ids=str)
@pytest.mark.parametrize("grid_name", sorted(GRIDS))
def test_normalize_gives_the_upstream_values(grid_name, case, kind):
    grid = GRIDS[grid_name]
    up_map = _quiet(lambda: _upstream_map(grid, kind))
    our_map = _quiet(lambda: _our_map(grid, kind))
    _, ours = _assert_same(
        lambda: up_map.normalize(**case), lambda: our_map.normalize(**case)
    )
    if ours is not None:
        assert ours.kind == kind
        assert ours.name == "map"


def test_normalize_is_refused_for_the_integer_kind_as_upstream():
    grid = GRIDS["whole"]
    _assert_same(
        lambda: _upstream_map(grid, "integer").normalize(),
        lambda: _our_map(grid, "integer").normalize(),
    )


def test_normalize_takes_changes_of_the_other_fields():
    scaled = _our_map(GRIDS["positive"]).normalize(mode="maximum", name="scaled")

    assert scaled.name == "scaled"
    assert np.nanmax(scaled.values) == 1.0


# --- create_mask_from_threshold ---------------------------------------------

_THRESHOLD_CASES = [
    {"threshold": 0.15, "fraction_of_max": True},
    {"threshold": 0.5, "fraction_of_max": True},
    {"threshold": 0.9, "fraction_of_max": True},
    {"threshold": 0.5},
    {"threshold": -1},
    {"threshold": 2},
    {"threshold": np.float32(0.25)},
    {"threshold": 0.5, "use_magnitude": True},
    {"threshold": 0.5, "use_magnitude": True, "fraction_of_max": True},
    {"threshold": 0.5, "comparator": np.less},
    {"threshold": 0.0, "comparator": operator.gt},
    {"threshold": "0.5"},
    {"threshold": 0.5, "comparator": "less"},
]


@pytest.mark.parametrize("case", _THRESHOLD_CASES, ids=str)
@pytest.mark.parametrize("grid_name", sorted(GRIDS))
def test_threshold_masks_equal_upstream(grid_name, case):
    grid = GRIDS[grid_name]
    up_map = _quiet(lambda: _upstream_map(grid, "tiv"))
    our_map = _our_map(grid, "tiv")
    up_captures: dict[str, Any] = {}
    our_captures: dict[str, Any] = {}
    _, ours = _assert_same(
        lambda: up_map.create_mask_from_threshold(**case, captures=up_captures),
        lambda: our_map.create_mask_from_threshold(**case, captures=our_captures),
    )
    assert our_captures == up_captures
    if ours is not None:
        assert ours.name == "map_mask"


def test_threshold_mask_takes_a_name():
    mask = _our_map(GRIDS["positive"]).create_mask_from_threshold(
        0.15, fraction_of_max=True, name="tiv_lungspace_mask"
    )

    assert mask.name == "tiv_lungspace_mask"


# --- arithmetic -------------------------------------------------------------

_OPERATORS = {
    "+": operator.add,
    "-": operator.sub,
    "*": operator.mul,
    "/": operator.truediv,
}


@pytest.mark.parametrize("symbol", sorted(_OPERATORS))
@pytest.mark.parametrize("kind", [None, "amplitude", "tiv"])
@pytest.mark.parametrize("left", ["mixed", "positive", "zeros", "all_nan"])
@pytest.mark.parametrize("right", ["mixed", "positive", "zeros"])
def test_map_with_map_equals_upstream(symbol, kind, left, right):
    function = _OPERATORS[symbol]
    up_left = _quiet(lambda: _upstream_map(GRIDS[left], kind))
    up_right = _quiet(lambda: _upstream_map(GRIDS[right], "tiv"))
    our_left = _quiet(lambda: _our_map(GRIDS[left], kind, name="a"))
    our_right = _our_map(GRIDS[right], "tiv", name="b")
    up_result, ours = _assert_same(
        lambda: function(up_left, up_right), lambda: function(our_left, our_right)
    )
    assert ours.kind == KIND_OF_CLASS[type(up_result)]
    assert ours.name == f"a {symbol} b"


@pytest.mark.parametrize("symbol", sorted(_OPERATORS))
@pytest.mark.parametrize("kind", [None, "amplitude", "tiv"])
@pytest.mark.parametrize("number", [2.5, -1, 0, 0.0, np.float64(3.0)], ids=str)
@pytest.mark.parametrize("grid_name", ["mixed", "positive", "zeros", "all_nan"])
@pytest.mark.parametrize("number_first", [False, True])
def test_map_with_number_equals_upstream(symbol, kind, number, grid_name, number_first):
    function = _OPERATORS[symbol]
    up_map = _quiet(lambda: _upstream_map(GRIDS[grid_name], kind))
    our_map = _quiet(lambda: _our_map(GRIDS[grid_name], kind))
    if number_first:
        upstream = lambda: function(number, up_map)
        ours = lambda: function(number, our_map)
    else:
        upstream = lambda: function(up_map, number)
        ours = lambda: function(our_map, number)
    # I20: dividing by the number 0 warns in m3resp only.
    by_zero = symbol == "/" and not number_first and number == 0
    extra = frozenset({"divide by 0"}) if by_zero else frozenset()
    up_result, result = _assert_same(upstream, ours, extra_ours=extra)
    assert result.kind == KIND_OF_CLASS[type(up_result)] == kind
    assert result.name == "map"


@pytest.mark.parametrize("symbol", sorted(_OPERATORS))
@pytest.mark.parametrize("grid_name", ["mixed", "positive", "zeros"])
@pytest.mark.parametrize("other_name", ["mixed", "positive", "zeros"])
def test_map_with_grid_equals_upstream(symbol, grid_name, other_name):
    function = _OPERATORS[symbol]
    other = GRIDS[other_name]
    up_map = _upstream_map(GRIDS[grid_name], "tiv")
    our_map = _our_map(GRIDS[grid_name], "tiv")
    _, result = _assert_same(
        lambda: function(up_map, other), lambda: function(our_map, other)
    )
    assert result.kind == "tiv"


@pytest.mark.parametrize("symbol", sorted(_OPERATORS))
def test_maps_of_other_shapes_are_refused_as_upstream(symbol):
    function = _OPERATORS[symbol]
    _assert_same(
        lambda: function(_upstream_map(GRIDS["positive"]), np.ones((5, 6))),
        lambda: function(_our_map(GRIDS["positive"]), np.ones((5, 6))),
    )


def test_numpy_integer_works_like_a_python_number():
    pixel_map = _our_map(GRIDS["positive"])

    np.testing.assert_array_equal(
        (pixel_map * np.int64(2)).values, pixel_map.values * 2
    )


def test_division_gives_no_numpy_warning():
    # The result is filled with NaN first.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _our_map(GRIDS["positive"]) / _our_map(GRIDS["positive"])
        2.0 / _our_map(GRIDS["positive"])


def test_sum_of_two_maps_keeps_a_shared_unit_and_category():
    first = PixelMap(name="a", values=[[1.0]], unit="AU", category="impedance")
    second = PixelMap(name="b", values=[[2.0]], unit="AU", category="impedance")

    total = first + second
    product = first * second

    assert (total.unit, total.category) == (first.unit, "impedance")
    assert (product.unit, product.category) == (None, None)


def test_map_with_a_number_keeps_its_fields_and_a_copy_of_the_metadata():
    pixel_map = PixelMap(
        name="tiv", values=[[1.0]], unit="AU", kind="tiv", metadata={"n": 1}
    )

    doubled = pixel_map * 2

    assert (doubled.name, doubled.unit, doubled.kind) == ("tiv", pixel_map.unit, "tiv")
    doubled.metadata["n"] = 2
    assert pixel_map.metadata == {"n": 1}


def test_copies_keep_the_switched_off_negative_warning():
    pixel_map = PixelMap(
        name="a",
        values=[[-1.0, 2.0], [3.0, 4.0]],
        kind="amplitude",
        suppress_negative_warning=True,
    )
    mask = PixelMask(name="m", values=[[1.0, 1.0], [1.0, np.nan]])

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        pixel_map * 2
        2 - pixel_map
        pixel_map.normalize(mode="maximum")
        pixel_map.convert_to("perfusion")
        mask.apply(pixel_map)
        PixelMaskCollection([mask]).apply(pixel_map, name_format="{mask_name} map")
        dataclasses.replace(pixel_map, name="b")


def test_mask_collection_apply_gives_each_masked_map_with_its_fields():
    pixel_map = PixelMap(
        name="a", values=[[1.0, 2.0], [3.0, 4.0]], kind="tiv", metadata={"n": 1}
    )
    masks = [
        PixelMask(name="left", values=[[1.0, np.nan], [1.0, np.nan]]),
        PixelMask(name="right", values=[[np.nan, 0.5], [np.nan, 0.5]]),
    ]
    found = PixelMaskCollection(masks).apply(
        pixel_map, name_format="tiv in {mask_name}", unit="AU"
    )

    assert list(found) == ["left", "right"]
    for mask in masks:
        result = found[mask.name]
        assert result.name == f"tiv in {mask.name}"
        assert (result.kind, result.unit) == ("tiv", "a.u.")
        assert np.array_equal(
            result.values, pixel_map.values * mask.values, equal_nan=True
        )
        result.metadata["n"] = 2
    assert pixel_map.metadata == {"n": 1}


def test_the_warning_of_a_copy_points_to_the_line_of_the_user():
    pixel_map = PixelMap(name="a", values=[[1.0, 2.0]], kind="amplitude")

    with pytest.warns(UserWarning, match="negative") as record:
        pixel_map * -1

    assert record[0].filename == __file__


def test_the_warning_of_a_map_made_inside_m3resp_points_to_the_line_of_the_user():
    # A function with a file name in the m3resp package, as an m3resp method
    # that makes an amplitude map.
    package = os.path.dirname(os.path.abspath(m3resp.__file__))
    source = (
        "def make_map():\n"
        "    return PixelMap(name='a', values=[[-1.0, 2.0]], kind='amplitude')\n"
    )
    namespace = {"PixelMap": PixelMap}
    code = compile(source, os.path.join(package, "eit", "method.py"), "exec")
    exec(code, namespace)  # noqa: S102 - fixed test source

    with pytest.warns(UserWarning, match="negative") as record:
        namespace["make_map"]()

    assert record[0].filename == __file__


@pytest.mark.parametrize("flag", [True, False, np.True_])
def test_calculation_with_a_map_refuses_true_and_false(flag):
    pixel_map = _our_map(GRIDS["positive"])

    for calculate in (
        lambda: pixel_map + flag,
        lambda: flag - pixel_map,
        lambda: pixel_map * flag,
        lambda: pixel_map / flag,
        lambda: flag / pixel_map,
    ):
        with pytest.raises(TypeError, match="not with True or False"):
            calculate()


def test_normalize_takes_numpy_numbers_and_refuses_true_and_false():
    pixel_map = _our_map(GRIDS["positive"])

    for reference in (np.int64(2), np.float32(2.0)):
        np.testing.assert_allclose(
            pixel_map.normalize(mode="reference", reference=reference).values,
            pixel_map.values / 2,
        )
    for reference in (True, np.bool_(True)):
        with pytest.raises(TypeError, match="must be a number"):
            pixel_map.normalize(mode="reference", reference=reference)
        with pytest.raises(TypeError, match="must be a number"):
            pixel_map.create_mask_from_threshold(reference)


def test_a_mask_leaves_the_given_grid_unchanged():
    grid = np.array([[0.0, 1.0], [0.5, np.nan]])
    weights = PixelMask(name="k", values=grid.copy(), keep_zeros=True)

    _quiet(lambda: PixelMask(name="x", values=grid))
    _quiet(lambda: PixelMaskCollection([weights]).combine(name="combined"))

    np.testing.assert_array_equal(grid, [[0.0, 1.0], [0.5, np.nan]])
    np.testing.assert_array_equal(weights.values, [[0.0, 1.0], [0.5, np.nan]])


def test_an_integer_map_with_fractions_names_convert_to():
    regions = PixelMap(name="regions", values=[[1.0, 2.0]], kind="integer")

    with pytest.raises(TypeError, match=r"convert_to\(None\)"):
        regions / 2
    np.testing.assert_array_equal((regions.convert_to(None) / 2).values, [[0.5, 1.0]])


# --- from_aggregate and arrays ----------------------------------------------


def _stack() -> list[np.ndarray]:
    rng = np.random.default_rng(7)
    maps = []
    for _ in range(5):
        grid = rng.normal(1.0, 0.5, size=(6, 5))
        grid[rng.random((6, 5)) < 0.3] = np.nan
        grid[0, 0] = np.nan  # NaN in every map
        maps.append(grid)
    return maps


@pytest.mark.parametrize(
    "aggregator",
    [np.nanmean, np.nanmedian, np.nanmax, np.nanmin, np.nanstd],
    ids=lambda function: function.__name__,
)
@pytest.mark.parametrize("as_maps", [False, True])
def test_from_aggregate_equals_upstream(aggregator, as_maps):
    grids = _stack()
    up_maps = [upstream_pixelmap.PixelMap(grid) for grid in grids] if as_maps else grids
    our_maps = [_our_map(grid) for grid in grids] if as_maps else grids
    _, ours = _assert_same(
        lambda: upstream_pixelmap.TIVMap.from_aggregate(up_maps, aggregator),
        lambda: PixelMap.from_aggregate(our_maps, aggregator, name="mean", kind="tiv"),
    )
    assert np.isnan(ours.values[0, 0])
    assert ours.kind == "tiv"


def test_from_aggregate_refuses_grids_that_are_not_2d_and_no_maps():
    _assert_same(
        lambda: upstream_pixelmap.PixelMap.from_aggregate([np.ones(3)], np.nanmean),
        lambda: PixelMap.from_aggregate([np.ones(3)], np.nanmean, name="m"),
    )
    _assert_same(
        lambda: upstream_pixelmap.PixelMap.from_aggregate([], np.nanmean),
        lambda: PixelMap.from_aggregate([], np.nanmean, name="m"),
    )


@pytest.mark.parametrize("grid_name", sorted(GRIDS))
def test_array_conversions_equal_upstream(grid_name):
    grid = GRIDS[grid_name]
    up_map = _quiet(lambda: _upstream_map(grid))
    our_map = _our_map(grid)
    for call in (
        lambda m: m.to_boolean_array(),
        lambda m: m.to_boolean_array(zero=True),
        lambda m: m.to_integer_array(),
        lambda m: m.to_non_nan_array(),
        lambda m: m.to_non_nan_array(nan=-1.0, dtype=np.float32),
    ):
        expected = call(up_map)
        result = call(our_map)
        assert result.dtype == expected.dtype
        np.testing.assert_array_equal(result, expected)


# --- PixelMask ----------------------------------------------------------------


def _upstream_mask(name: str) -> Any:
    grid, keep_zeros = MASKS[name]
    return _quiet(lambda: upstream_roi.PixelMask(grid, keep_zeros=keep_zeros))


def _our_mask(name: str) -> PixelMask:
    grid, keep_zeros = MASKS[name]
    return _quiet(lambda: PixelMask(name=name, values=grid, keep_zeros=keep_zeros))


_MASK_OPERATORS = {"*": operator.mul, "+": operator.add, "-": operator.sub}
_MASK_WORDS = {"*": "and", "+": "or", "-": "without"}


@pytest.mark.parametrize("symbol", sorted(_MASK_OPERATORS))
@pytest.mark.parametrize("left", sorted(MASKS))
@pytest.mark.parametrize("right", sorted(MASKS))
def test_mask_set_rules_equal_upstream(symbol, left, right):
    function = _MASK_OPERATORS[symbol]
    up_left, up_right = _upstream_mask(left), _upstream_mask(right)
    our_left, our_right = _our_mask(left), _our_mask(right)
    _, ours = _assert_same(
        lambda: function(up_left, up_right), lambda: function(our_left, our_right)
    )
    assert ours.name == f"{left} {_MASK_WORDS[symbol]} {right}"


@pytest.mark.parametrize("name", sorted(MASKS))
def test_is_weighted_equals_upstream(name):
    assert _our_mask(name).is_weighted == _upstream_mask(name).is_weighted


def test_masks_of_other_shapes_are_refused():
    first = PixelMask(name="a", values=np.ones((2, 2)))
    second = PixelMask(name="b", values=np.ones((1, 2)))

    for function in _MASK_OPERATORS.values():
        with pytest.raises(ValueError, match="same shape"):
            function(first, second)


def test_a_mask_does_not_combine_with_a_number():
    with pytest.raises(TypeError):
        PixelMask(name="a", values=np.ones((2, 2))) * 2


# --- PixelMaskCollection --------------------------------------------------------

_COLLECTIONS = [
    ("binary_a", "binary_b"),
    ("binary_a", "weights"),
    ("binary_a", "binary_b", "weights"),
    ("binary_a", "all_nan"),
    ("weights",),
    ("binary_a",),
]


@pytest.mark.parametrize("method", ["sum", "product", "mean"])
@pytest.mark.parametrize("names", _COLLECTIONS, ids=str)
def test_combine_equals_upstream(method, names):
    up_collection = upstream_collection.PixelMaskCollection(
        [_upstream_mask(name) for name in names]
    )
    our_collection = PixelMaskCollection(items=[_our_mask(name) for name in names])
    up_result, up_error, _ = _run(lambda: up_collection.combine(method), 0)
    our_result, our_error, _ = _run(
        lambda: our_collection.combine(method, name="combined"), 1
    )
    assert our_error is up_error
    if up_error is None:
        np.testing.assert_array_equal(our_result.values, up_result.mask)
        assert our_result.name == "combined"
        assert [mask.name for mask in our_collection] == list(names)


def test_combine_of_an_empty_collection_is_refused():
    with pytest.raises(ValueError, match="empty"):
        PixelMaskCollection().combine(name="combined")


def test_apply_equals_upstream_on_arrays_and_maps():
    names = ("binary_a", "weights", "ones")
    up_collection = upstream_collection.PixelMaskCollection(
        [_quiet(lambda n=name: _upstream_mask(n).update(label=n)) for name in names]
    )
    # `update` runs the checks again with the default settings, so a weight
    # of 0 becomes NaN; `dataclasses.replace` does the same here.
    our_collection = PixelMaskCollection(
        items=[
            _quiet(lambda n=name: dataclasses.replace(_our_mask(n), name=n))
            for name in names
        ]
    )
    images = np.random.default_rng(3).normal(size=(4, 6, 5))

    up_arrays = up_collection.apply(images)
    our_arrays = our_collection.apply(images)
    assert list(our_arrays) == list(names)
    for name in names:
        np.testing.assert_array_equal(our_arrays[name], up_arrays[name])

    grid = GRIDS["mixed"]
    up_maps = up_collection.apply(
        upstream_pixelmap.TIVMap(grid), label_format="tiv in {mask_label}"
    )
    our_maps = our_collection.apply(
        _our_map(grid, "tiv"), name_format="tiv in {mask_name}"
    )
    for name in names:
        np.testing.assert_array_equal(our_maps[name].values, up_maps[name].values)
        assert our_maps[name].name == up_maps[name].label == f"tiv in {name}"
        assert our_maps[name].kind == "tiv"


def test_apply_on_a_signal_keeps_its_fields():
    signal = Signal(
        name="pixels",
        values=np.ones((3, 2, 2)),
        time=np.array([0.0, 0.1, 0.2]),
        sample_frequency=10.0,
        modality="eit",
    )
    collection = PixelMaskCollection(
        items=[PixelMask(name="left", values=[[1.0, np.nan], [1.0, np.nan]])]
    )

    masked = collection.apply(signal, name_format="{mask_name} pixels")["left"]

    assert masked.name == "left pixels"
    assert masked.sample_frequency == 10.0
    np.testing.assert_array_equal(masked.time, signal.time)
    assert np.isnan(masked.values[:, :, 1]).all()


@pytest.mark.parametrize(
    ("data", "options", "error"),
    [
        (np.ones((2, 2)), {"name_format": "{mask_name}"}, ValueError),
        (np.ones((2, 2)), {"unit": "AU"}, ValueError),
        (PixelMap(name="m", values=np.ones((2, 2))), {"name_format": "x"}, ValueError),
        (PixelMap(name="m", values=np.ones((2, 2))), {"name": "x"}, ValueError),
        ([[1.0, 1.0]], {}, TypeError),
    ],
)
def test_apply_refuses_wrong_options(data, options, error):
    collection = PixelMaskCollection(
        items=[PixelMask(name="a", values=np.ones((2, 2)))]
    )

    with pytest.raises(error):
        collection.apply(data, **options)


def test_apply_refuses_two_masks_with_the_same_name_and_an_empty_collection():
    mask = PixelMask(name="a", values=np.ones((2, 2)))
    twin = PixelMask(name="a", values=np.ones((2, 2)))

    with pytest.raises(ValueError, match="more than one mask"):
        PixelMaskCollection(items=[mask, twin]).apply(np.ones((2, 2)))
    with pytest.raises(ValueError, match="empty"):
        PixelMaskCollection().apply(np.ones((2, 2)))
