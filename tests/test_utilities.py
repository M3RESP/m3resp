"""Small argument helpers in `m3resp.core.utilities` and
`m3resp.processing.validation`.

`_validate_incompatible_kwargs` exists in both modules with the same code, so
its tests run against both copies.
"""

from __future__ import annotations

import numpy as np
import pytest

from m3resp.core import utilities
from m3resp.core.exceptions import MutuallyExclusiveArgsError
from m3resp.processing import validation

VALIDATORS = [
    pytest.param(utilities._validate_incompatible_kwargs, id="core.utilities"),
    pytest.param(validation._validate_incompatible_kwargs, id="processing.validation"),
]


@pytest.mark.parametrize("validate", VALIDATORS)
class TestValidateIncompatibleKwargs:
    def test_neither_set_returns_default(self, validate):
        assert validate("a", "b", default_value=0.1) == 0.1

    def test_first_named_argument(self, validate):
        assert validate("a", "b", arg_value1=5, default_value=0.1) == 5

    def test_second_named_argument(self, validate):
        assert validate("a", "b", arg_value2=7) == 7

    def test_values_found_in_kwargs(self, validate):
        assert validate("a", "b", default_value=0.1, b=3) == 3
        assert validate("a", "b", default_value=0.1, a=2, other=9) == 2

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"arg_value1": 1, "arg_value2": 2},
            {"arg_value1": 1, "b": 2},
            {"a": 1, "b": 2},
        ],
        ids=["named", "named+kwargs", "kwargs"],
    )
    def test_both_set_raises(self, validate, kwargs):
        with pytest.raises(MutuallyExclusiveArgsError, match="a and b cannot both"):
            validate("a", "b", **kwargs)

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "Bug: `arg_value1 or kwargs.get(...)` treats a falsy value such as "
            "0 as unset, so it falls through to the other argument or default."
        ),
    )
    def test_zero_is_a_set_value(self, validate):
        assert validate("a", "b", arg_value1=0, default_value=0.1) == 0


class TestCaptureValue:
    def test_none_captures_is_a_no_op(self):
        utilities.capture_value(None, "key", 1)

    def test_stores_value(self):
        captures: dict = {}
        utilities.capture_value(captures, "key", 1)
        utilities.capture_value(captures, "key", 2)
        assert captures == {"key": 2}

    def test_appends_to_list(self):
        captures: dict = {}
        utilities.capture_value(captures, "key", 1, append_to_list=True)
        utilities.capture_value(captures, "key", 2, append_to_list=True)
        assert captures == {"key": [1, 2]}


class TestRequireSamplingFrequency:
    def test_returns_float(self):
        result = validation._require_sampling_frequency(2048)
        assert result == 2048.0
        assert isinstance(result, float)

    def test_none_is_rejected(self):
        with pytest.raises(ValueError, match="sample_frequency is required"):
            validation._require_sampling_frequency(None)

    @pytest.mark.parametrize("fs", [0, -10.0, np.inf, np.nan])
    def test_non_positive_or_non_finite_is_rejected(self, fs):
        with pytest.raises(ValueError, match="finite and positive"):
            validation._require_sampling_frequency(fs)
