"""Checks of argument values that functions in several modules use."""

from __future__ import annotations

from typing import Any


def check_fraction(value: Any, *, name: str, function: str) -> None:
    """Raise an error unless ``value`` is a float between 0 and 1 (both left out).

    Args:
        value (Any): The fraction.
        name (str): Name of the argument, for the error message.
        function (str): Name of the function, for the error message.

    Raises:
        TypeError: If ``value`` is not a float (an int such as ``1`` too).
        ValueError: If ``value`` is not between 0 and 1.
    """

    if not isinstance(value, float):
        raise TypeError(
            f"{function} '{name}' must be a float, got {type(value).__name__}."
        )
    if not 0 < value < 1:
        raise ValueError(f"{function} '{name}' must be between 0 and 1, got {value!r}.")
