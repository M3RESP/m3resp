"""Store extra results of a processing function in an optional dictionary.

A processing function that has useful in-between results (filter settings,
peak properties) takes a `captures` dictionary argument. When the caller
passes a dictionary, the function stores those results in it. When the
caller passes None, nothing is stored and the function returns only its main
result.
"""

from __future__ import annotations

from typing import Any


def capture_value(
    captures: dict[str, Any] | None,
    key: str,
    value: Any,
    *,
    append_to_list: bool = False,
) -> None:
    """Capture a value in a dictionary.

    Args:
        captures (dict or None): Dictionary to capture values in.
            If None, no values will be captured.
        key (str): Key to use for capturing the value.
        value (Any): Value to capture.
        append_to_list (bool): If True, the value will be appended to a list
            under the given key.
    """
    if captures is None:
        return
    if append_to_list:
        captures.setdefault(key, []).append(value)
    else:
        captures[key] = value
