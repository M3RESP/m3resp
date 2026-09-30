"""Utility functions for the m3resp package."""

from typing import Any

from m3resp.core.exceptions import MutuallyExclusiveArgsError


def _validate_incompatible_kwargs(
    arg_name1: str = "Arg1",
    arg_name2: str = "Arg2",
    arg_value1: Any | None = None,  # noqa: ANN401
    arg_value2: Any | None = None,  # noqa: ANN401
    *,
    default_value: Any | None = None,  # noqa: ANN401
    **kwargs,
) -> Any:  # noqa: ANN401
    """Validate that two mutually exclusive parameters are not both set.

    Checks that only one of two mutually exclusive parameters is set,
    and returns the value of the one that is set, or a default value if neither is set.
    This is useful for functions that accept parameters under multiple names for
    backward compatibility or convenience: in such cases, the "alternative"
    parameter name is often passed in via ``kwargs``.

    Arguments:
        arg_name1 (str): Name of the first argument.
        arg_name2 (str): Name of the second argument.
        arg_value1 (Any | None): Value of the first argument.
            If not provided, it will be looked up in ``kwargs``.
        arg_value2 (Any | None, optional): Value of the second argument.
            If not provided, it will be looked up in ``kwargs``.
        default_value (Any | None): Default value to use if neither argument is provided
        **kwargs: caller function kwargs.

    Returns:
        Any: The value of the argument that is set, or the default value if neither is set.

    Raises:
        MutuallyExclusiveArgsError: If both arguments are set.

    Example:
        case 1: arg1 is a named argument, arg2 is passed in via kwargs
        ```python
        def my_function(arg1=None, **kwargs):
            used_arg_name, arg_value = _validate_incompatible_kwargs(
                "arg1", "arg2", arg_value1=arg1, default_value=0.1, **kwargs
            )
            ```

        case 2: both arg1 and arg2 are named arguments
        ```python
        def my_function(arg1=None, arg2=None):
            used_arg_name, arg_value = _validate_incompatible_kwargs(
                "arg1", "arg2", arg_value1=arg1, arg_value2=arg2, default_value=0.1
            )
        ```

        case 3: both arg1 and arg2 are passed in via kwargs
        ```python
        def my_function(**kwargs):
            used_arg_name, arg_value = _validate_incompatible_kwargs(
                "arg1", "arg2", default_value=0.1, **kwargs
            )
        ```
    """
    arg_value1 = arg_value1 or kwargs.get(arg_name1)
    arg_value2 = arg_value2 or kwargs.get(arg_name2)
    if arg_value1 is not None and arg_value2 is not None:
        msg_0 = f"{arg_name1} and {arg_name2} cannot both be set at the same time."
        raise MutuallyExclusiveArgsError(msg_0)
    if arg_value1 is not None:
        return arg_value1
    if arg_value2 is not None:
        return arg_value2
    return default_value


def capture_value(
    captures: dict[str, Any] | None,
    key: str,
    value: Any,  # noqa: ANN401
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
