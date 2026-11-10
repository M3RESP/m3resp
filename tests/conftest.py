"""Marks the comparison tests with the ``reference`` marker.

A comparison test checks m3resp against eitprocessing, so it needs the
``reference`` extra. A test gets the marker when its own code, the code of a
fixture it uses, or the top level of its file imports eitprocessing, calls
``pytest.importorskip("eitprocessing")`` or imports the reference code
(``reference.eitprocessing``). ``pytest -m reference`` runs these tests only;
without eitprocessing they skip.
"""

from __future__ import annotations

import ast
import inspect
import re
from functools import cache
from pathlib import Path
from typing import Any

import pytest

#: Text that shows that code needs eitprocessing.
_NEEDS_EITPROCESSING = re.compile(
    r"^\s*(?:import|from)\s+(?:eitprocessing|reference)\b"
    r"|importorskip\(\s*[\"']eitprocessing[\"']",
    re.MULTILINE,
)


@cache
def _file_top_level_needs_eitprocessing(path: Path) -> bool:
    """Return True when the top level of a test file needs eitprocessing.

    Args:
        path (Path): The test file.

    Returns:
        bool: True for an import of eitprocessing or of the reference code,
            or a ``pytest.importorskip("eitprocessing")``, outside the
            functions and classes of the file.
    """

    tree = ast.parse(path.read_text(encoding="utf-8"))
    top_level = [
        node
        for node in tree.body
        if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef)
    ]
    return any(_NEEDS_EITPROCESSING.search(ast.unparse(node)) for node in top_level)


def _source_needs_eitprocessing(function: Any) -> bool:
    """Return True when the code of a function needs eitprocessing."""

    try:
        source = inspect.getsource(function)
    except (OSError, TypeError):
        return False
    return bool(_NEEDS_EITPROCESSING.search(source))


def _needs_eitprocessing(item: pytest.Item) -> bool:
    """Return True when a collected test needs eitprocessing.

    Args:
        item (pytest.Item): The test.

    Returns:
        bool: True when the file top level, the test function or one of its
            fixtures (also fixtures of fixtures) needs eitprocessing.
    """

    path = Path(str(item.fspath))
    if path.suffix == ".py" and _file_top_level_needs_eitprocessing(path):
        return True
    function = getattr(item, "function", None)
    if function is not None and _source_needs_eitprocessing(function):
        return True
    fixture_info = getattr(item, "_fixtureinfo", None)
    if fixture_info is None:
        return False
    return any(
        _source_needs_eitprocessing(definition.func)
        for definitions in fixture_info.name2fixturedefs.values()
        for definition in definitions
    )


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Add the ``reference`` marker to every test that needs eitprocessing."""

    for item in items:
        if _needs_eitprocessing(item):
            item.add_marker(pytest.mark.reference)
