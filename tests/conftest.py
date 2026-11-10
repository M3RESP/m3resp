"""Marks the comparison tests with the ``reference`` marker.

A comparison test checks m3resp against eitprocessing, so it needs the
``reference`` extra. A test gets the marker when it needs eitprocessing:

- the top level of its file imports eitprocessing or calls
  ``pytest.importorskip("eitprocessing...")``, so every test in the file needs
  it;
- its own code, or the code of a fixture it uses, imports eitprocessing or the
  reference code (``reference.eitprocessing``), or calls
  ``pytest.importorskip("eitprocessing...")``;
- its own code, or the code of a fixture it uses, uses a name that the file
  imports from the reference code, or a function of the file that needs
  eitprocessing.

``pytest -m reference`` runs these tests only, and ``pytest -m "not reference"``
runs the other tests. When eitprocessing is not installed, the marked tests
skip.
"""

from __future__ import annotations

import ast
import importlib.util
import inspect
import re
from functools import cache
from pathlib import Path
from typing import Any

import pytest

#: Text that shows that code needs eitprocessing.
_NEEDS_EITPROCESSING = re.compile(
    r"^\s*(?:import|from)\s+(?:eitprocessing|reference)\b"
    r"|importorskip\(\s*[\"']eitprocessing[\w.]*[\"']",
    re.MULTILINE,
)
#: Text of a direct need for eitprocessing at the top level of a file.
_FILE_NEEDS_EITPROCESSING = re.compile(
    r"^\s*(?:import|from)\s+eitprocessing\b"
    r"|importorskip\(\s*[\"']eitprocessing[\w.]*[\"']",
    re.MULTILINE,
)


def _is_function_or_class(node: ast.stmt) -> bool:
    return isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef)


def _reference_names(node: ast.stmt) -> set[str]:
    """Return the names that a top-level import of the reference code binds.

    Args:
        node (ast.stmt): A statement at the top level of a test file.

    Returns:
        set[str]: The names bound by ``from reference... import a, b`` or
            ``import reference... as r``. Empty for any other statement.
    """

    if isinstance(node, ast.ImportFrom) and (node.module or "").split(".")[0] == (
        "reference"
    ):
        return {alias.asname or alias.name for alias in node.names}
    if isinstance(node, ast.Import):
        return {
            alias.asname or alias.name.split(".")[0]
            for alias in node.names
            if alias.name.split(".")[0] == "reference"
        }
    return set()


def _uses_any(source: str, names: set[str]) -> bool:
    """Return True when ``source`` contains one of ``names`` as a word."""

    return any(re.search(rf"\b{re.escape(name)}\b", source) for name in names)


@cache
def _file_needs(path: Path) -> tuple[bool, frozenset[str]]:
    """Return what a test file needs from eitprocessing.

    Args:
        path (Path): The test file.

    Returns:
        tuple[bool, frozenset[str]]: True when the top level of the file
            imports eitprocessing or calls
            ``pytest.importorskip("eitprocessing...")``. Then the names that
            show a need for eitprocessing in code: the names the file imports
            from the reference code, and the top-level functions and classes
            of the file that use eitprocessing or one of these names.
    """

    tree = ast.parse(path.read_text(encoding="utf-8"))
    whole_file = False
    names: set[str] = set()
    definitions: dict[str, str] = {}
    for node in tree.body:
        if _is_function_or_class(node):
            definitions[node.name] = ast.unparse(node)
            continue
        names |= _reference_names(node)
        if _FILE_NEEDS_EITPROCESSING.search(ast.unparse(node)):
            whole_file = True

    changed = True
    while changed:
        changed = False
        for name, source in definitions.items():
            if name not in names and (
                _NEEDS_EITPROCESSING.search(source) or _uses_any(source, names)
            ):
                names.add(name)
                changed = True
    return whole_file, frozenset(names)


def _code_needs_eitprocessing(function: Any) -> bool:
    """Return True when the code of a test or fixture needs eitprocessing.

    Args:
        function (Any): The test function or the fixture function.

    Returns:
        bool: True when the code imports eitprocessing or the reference
            code, calls ``pytest.importorskip("eitprocessing...")``, or uses
            a name that shows a need for eitprocessing in its own file.
    """

    try:
        source = inspect.getsource(function)
        source_file = inspect.getsourcefile(function)
    except (OSError, TypeError):
        return False
    if _NEEDS_EITPROCESSING.search(source):
        return True
    if source_file is None or not source_file.endswith(".py"):
        return False
    _, names = _file_needs(Path(source_file))
    return _uses_any(source, set(names))


def _needs_eitprocessing(item: pytest.Item) -> bool:
    """Return True when a collected test needs eitprocessing.

    Args:
        item (pytest.Item): The test.

    Returns:
        bool: True when the file top level, the test function or one of its
            fixtures (also fixtures of fixtures) needs eitprocessing.
    """

    path = Path(str(item.fspath))
    if path.suffix == ".py" and _file_needs(path)[0]:
        return True
    function = getattr(item, "function", None)
    if function is not None and _code_needs_eitprocessing(function):
        return True
    fixture_info = getattr(item, "_fixtureinfo", None)
    if fixture_info is None:
        return False
    return any(
        _code_needs_eitprocessing(definition.func)
        for definitions in fixture_info.name2fixturedefs.values()
        for definition in definitions
    )


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Mark every test that needs eitprocessing, and skip it without it."""

    eitprocessing_installed = importlib.util.find_spec("eitprocessing") is not None
    skip = pytest.mark.skip(reason="needs eitprocessing (the `reference` extra)")
    for item in items:
        if _needs_eitprocessing(item):
            item.add_marker(pytest.mark.reference)
            if not eitprocessing_installed:
                item.add_marker(skip)
