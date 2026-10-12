"""NOTICE.md and the "Provenance" blocks in the modules list the same files.

A module that holds code taken from eitprocessing or ReSurfEMG starts its
docstring with a "Provenance" block containing the line
``Portions of this module are derived from <project>.``. NOTICE.md lists each
such file under that project's heading.
"""

from __future__ import annotations

import os
import re

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOURCE_DIR = os.path.join(REPO_ROOT, "src", "m3resp")
NOTICE_PATH = os.path.join(REPO_ROOT, "NOTICE.md")

PROJECTS = ("eitprocessing", "ReSurfEMG")


def _files_with_provenance_block(project: str) -> set[str]:
    """Repository-relative paths (with ``/``) of modules naming ``project``."""

    marker = f"Portions of this module are derived from {project}."
    found: set[str] = set()
    for folder, _, file_names in os.walk(SOURCE_DIR):
        for file_name in file_names:
            if not file_name.endswith(".py"):
                continue
            path = os.path.join(folder, file_name)
            with open(path, encoding="utf-8") as handle:
                if marker in handle.read():
                    relative = os.path.relpath(path, REPO_ROOT)
                    found.add(relative.replace(os.sep, "/"))
    return found


def _files_listed_in_notice(project: str) -> set[str]:
    """Paths in backticks under the ``## <project>`` heading of NOTICE.md."""

    with open(NOTICE_PATH, encoding="utf-8") as handle:
        text = handle.read()
    sections = re.split(r"^## ", text, flags=re.MULTILINE)
    section = next(part for part in sections if part.startswith(project))
    return set(re.findall(r"`(src/[^`]+\.py)`", section))


@pytest.mark.parametrize("project", PROJECTS)
def test_notice_lists_every_module_with_a_provenance_block(project: str) -> None:
    in_modules = _files_with_provenance_block(project)
    in_notice = _files_listed_in_notice(project)

    assert in_modules, f"no module names {project}; check the marker text"
    assert in_modules - in_notice == set(), (
        f"modules with a {project} Provenance block missing from NOTICE.md"
    )
    assert in_notice - in_modules == set(), (
        f"files listed under {project} in NOTICE.md without a Provenance block"
    )
