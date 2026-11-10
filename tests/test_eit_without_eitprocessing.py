"""EIT data is read, processed and plotted with m3resp code alone.

These tests block eitprocessing, so Python treats it as not installed, and
then run the EIT parts of m3resp: the package import, the step list, the
session methods, two example workflows and the EIT figures. A static check
looks for eitprocessing imports in ``src/m3resp``.
"""

from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

import m3resp.workflows.steps  # noqa: F401 - registers the built-in steps
from m3resp.workflows.registry import describe_steps, step_capability_state

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = REPO_ROOT / "src" / "m3resp"
DATA_DIR = REPO_ROOT / "tests" / "data"
EXAMPLES_DIR = REPO_ROOT / "examples"

#: Code run first in each subprocess: Python then refuses to import
#: eitprocessing and all its modules, as when it is not installed.
_BLOCK_EITPROCESSING = "import sys\nsys.modules['eitprocessing'] = None\n"

#: The three EIT test files, one of each vendor.
_TEST_FILES = {
    "draeger": DATA_DIR / "draeger_synthetic_draeger_20Hz.bin",
    "sentec": DATA_DIR / "sentec_healthy_volunteer_1b.zri",
    "timpel": DATA_DIR / "timpel_healthy_volunteer_1_60-120s.txt",
}


def _run_blocked(code: str, tmp_path: Path) -> subprocess.CompletedProcess[str]:
    """Run ``code`` in a new Python process with eitprocessing blocked.

    Args:
        code (str): Python code to run after the block.
        tmp_path (Path): Working folder of the process.

    Returns:
        subprocess.CompletedProcess[str]: The finished process, with its
            output and error text.
    """

    return subprocess.run(
        [sys.executable, "-c", _BLOCK_EITPROCESSING + code],
        capture_output=True,
        text=True,
        cwd=tmp_path,
        check=False,
    )


def _assert_success(result: subprocess.CompletedProcess[str]) -> None:
    assert result.returncode == 0, (
        f"The process stopped with code {result.returncode}.\n"
        f"Output:\n{result.stdout[-3000:]}\nErrors:\n{result.stderr[-6000:]}"
    )


def _spec_copy(spec_path: Path, tmp_path: Path) -> Path:
    """Copy an example workflow file with absolute input paths and outputs in tmp_path.

    Args:
        spec_path (Path): The example workflow file.
        tmp_path (Path): Folder for the copy and its outputs.

    Returns:
        Path: The copied workflow file.
    """

    spec = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
    for key, value in (spec.get("inputs") or {}).items():
        if isinstance(value, str) and (spec_path.parent / value).exists():
            spec["inputs"][key] = str((spec_path.parent / value).resolve())
    spec.setdefault("outputs", {})
    spec["outputs"]["dir"] = str(tmp_path / "outputs")
    spec["outputs"]["timestamped"] = False
    target = tmp_path / spec_path.name
    target.write_text(yaml.safe_dump(spec, sort_keys=False), encoding="utf-8")
    return target


# -- the package, the step list and the session -------------------------------


def test_import_m3resp_without_eitprocessing(tmp_path):
    code = (
        "import m3resp\n"
        "import m3resp.workflows.steps\n"
        "import m3resp.core.session\n"
        "import m3resp.visualization.eit_figures\n"
    )
    _assert_success(_run_blocked(code, tmp_path))


def test_step_list_without_eitprocessing(tmp_path):
    code = (
        "from m3resp.__main__ import main\n"
        "try:\n"
        "    main(['steps', '--json'])\n"
        "except SystemExit as exit:\n"
        "    raise SystemExit(exit.code)\n"
    )
    result = _run_blocked(code, tmp_path)
    _assert_success(result)
    names = {entry["name"] for entry in json.loads(result.stdout)}
    assert "eit.load" in names


@pytest.mark.parametrize("vendor", sorted(_TEST_FILES))
def test_load_and_preprocess_eit_without_eitprocessing(vendor, tmp_path):
    code = (
        "from m3resp.core.session import M3Session\n"
        "session = M3Session()\n"
        f"session.load_eit({str(_TEST_FILES[vendor])!r}, vendor={vendor!r})\n"
        "result = session.preprocess_eit()\n"
        "assert len(result['breath_intervals']) > 0\n"
        "assert len(session.interval_data) > 0\n"
    )
    _assert_success(_run_blocked(code, tmp_path))


# -- example workflows and figures ---------------------------------------------


def test_eit_full_example_without_eitprocessing(tmp_path):
    spec = _spec_copy(
        EXAMPLES_DIR / "eit_full_preprocessing" / "eit-full.workflow.yaml", tmp_path
    )
    code = (
        "from m3resp.core.session import M3Session\n"
        "from m3resp.workflows.engine.spec_runner import run_spec\n"
        f"result = run_spec({str(spec)!r}, session=M3Session())\n"
        "assert result.status == 'succeeded', result.status\n"
    )
    _assert_success(_run_blocked(code, tmp_path))
    assert sorted(path.name for path in (tmp_path / "outputs" / "figures").iterdir())


def test_multimodal_full_example_without_eitprocessing(tmp_path):
    pytest.importorskip("resurfemg")
    spec = _spec_copy(
        EXAMPLES_DIR / "multimodal_full" / "multimodal-full.workflow.yaml", tmp_path
    )
    code = (
        "from m3resp.core.session import M3Session\n"
        "from m3resp.workflows.engine.spec_runner import run_spec\n"
        f"result = run_spec({str(spec)!r}, session=M3Session())\n"
        "assert result.status == 'succeeded', result.status\n"
    )
    _assert_success(_run_blocked(code, tmp_path))


def test_save_eit_figures_without_eitprocessing(tmp_path):
    spec = EXAMPLES_DIR / "eit_full_preprocessing" / "eit-full.workflow.yaml"
    code = (
        "from m3resp.core.session import M3Session\n"
        "from m3resp.visualization.eit_figures import save_eit_figures\n"
        "from m3resp.workflows import run_workflow\n"
        "session = M3Session()\n"
        f"result = run_workflow({str(spec)!r}, session=session)\n"
        f"paths = save_eit_figures(result.outputs, {str(tmp_path)!r}, session)\n"
        "names = sorted(path.name for path in paths)\n"
        "assert names == ['global_impedance.png', 'pixel_tiv.png', "
        "'rate_detection.png'], names\n"
    )
    _assert_success(_run_blocked(code, tmp_path))


# -- every EIT step is available ----------------------------------------------

_EIT_STEPS = sorted(d.name for d in describe_steps() if d.name.startswith("eit."))


@pytest.mark.parametrize("step_name", _EIT_STEPS)
def test_every_eit_step_is_available_without_eitprocessing(step_name, monkeypatch):
    monkeypatch.setitem(sys.modules, "eitprocessing", None)
    assert step_capability_state(step_name) == "available"


# -- no eitprocessing import in the package -------------------------------------

#: Functions that import a module given by name as text.
_IMPORT_BY_NAME = {"import_module", "_lazy_import", "__import__", "find_spec"}


def _eitprocessing_imports(path: Path) -> list[str]:
    """List the lines of ``path`` that import eitprocessing.

    Args:
        path (Path): A Python file.

    Returns:
        list[str]: ``"<file>:<line>"`` for each ``import eitprocessing``,
            ``from eitprocessing ...`` and import by name, such as
            ``import_module("eitprocessing.roi")``.
    """

    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found = []
    for node in ast.walk(tree):
        modules: list[str] = []
        if isinstance(node, ast.Import):
            modules = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
            modules = [node.module or ""]
        elif isinstance(node, ast.Call) and node.args:
            function = node.func
            name = (
                function.attr
                if isinstance(function, ast.Attribute)
                else getattr(function, "id", "")
            )
            first = node.args[0]
            if (
                name in _IMPORT_BY_NAME
                and isinstance(first, ast.Constant)
                and isinstance(first.value, str)
            ):
                modules = [first.value]
        if any(module.split(".")[0] == "eitprocessing" for module in modules):
            found.append(f"{path.relative_to(REPO_ROOT)}:{node.lineno}")
    return found


def test_no_eitprocessing_import_in_the_package():
    found = [
        line
        for path in sorted(SOURCE_DIR.rglob("*.py"))
        for line in _eitprocessing_imports(path)
    ]
    assert found == []
