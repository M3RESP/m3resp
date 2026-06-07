from pathlib import Path
import tomllib
from __version__ import __version__ as version
from .settings import load_settings
from . import run


def _load_pyproject():
    pyproject_path = Path(__file__).resolve().parent.parent.parent / "pyproject.toml"
    if not pyproject_path.exists():
        return {}

    with pyproject_path.open("rb") as f:
        return tomllib.load(f).get("project", {})


_pyproject_data = _load_pyproject()
name = _pyproject_data.get("name")

__all__ = ["load_settings", "version", "name", "run"]
