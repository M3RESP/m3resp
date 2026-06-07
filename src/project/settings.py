from omegaconf import DictConfig
from pathlib import Path
from typing import Any
import warnings
import tomllib

import omegaconf


def load_settings(*, warn_if_missing: bool = True):
    """Load settings from TOML config files using OmegaConf."""

    config_files = sorted(Path("config").glob("*.toml"))

    # Load base and local config
    toml_configs = [read_toml_config(file) for file in config_files]
    configs = [
        omegaconf.OmegaConf.create(config)
        for config in toml_configs
        if config is not None
    ]

    # Merge recursively — subsequent files override earlier files
    settings = omegaconf.OmegaConf.merge(*configs)
    omegaconf.OmegaConf.resolve(settings)
    convert_paths(settings.paths, warn_if_missing=warn_if_missing)
    return settings


# Read TOML config file (OmegaConf does not support TOML by default)
def read_toml_config(file: str) -> dict[str, Any] | None:
    path = Path(file)
    if not path.exists():
        warnings.warn(f"Could not load config file '{path}'.")
        return None
    with Path(file).open("rb") as f:
        return tomllib.load(f)


# Convert all path strings to Path objects
def convert_paths(obj, warn_if_missing: bool = True):
    for k, v in obj.items():
        if isinstance(v, str):
            obj[k] = Path(v)
            if warn_if_missing and not obj[k].exists():
                warnings.warn(f"Could not find configured path '{obj[k]}'.")
        elif isinstance(v, omegaconf.omegaconf.DictConfig):
            convert_paths(v)


def create_missing_directories(settings):
    """Create all directories specified in settings.paths if they do not
    exist."""
    for obj in settings.values():
        if isinstance(obj, DictConfig):
            create_missing_directories(obj)
        elif isinstance(obj, Path) and not obj.exists():
            if obj.suffix != "":
                obj = obj.parent
            obj.mkdir(parents=True, exist_ok=True)
