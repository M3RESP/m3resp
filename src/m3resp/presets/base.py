"""Built-in presets for fixed sequences of session processing calls.

Each preset processes a loaded session and stores results through session
methods or registered steps. Config supplies settings for the individual
operations. See ``docs/developer/preset-contracts.md`` for the available
presets and ``docs/workflows.md`` for custom YAML/JSON step sequences.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from m3resp.core.session import M3Session

#: Per-method keyword arguments, keyed by the session method name a concrete
#: `Preset` calls (e.g. ``{"preprocess": {...}, "detect_breaths": {...}}``).
PresetConfig = Mapping[str, Mapping[str, Any]]


class Preset(ABC):
    """A named preset that runs a fixed sequence of `M3Session` methods."""

    name: str

    @abstractmethod
    def run(
        self, session: M3Session, *, config: PresetConfig | None = None
    ) -> M3Session:
        """Run the preset's operations on a loaded session.

        Args:
            session: Session containing the recordings to process.
            config: Settings grouped by the keys defined by the concrete preset.
                None uses each operation's defaults.

        Returns:
            M3Session: The supplied session with processing results and provenance
                stored by the operations that ran.
        """

    @staticmethod
    def _kwargs_for(config: PresetConfig | None, step_name: str) -> dict[str, Any]:
        """Copy the settings for step_name, or return an empty dictionary."""

        return dict((config or {}).get(step_name, {}))
