"""``LinkedBreath``: one respiratory cycle observed across modalities.

Produced by ``m3resp.synchronization.linking.link_breaths_by_time``, which
matches ``Breath``/``BreathEvent`` objects across any number of modalities'
event lists by how close their times are, rather than by a hard foreign key.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from m3resp.core.events import BreathEvent


@dataclass
class LinkedBreath:
    """One physiological breath, matched across any number of modalities."""

    breaths: dict[str, BreathEvent] = field(default_factory=dict)
    time_tolerance: float = 0.5
    confidence: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def modalities(self) -> list[str]:
        """Which modalities contributed a breath to this link, e.g. ``["eit", "emg"]``."""

        return list(self.breaths)
