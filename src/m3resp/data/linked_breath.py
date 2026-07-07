"""``LinkedBreath``: one respiratory cycle observed across modalities
(plan_stage2.md Sec 20, Milestone 2.5).

Produced by ``m3resp.synchronization.linking.link_breaths_by_time``, which
matches ``Breath``/``BreathEvent`` objects across EIT, EMG, and ventilator
event lists by how close their times are, rather than by a hard foreign key.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from m3resp.core.events import BreathEvent


@dataclass
class LinkedBreath:
    """One physiological breath, matched across up to three modalities."""

    eit_breath: BreathEvent | None = None
    emg_breath: BreathEvent | None = None
    ventilator_breath: BreathEvent | None = None
    time_tolerance: float = 0.5
    confidence: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def modalities(self) -> list[str]:
        """Which modalities contributed a breath to this link, e.g. ``["eit", "emg"]``."""

        return [
            modality
            for modality, breath in (
                ("eit", self.eit_breath),
                ("emg", self.emg_breath),
                ("ventilator", self.ventilator_breath),
            )
            if breath is not None
        ]
