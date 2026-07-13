"""Typed collections for `Signal`/`ParameterResult`/`QualityFlag` (plan_stage2.md
Sec 6, Milestone 2.2).

Built now rather than in Milestone 2.1: `M3Session` is the first real
consumer that needs list-like containers with query helpers, and
`plan/stage2_consolidation.md` calls for adding collections only once
something actually needs them.

`Event`/`Breath` are deliberately not given a collection type here: they
already have one, `session.events` (a `dict[str, list[BreathEvent]]`, see
`M3Session.add_events`/`get_events`), which predates this milestone and is
depended on throughout Stage 1. Introducing a second container would fork
that API rather than reconcile with it.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from typing import Any

from m3resp.data.parameters import ParameterResult
from m3resp.data.quality import QualityFlag
from m3resp.data.signals import Signal


@dataclass
class SignalCollection:
    """An ordered collection of `Signal` objects, queryable by modality."""

    items: list[Signal] = field(default_factory=list)

    def add(self, signal: Signal) -> Signal:
        """Append `signal`, unless this exact object is already present.

        Identity (not value) equality: re-adding the same `Signal` instance -
        e.g. if it gets assembled twice on an export path - is a no-op, while
        two distinct results that happen to compare equal both get added.
        """

        if not any(existing is signal for existing in self.items):
            self.items.append(signal)
        return signal

    def for_modality(self, modality: str) -> list[Signal]:
        return [signal for signal in self.items if signal.modality == modality]

    def to_manifest_rows(self) -> list[dict[str, Any]]:
        return [signal.to_manifest_row() for signal in self.items]

    def __iter__(self) -> Iterator[Signal]:
        return iter(self.items)

    def __len__(self) -> int:
        return len(self.items)


@dataclass
class ParameterResultCollection:
    """An ordered collection of `ParameterResult` objects."""

    items: list[ParameterResult] = field(default_factory=list)

    def add(self, parameter: ParameterResult) -> ParameterResult:
        """Append `parameter`, unless this exact object is already present
        (see `SignalCollection.add`)."""

        if not any(existing is parameter for existing in self.items):
            self.items.append(parameter)
        return parameter

    def for_modality(self, modality: str) -> list[ParameterResult]:
        return [p for p in self.items if p.modality == modality]

    def for_name(self, name: str) -> list[ParameterResult]:
        return [p for p in self.items if p.name == name]

    def to_rows(self) -> list[dict[str, Any]]:
        return [p.to_dict() for p in self.items]

    def __iter__(self) -> Iterator[ParameterResult]:
        return iter(self.items)

    def __len__(self) -> int:
        return len(self.items)


@dataclass
class QualityReport:
    """An ordered collection of `QualityFlag` results."""

    items: list[QualityFlag] = field(default_factory=list)

    def add(self, flag: QualityFlag) -> QualityFlag:
        self.items.append(flag)
        return flag

    def failed(self) -> list[QualityFlag]:
        return [flag for flag in self.items if not flag.passed]

    def for_modality(self, modality: str) -> list[QualityFlag]:
        return [flag for flag in self.items if flag.modality == modality]

    def to_rows(self) -> list[dict[str, Any]]:
        return [asdict(flag) for flag in self.items]

    def __iter__(self) -> Iterator[QualityFlag]:
        return iter(self.items)

    def __len__(self) -> int:
        return len(self.items)
