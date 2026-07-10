"""Registered session-level pipeline steps.

These wrap ``M3Session`` cross-modality operations so multimodal workflows can be
expressed as a declarative pipeline spec.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from m3resp.core.session import M3Session
from m3resp.workflows.registry import register_step


@register_step(
    "session.sync_raw",
    reads={"session": "session"},
    writes=("sync_summary",),
    summary="Crop raw modality signals by a manual time offset before processing.",
)
def sync_raw(
    session: M3Session,
    *,
    method: str = "manual_offset",
    offset_seconds: float | Mapping[str, float] = 0.0,
    reference_modality: str | None = None,
) -> dict[str, Any]:
    summary = session.synchronize_raw_modalities(
        method=method,
        offset_seconds=offset_seconds,
        reference_modality=reference_modality,
    )
    return {"sync_summary": summary}
