"""Configuration models for M3Resp workflows."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class M3Config:
    """Small Stage 1 configuration object."""

    time_unit: str = "seconds"
    default_alignment_method: str = "manual_offset"
