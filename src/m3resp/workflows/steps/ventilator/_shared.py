"""Helpers for recording ventilator calculations and per-breath quality flags."""

from __future__ import annotations

from typing import Any

import numpy as np

from m3resp.adapters.ventilator_adapter import primary_channel
from m3resp.core.exceptions import MissingModalityDataError
from m3resp.workflows.registry import StepArtifact

#: Ventilator loading/quality steps currently go through ReSurfEMGAdapter
#: (loading shares the sEMG's file; Pocc quality assessment wraps
#: resurfemg's quality_assessment module), so they declare this the same way
#: EMG steps do.
_RESURFEMG = ("resurfemg",)

_SESSION_ARTIFACT = StepArtifact(
    name="session",
    artifact_type="m3session",
    default_context_key="session",
    description="Backing M3Session the step reads from and/or records provenance onto.",
    public=False,
)


def _upstream_metadata(
    *,
    source_function: str,
    operation: str,
    parameters: dict[str, Any],
    source_package: str = "resurfemg",
    implementation: str = "upstream_adapter",
) -> dict[str, Any]:
    """Build the shared provenance metadata schema (see
    `m3resp.workflows.steps.emg._shared._upstream_metadata`).

    `source_package`/`implementation` default to the ReSurfEMG-adapter case
    (`ventilator.pocc_quality`); pass `source_package="m3resp"`,
    `implementation="m3resp.processing.<module>"` for a step whose value comes
    from a native primitive instead (`ventilator.pocc_intervals`/
    `.pocc_time_product`).
    """

    return {
        "source_package": source_package,
        "source_function": source_function,
        "implementation": implementation,
        "parameters": parameters,
        "operation": operation,
    }


def _airway_pressure(ventilator_signals: Any, step_name: str) -> tuple[np.ndarray, str]:
    """Read the bundle's main airway-pressure array and its unit.

    Uses the primary channel mapping, including qualified channel keys. Units
    come from the per-channel units mapping, then the bundle's unit field, then
    "cmH2O". Values are converted to a float array in those units.

    Args:
        ventilator_signals (Any): Ventilator channel bundle.
        step_name (str): Workflow step name included in error messages.

    Returns:
        tuple[numpy.ndarray, str]: Pressure samples and their unit.

    Raises:
        MissingModalityDataError: If the bundle lacks airway-pressure values.
    """

    key = primary_channel(ventilator_signals, "airway_pressure")
    values = ventilator_signals.get(key) if key is not None else None
    if values is None:
        raise MissingModalityDataError(
            f"{step_name} needs an 'airway_pressure' channel; ask "
            "ventilator.channels for it (e.g. airway_pressure_channel=0)."
        )
    units = ventilator_signals.get("units") or {}
    unit = units.get(key) or ventilator_signals.get("unit") or "cmH2O"
    return np.asarray(values, dtype=float), str(unit)
