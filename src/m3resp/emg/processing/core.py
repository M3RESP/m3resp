"""Load/preprocess/postprocess orchestration methods of `ReSurfEMG`."""

from __future__ import annotations

from collections.abc import Mapping
from importlib import import_module
from importlib.util import find_spec
from typing import TYPE_CHECKING, Any

import numpy as np

from m3resp.core.exceptions import (
    InvalidM3respModuleError,
    OptionalDependencyError,
    UnsupportedWorkflowError,
)
from m3resp.data import ParameterResult, QualityFlag, Signal
from m3resp.data.events import BreathEvent, coerce_breath_events
from m3resp.processing.quality import quality_flag_from_result, skipped_quality_flag

from .defaults import _detect_breaths_default, _postprocess_default, _preprocess_default
from .shared import (
    POSTPROCESSING_FUNCTIONS,
    _as_parameter_value,
    _computed_category,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from m3resp.data.signals import ProcessingState


def available_postprocessing() -> dict[str, list[str]]:
    """Return ReSurfEMG postprocessing functions exposed by M3Resp."""
    return {
        category: list(functions)
        for category, functions in POSTPROCESSING_FUNCTIONS.items()
    }


def preprocess(signal: Mapping, **kwargs) -> dict[str, Any]:
    """Preprocess EMG data through ReSurfEMG or a provided callable."""
    preprocess = kwargs.pop("preprocess", None)
    if preprocess is not None:
        return preprocess(signal, **kwargs)

    return _preprocess_default(signal, **kwargs)


def detect_breaths(signal: Mapping, **kwargs) -> list[BreathEvent]:
    """Detect EMG breaths and normalize them into `BreathEvent` objects."""
    detector = kwargs.pop("detector", None)
    if detector is not None:
        detections = detector(signal, **kwargs)
        return coerce_breath_events(
            detections,
            modality="emg",
            source="resurfemg",
        )

    detections = _detect_breaths_default(signal, **kwargs)
    return coerce_breath_events(
        detections,
        modality="emg",
        source="resurfemg.detect_emg_breaths",
    )


def compute_features(signal: Mapping, events: Sequence[BreathEvent], **kwargs) -> Any:
    """Compute EMG features when an upstream callable is provided."""
    compute = kwargs.pop("compute", None)
    if compute is None:
        msg = (
            "EMG feature extraction needs an upstream callable in Stage 1. "
            "Pass `compute=callable`."
        )
        raise UnsupportedWorkflowError(msg)
    return compute(signal, events, **kwargs)


def to_signals(processed_emg: dict[str, Any]) -> list[Signal]:
    """Convert preprocessed EMG channel arrays into `Signal` objects."""
    if not isinstance(processed_emg, dict) or "fs" not in processed_emg:
        msg = "to_signals expects processed EMG data from preprocess_emg()."
        raise UnsupportedWorkflowError(msg)

    fs = float(processed_emg["fs"])
    channel = processed_emg.get("channel")
    channel_name = str(channel) if channel is not None else None

    signals: list[Signal] = []
    channel_sources: list[tuple[str, str, ProcessingState]] = [
        ("raw_channel", "raw_channel", "raw"),
        ("filtered", "filtered", "intermediate"),
        ("envelope", "envelope", "processed"),
    ]
    for key, name, processing_state in channel_sources:
        array = processed_emg.get(key)
        if array is None:
            continue
        array = np.asarray(array, dtype=float)
        time = np.arange(array.shape[0], dtype=float) / fs
        signals.append(
            Signal(
                values=array,
                time=time,
                sample_frequency=fs,
                name=name,
                modality="emg",
                # Raw trace and envelope are both electrical potentials -
                # what differs between them is processing_state, not the
                # physical quantity.
                category="electrical_potential",
                channel=channel_name,
                processing_state=processing_state,
            )
        )
    return signals


def to_parameters(postprocessed: dict[str, Any]) -> list[ParameterResult]:
    """Convert computed EMG features into `ParameterResult` objects."""
    features = _computed_category(postprocessed, "features")
    return [
        ParameterResult(
            name=name,
            value=_as_parameter_value(value),
            modality="emg",
            method=f"resurfemg.{name}",
            metadata={
                "source_method": f"resurfemg.{name}",
                "implementation": "m3resp.processing.metrics",
            },
        )
        for name, value in features.items()
    ]


def to_quality_flags(postprocessed: dict[str, Any]) -> list[QualityFlag]:
    """Convert EMG quality-assessment results into `QualityFlag` objects.

    ReSurfEMG's ``quality_assessment`` functions return heterogeneous
    shapes (booleans, SNR floats, per-breath arrays), so this performs a
    best-effort structural conversion rather than clinical judgment:
    boolean-like results become pass/fail, everything else is recorded as
    an informational flag with its scalar value attached where possible.
    Functions skipped for missing inputs (`postprocessed["skipped"]`)
    become failed, ``warning``-severity flags.
    """
    if not isinstance(postprocessed, dict):
        return []

    quality_results = _computed_category(postprocessed, "quality_assessment")
    flags = [
        quality_flag_from_result(
            name=name,
            result=value,
            modality="emg",
            source_method=f"resurfemg.{name}",
        )
        for name, value in quality_results.items()
    ]
    for name, reason in postprocessed.get("skipped", {}).items():
        flags.append(
            skipped_quality_flag(
                name=name,
                modality="emg",
                reason=str(reason),
            )
        )
    return flags


def postprocess(
    processed_emg: dict[str, Any],
    events: Sequence[BreathEvent] | None = None,
    **kwargs,
) -> dict[str, Any]:
    """Run default EMG postprocessing and report unavailable input needs."""
    custom = kwargs.pop("postprocess", None)
    if custom is not None:
        return custom(processed_emg, events=events, **kwargs)

    return _postprocess_default(processed_emg, events=events, **kwargs)


def run_postprocessing_function(
    category: str, function_name: str, *args: Any, **kwargs
) -> Any:
    """Call any exposed postprocessing function by name."""

    def _import_m3resp_module_or_class(object_to_import: str) -> object:
        name_parts = object_to_import.split(".")

        def _iter_module_parts(sub_package: str) -> tuple[object, str]:
            try:
                find_spec(".".join(name_parts))
                return getattr(import_module(".".join(name_parts)), sub_package[1:])
            except (ModuleNotFoundError, AttributeError) as exc:
                sub_package = "." + name_parts.pop() + sub_package
                if not name_parts:
                    msg_0 = f"Could not find {object_to_import}."
                    raise InvalidM3respModuleError(msg_0) from exc
                return _iter_module_parts(sub_package), sub_package

        return _iter_module_parts("")[0]

    if function_name not in POSTPROCESSING_FUNCTIONS.get(category, ()):
        msg = f"Unknown ReSurfEMG postprocessing function {category}.{function_name}."
        raise ValueError(msg)

    try:
        m3resp_module = _import_m3resp_module_or_class(
            POSTPROCESSING_FUNCTIONS[category][function_name][0]
        )
        m3resp_function = getattr(
            m3resp_module, POSTPROCESSING_FUNCTIONS[category][function_name][1]
        )
    except ImportError as exc:
        msg = (
            "EMG postprocessing requires the optional dependency `resurfemg`. "
            'Install with `pip install "m3resp[emg]"`.'
        )
        raise OptionalDependencyError(msg) from exc
    return m3resp_function(*args, **kwargs)


class _CoreMixin:
    def __init__(self, loader: Callable[..., Any] | None = None):
        self._loader = loader

    preprocess = staticmethod(preprocess)
    detect_breaths = staticmethod(detect_breaths)
    compute_features = staticmethod(compute_features)
    to_signals = staticmethod(to_signals)
    to_parameters = staticmethod(to_parameters)
    to_quality_flags = staticmethod(to_quality_flags)
    postprocess = staticmethod(postprocess)
    run_postprocessing_function = staticmethod(run_postprocessing_function)
    available_postprocessing = staticmethod(available_postprocessing)
