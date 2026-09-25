"""Load/preprocess/postprocess orchestration methods of `ReSurfEMGAdapter`."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from m3resp.core.events import BreathEvent, coerce_breath_events
from m3resp.core.exceptions import OptionalDependencyError, UnsupportedWorkflowError
from m3resp.data import ParameterResult, QualityFlag, Signal

# from m3resp.emg.io.loaders import load
from m3resp.processing.quality import quality_flag_from_result, skipped_quality_flag

from .shared import (
    _POSTPROCESSING_MODULES,
    POSTPROCESSING_FUNCTIONS,
    _as_parameter_value,
    _computed_category,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from m3resp.data.signals import ProcessingState

    from .protocols import _DefaultsProtocol


class _CoreMixin:
    # def __init__(self):#, loader: Callable[..., Any] | None = None):
    # self._loader = loader

    # def load(self, path: str, **kwargs: Any) -> Any:
    #     # """Load EMG data through m3resp or an injected loader."""
    #     # if self._loader is not None:
    #     #     return self._loader(path, **kwargs)

    #     return load(path, **kwargs)

    # TODO
    def preprocess(self, signal: Any, **kwargs: Any) -> Any:
        """Preprocess EMG data through ReSurfEMG or a provided callable."""
        preprocess = kwargs.pop("preprocess", None)
        if preprocess is not None:
            return preprocess(signal, **kwargs)

        return cast("_DefaultsProtocol", self)._preprocess_default(signal, **kwargs)

    # TODO
    def detect_breaths(self, signal: Any, **kwargs: Any) -> list[BreathEvent]:
        """Detect EMG breaths and normalize them into `BreathEvent` objects."""
        detector = kwargs.pop("detector", None)
        if detector is not None:
            detections = detector(signal, **kwargs)
            return coerce_breath_events(
                detections,
                modality="emg",
                source="resurfemg",
            )

        detections = cast("_DefaultsProtocol", self)._detect_breaths_default(
            signal, **kwargs
        )
        return coerce_breath_events(
            detections,
            modality="emg",
            source="resurfemg.detect_emg_breaths",
        )

    # TODO
    def compute_features(
        self, signal: Any, events: Sequence[BreathEvent], **kwargs: Any
    ) -> Any:
        """Compute EMG features when an upstream callable is provided."""
        compute = kwargs.pop("compute", None)
        if compute is None:
            msg = (
                "EMG feature extraction needs an upstream callable in Stage 1. "
                "Pass `compute=callable`."
            )
            raise UnsupportedWorkflowError(msg)
        return compute(signal, events, **kwargs)

    # TODO
    def to_signals(self, processed_emg: dict[str, Any]) -> list[Signal]:
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

    # TODO
    def to_parameters(self, postprocessed: dict[str, Any]) -> list[ParameterResult]:
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

    # TODO
    def to_quality_flags(self, postprocessed: dict[str, Any]) -> list[QualityFlag]:
        """Convert computed EMG quality-assessment results into `QualityFlag`
        objects.

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

    # TODO
    def available_postprocessing(self) -> dict[str, list[str]]:
        """Return ReSurfEMG postprocessing functions exposed by M3Resp."""
        return {
            category: list(functions)
            for category, functions in POSTPROCESSING_FUNCTIONS.items()
        }

    # TODO
    def postprocess(
        self,
        processed_emg: Any,
        events: Sequence[BreathEvent] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Run default EMG postprocessing and report unavailable input needs."""
        custom = kwargs.pop("postprocess", None)
        if custom is not None:
            return custom(processed_emg, events=events, **kwargs)

        return cast("_DefaultsProtocol", self)._postprocess_default(
            processed_emg, events=events, **kwargs
        )

    # TODO
    def run_postprocessing_function(
        self, category: str, function_name: str, *args: Any, **kwargs: Any
    ) -> Any:
        """Call any exposed `resurfemg.postprocessing` function by name."""
        if function_name not in POSTPROCESSING_FUNCTIONS.get(category, ()):
            msg = (
                f"Unknown ReSurfEMG postprocessing function {category}.{function_name}."
            )
            raise ValueError(msg)

        try:
            module = import_module(_POSTPROCESSING_MODULES[category])
        except ImportError as exc:
            msg = (
                "EMG postprocessing requires the optional dependency `resurfemg`. "
                'Install with `pip install "m3resp[emg]"`.'
            )
            raise OptionalDependencyError(msg) from exc

        return getattr(module, function_name)(*args, **kwargs)
