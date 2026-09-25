"""The central Stage 1 M3Resp session object."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from m3resp.adapters.eitprocessing_adapter import EITProcessingAdapter
from m3resp.adapters.resurfemg_adapter import ReSurfEMGAdapter
from m3resp.adapters.ventilator_adapter import VentilatorAdapter, primary_channel
from m3resp.core.events import BreathEvent
from m3resp.core.exceptions import MissingModalityDataError, VariantAlreadyExistsError
from m3resp.core.metadata import SessionMetadata
from m3resp.core.provenance import ProvenanceRecord, record
from m3resp.data.collections import (
    ParameterResultCollection,
    QualityReport,
    SignalCollection,
)
from m3resp.data.linked_breath import LinkedBreath
from m3resp.data.parameters import ParameterResult
from m3resp.data.processing import ProcessingHistory
from m3resp.export.session_export import export_session_summary
from m3resp.modalities.eit import EITRecording, frame_window, keep_frames
from m3resp.modalities.eit import load as load_eit_recording
from m3resp.modalities.emg import EMGRecording
from m3resp.modalities.emg import keep_samples as keep_emg_samples
from m3resp.modalities.emg import load as load_emg_recording
from m3resp.modalities.emg import sample_window as emg_sample_window
from m3resp.modalities.names import VENTILATOR, normalize_modality
from m3resp.modalities.ventilator import (
    DEFAULT_VENTILATOR_NAME,
    VentilatorRecording,
    cut_seconds_off_ends,
    ventilator_clock,
    ventilator_payload,
    ventilator_raw,
    ventilator_recordings,
)
from m3resp.modalities.ventilator import keep_samples as keep_ventilator_samples
from m3resp.modalities.ventilator import load as load_ventilator_recording
from m3resp.modalities.ventilator import sample_window as ventilator_sample_window
from m3resp.synchronization.alignment import (
    align_events_by_modality_offset,
    normalize_offset_key,
    offsets_relative_to_reference,
    resolve_alignment_offsets,
    ventilator_start_key,
)
from m3resp.synchronization.linking import link_breaths_by_time
from m3resp.synchronization.multimodal_parameters import compute_multimodal_parameters
from m3resp.synchronization.raw_traces import raw_synchronization_traces
from m3resp.synchronization.start_times import (
    loaded_modalities,
    recording_start_time,
    shared_clock_shift,
    shared_clock_shifts,
    shift_trace,
)
from m3resp.synchronization.ventilator import (
    infer_ventilator_duration,
    infer_ventilator_fs,
    iter_ventilator_detections,
    normalize_ventilator_breath,
)

if TYPE_CHECKING:
    from m3resp.datamodel.recorder import DataModelRecorder

ALIGNMENT_EVENT_LISTS = {
    "eit": "eit_breaths",
    "emg": "emg_breaths",
    VENTILATOR: "ventilator_breaths",
}


#: `DEFAULT_VENTILATOR_NAME` (imported above) is the name the first ventilator
#: recording is filed under when the caller does not give one. It is also the
#: recording `session.ventilator` and `session.raw["ventilator"]` point at, so
#: single-recording code is unchanged.


def set_ventilator_raw(raw: dict[str, Any], recording: Any) -> None:
    """Store a ventilator recording under both `session.raw` keys.

    ``"ventilator"`` is canonical; ``"vent"`` is the key Stage 1 shipped and is
    still read by existing notebooks and specs. Both reference the *same*
    object, and slicing changes the underlying payload in place, so the two
    views can never drift apart.

    `M3Session.load_ventilator` passes a `VentilatorRecording` here, matching
    what `raw["eit"]`/`raw["emg"]` hold. A bare payload dict (what Stage 1
    stored) is still accepted, and
    `m3resp.modalities.ventilator.ventilator_payload` unwraps either shape.
    """

    raw[VENTILATOR] = recording
    raw["vent"] = recording


class M3Session:
    """Small, explicit session object for Stage 1 multimodal workflows."""

    def __init__(
        self,
        eit_adapter: EITProcessingAdapter | None = None,
        emg_adapter: ReSurfEMGAdapter | None = None,
        metadata: SessionMetadata | dict[str, Any] | None = None,
        allow_overwrite: bool = False,
        ventilator_adapter: Any | None = None,
    ):
        self.eit_adapter = eit_adapter or EITProcessingAdapter()
        self.emg_adapter = emg_adapter or ReSurfEMGAdapter()
        # Ventilator processing is native (`VentilatorAdapter` wraps no upstream
        # library), but *loading* borrows whichever adapter owns the file the
        # ventilator channels arrived in: the sEMG's multi-channel export, or
        # the EIT `*.bin` that stores ventilator waveforms beside its impedance
        # frames. Wiring both through this session's own adapters means a
        # loader injected for EMG or EIT covers the ventilator path too, with
        # no second injection. Pass `ventilator_adapter=` to separate them.
        self.ventilator_adapter = ventilator_adapter or VentilatorAdapter(
            loader=lambda path, **kwargs: self.emg_adapter.load(path, **kwargs),
            eit_loader=lambda path, **kwargs: self.eit_adapter.load(path, **kwargs),
        )

        self.eit: EITRecording | None = None
        self.emg: EMGRecording | None = None
        self.ventilator: VentilatorRecording | None = None
        # Ventilator data can arrive from several instruments at once - a
        # ventilator export and the EIT `*.bin`'s own Medibus channels both
        # carry an airway pressure, and they are different measurements. Each
        # loaded recording is filed under a name here; `self.ventilator` is the
        # primary one, so code that only ever loads one is unaffected.
        self.ventilators: dict[str, VentilatorRecording] = {}
        self.raw: dict[str, Any] = {}
        # Where each recording's first sample sits on the shared clock, in
        # seconds, set by `synchronize_raw_modalities`. Empty means every
        # recording is taken to have started at the same moment. See
        # `m3resp.synchronization.start_times`.
        self.start_times: dict[str, float] = {}
        self.processed: dict[str, Any] = {}
        # Named alternate preprocessing results, e.g. for algorithms that
        # need the same raw recording preprocessed differently (see
        # preprocess_eit`/`preprocess_emg`'s `variant` parameter).
        self.processed_variants: dict[str, dict[str, Any]] = {
            "eit": {},
            "emg": {},
            VENTILATOR: {},
        }
        # Session-wide default for preprocess_eit/preprocess_emg's `overwrite`
        # kwarg, so notebook/exploratory code can set this once instead of
        # passing `overwrite=True` on every call. Left off by default so code
        # copied into reusable/production paths is safe unless it opts in
        # explicitly, per-call, or here.
        self.allow_overwrite = allow_overwrite
        self.events: dict[str, Any] = {}
        self.parameters: dict[str, Any] = {}
        # Milestone 2.2 (plan/plan_stage2.md Sec 14): typed collections that
        # let EIT and EMG data live in the same structure, populated from the
        # default preprocess/postprocess paths via each adapter's
        # to_signals/to_parameters/to_quality_flags. These are additive: the
        # `raw`/`processed`/`parameters` dicts above keep their Stage 1 shape
        # and behavior unchanged.
        self.signals = SignalCollection()
        self.parameter_results = ParameterResultCollection()
        self.quality = QualityReport()
        # Milestone 2.5 (plan/plan_stage2.md Sec 20): breaths matched across
        # modalities by `link_breaths`, once per-modality breath events exist.
        self.linked_breaths: list[LinkedBreath] = []
        self.metadata = _coerce_metadata(metadata)
        self.provenance: list[ProvenanceRecord] = []
        # Stage 2 pipeline-structure Phase 5.1: a universal, engine-populated
        # log of every executed workflow step (name/bindings/parameters/
        # timing), independent of whether any step function calls
        # `self._record()` itself. Distinct from `provenance` (the older,
        # session-method-level "action + modality" log) and from the
        # datamodel's per-pipeline `ProcessingRun` (see
        # `m3resp.workflows.engine.run_pipeline` and
        # `DataModelRecorder.record_pipeline_result`).
        self.processing_history = ProcessingHistory()
        # Stage 2 data model wrapper (opt-in, see m3resp.datamodel). ``None``
        # leaves Stage 1 behavior completely unchanged.
        self.datamodel: DataModelRecorder | None = None

    def load_eit(
        self, path: str | Path, vendor: str | None = None, **kwargs: Any
    ) -> Any:
        """Load EIT data and store it under `raw["eit"]`."""

        recording = load_eit_recording(
            path,
            vendor=vendor,
            adapter=self.eit_adapter,
            **kwargs,
        )
        self.eit = recording
        self.raw["eit"] = recording
        self._record("load_eit", "eit", path=str(path), vendor=vendor)
        return recording.data

    def load_emg(self, path: str | Path, **kwargs: Any) -> Any:
        """Load EMG data and store it under `raw["emg"]`."""

        recording = load_emg_recording(path, adapter=self.emg_adapter, **kwargs)
        self.emg = recording
        self.raw["emg"] = recording
        self._record("load_emg", "emg", path=str(path))
        return recording.data

    def load_ventilator(
        self, path: str | Path, *, name: str | None = None, **kwargs: Any
    ) -> Any:
        """Load ventilator data and store it under `raw["ventilator"]`.

        Mirrors `load_eit`/`load_emg`. The recording is additionally stored
        under the legacy `raw["vent"]` key, pointing at the same object.

        `path` may be either file ventilator data arrives in: the multi-channel
        export shared with the sEMG, or an EIT ``*.bin`` carrying ventilator
        waveforms beside its impedance frames. `VentilatorAdapter` picks by
        suffix; pass ``source="eit"``/``"emg"`` to force one, and
        ``ventilator_channels=`` to select which channels to read from a
        ``*.bin`` (see `m3resp.adapters.ventilator_adapter`).

        `name` files this recording alongside any already loaded, for a study
        where more than one instrument recorded ventilator data - a ventilator
        export and the EIT file's own Medibus channels, say, each with its own
        airway pressure. Without a name the recording is the primary one, which
        is what `session.ventilator` and `raw["ventilator"]` point at.
        Preprocess a named recording with
        `preprocess_ventilator(name=...)`, which qualifies its channel keys so
        the two airway pressures stay distinct in `session.signals`.
        """

        recording = load_ventilator_recording(
            path, adapter=self.ventilator_adapter, **kwargs
        )
        key = name or DEFAULT_VENTILATOR_NAME
        self.ventilators[key] = recording
        if key == DEFAULT_VENTILATOR_NAME or self.ventilator is None:
            self.ventilator = recording
            set_ventilator_raw(self.raw, recording)
        self._record("load_ventilator", VENTILATOR, path=str(path), name=key)
        return recording.data

    def primary_ventilator_name(self) -> str | None:
        """The name of the recording `session.ventilator` points at."""

        if DEFAULT_VENTILATOR_NAME in self.ventilators:
            return DEFAULT_VENTILATOR_NAME
        return next(iter(self.ventilators), None)

    def get_ventilator(self, name: str | None = None) -> VentilatorRecording:
        """A loaded ventilator recording by name, or the primary one."""

        key = name or self.primary_ventilator_name()
        if key is None or key not in self.ventilators:
            known = sorted(self.ventilators)
            raise MissingModalityDataError(
                f"No ventilator recording named {key!r}. "
                f"Loaded: {known or 'none'}. Call load_ventilator(path"
                f"{', name=...' if known else ''}) first."
            )
        return self.ventilators[key]

    def preprocess_eit(
        self,
        *,
        variant: str | None = None,
        overwrite: bool = False,
        **kwargs: Any,
    ) -> Any:
        """Run a provided or upstream EIT preprocessing function.

        Every result is stored under `session.processed_variants["eit"][name]`,
        `name` being `variant` if given, otherwise `"default"` - there is no
        implicit, ambiguously-overwritten "current" result. Writing to a name
        that's already populated raises `VariantAlreadyExistsError` unless
        `overwrite=True` is passed (or `session.allow_overwrite = True` is
        set, so notebook/exploratory code can opt in once instead of passing
        `overwrite=True` on every call), so a reference like
        `processed_variants["eit"]["mdn"]` can't silently change meaning
        underneath a caller that stashed it earlier. `session.processed["eit"]`
        mirrors the `"default"` variant only, for convenience/backwards
        compatibility with code that just wants "the" EIT result.

        `preprocess_eit(filter_mode="mdn", variant="mdn")` and
        `preprocess_eit(filter_mode="lowpass", variant="lowpass")` can both
        coexist. See `detect_eit_breaths(variant=...)` to detect breaths
        against a specific variant.
        """

        recording = self._require_raw("eit")
        name = variant if variant is not None else "default"
        if (
            not (overwrite or self.allow_overwrite)
            and name in self.processed_variants["eit"]
        ):
            raise VariantAlreadyExistsError(
                f"EIT preprocessing variant {name!r} already exists; pass "
                "a different `variant=`, or `overwrite=True` to replace it."
            )
        preprocess = kwargs.pop("preprocess", None)
        if preprocess is None:
            result = self.eit_adapter.preprocess(recording.data, **kwargs)
            self._extend_typed_collections_from_eit(result)
        else:
            # A custom `preprocess` callable's output shape is not guaranteed
            # to match what `EITProcessingAdapter.to_signals/to_parameters/
            # to_quality_flags` expect, so the typed collections are only
            # populated on the default adapter path.
            result = preprocess(recording.data, **kwargs)
        self.processed_variants["eit"][name] = result
        if name == "default":
            self.processed["eit"] = result
        self._record("preprocess_eit", "eit", variant=variant, **kwargs)
        return result

    def preprocess_emg(
        self,
        *,
        variant: str | None = None,
        overwrite: bool = False,
        **kwargs: Any,
    ) -> Any:
        """Run EMG preprocessing through the adapter.

        See `preprocess_eit` for what `variant`/`overwrite`/`allow_overwrite`
        do - it persists this result under
        `session.processed_variants["emg"][name]`, raising
        `VariantAlreadyExistsError` if `name` is already populated, and
        mirrors it onto `session.processed["emg"]` only when `name` is
        `"default"`.
        """

        recording = self._require_raw("emg")
        name = variant if variant is not None else "default"
        if (
            not (overwrite or self.allow_overwrite)
            and name in self.processed_variants["emg"]
        ):
            raise VariantAlreadyExistsError(
                f"EMG preprocessing variant {name!r} already exists; pass "
                "a different `variant=`, or `overwrite=True` to replace it."
            )
        result = self.emg_adapter.preprocess(recording.data, **kwargs)
        if self.emg is not None and isinstance(result, dict):
            self.emg.filtered = result.get("filtered")
            self.emg.ecg_cleaned = result.get("ecg_cleaned")
            self.emg.envelope = result.get("envelope")
            self.emg.channel = result.get("channel")
            self.emg.fs = result.get("fs")
        for signal in self.emg_adapter.to_signals(result):
            self.signals.add(signal)
        self.processed_variants["emg"][name] = result
        if name == "default":
            self.processed["emg"] = result
        self._record("preprocess_emg", "emg", variant=variant, **kwargs)
        return result

    def preprocess_ventilator(
        self,
        *,
        name: str | None = None,
        variant: str | None = None,
        overwrite: bool = False,
        **kwargs: Any,
    ) -> Any:
        """Split and filter the ventilator channels through the adapter.

        Splits the recording into its pressure, flow and volume channels and
        low-passes each one. See `preprocess_eit` for what
        `variant`/`overwrite`/`allow_overwrite` do - the result persists under
        `session.processed_variants["ventilator"][variant]`, an already
        populated variant raises `VariantAlreadyExistsError` unless one of the
        two overwrite flags is set, and the result is mirrored onto
        `session.processed["ventilator"]` only when the variant is
        `"default"`.

        `name` selects which loaded recording to preprocess when a study
        recorded ventilator data on more than one instrument (see
        `load_ventilator`). A non-primary recording's channel keys are
        qualified with its name - ``pressure__pod`` rather than ``pressure`` -
        so its airway pressure does not collide with the primary recording's
        in `session.signals`. `variant` defaults to `name`, so each recording
        lands in its own slot rather than overwriting.

        Unlike its EIT/EMG siblings this runs native code rather than an
        upstream library: nothing in `eitprocessing`/`resurfemg` preprocesses
        ventilator data, which is why these channels used to be consumed
        unfiltered. Remaining keyword arguments reach
        `VentilatorAdapter.preprocess`: `lowpass_hz` sets the cut-off, with
        `lowpass_hz=None` skipping the filter, and `preprocess` replaces the
        whole step with a callable of your own. See
        `m3resp.adapters.ventilator_adapter` for the defaults.
        """

        primary = self.primary_ventilator_name()
        if name is None or name == primary:
            recording = self._require_raw(VENTILATOR)
            target = self.ventilator
        else:
            recording = self.get_ventilator(name)
            target = recording
            # A second instrument's airway pressure is a different
            # measurement from the first's, so its channels are named apart.
            kwargs.setdefault("origin", name)
            kwargs.setdefault("qualify", True)

        variant_name = variant if variant is not None else (name or "default")
        if (
            not (overwrite or self.allow_overwrite)
            and variant_name in self.processed_variants[VENTILATOR]
        ):
            raise VariantAlreadyExistsError(
                f"Ventilator preprocessing variant {variant_name!r} already "
                "exists; pass a different `variant=`, or `overwrite=True` to "
                "replace it."
            )
        result = self.ventilator_adapter.preprocess(recording, **kwargs)
        if target is not None and isinstance(result, dict):
            target.pressure = result.get(primary_channel(result, "pressure") or "")
            target.flow = result.get(primary_channel(result, "flow") or "")
            target.volume = result.get(primary_channel(result, "volume") or "")
            target.fs = result.get("fs")
        for signal in self.ventilator_adapter.to_signals(result):
            self.signals.add(signal)
        for parameter in self.ventilator_adapter.to_parameters(result):
            self.parameter_results.add(parameter)
        for flag in self.ventilator_adapter.to_quality_flags(result):
            self.quality.add(flag)
        self.processed_variants[VENTILATOR][variant_name] = result
        # `session.processed` means "the primary recording's result", so mirror
        # it whether that recording was reached by default or asked for by its
        # own name. Testing only for `"default"` missed the second case, and
        # `detect_ventilator_breaths` then silently re-split the raw recording
        # with default settings instead of using what was preprocessed here.
        if variant_name in ("default", primary):
            self.processed[VENTILATOR] = result
        self._record("preprocess_ventilator", VENTILATOR, variant=variant, **kwargs)
        return result

    def synchronize_raw_modalities(
        self,
        method: str = "manual_offset",
        offset_seconds: float | Mapping[str, float] = 0.0,
        reference_modality: str | None = None,
    ) -> dict[str, Any]:
        """Set when each loaded recording started, on one shared clock.

        `offset_seconds` gives each modality's start time in seconds, for
        example ``{"emg": 5.0}`` for "EMG started 5 s after EIT", or
        ``{"emg": -5.0}`` for "EMG started 5 s before EIT". A single number
        is the EMG start time. Start times are counted from the start of
        `reference_modality`, which is therefore always 0.

        ``"ventilator"`` is the start time of the standalone ventilator
        recordings (those loaded with ``source="ventilator"``). When two of
        them started at different moments, give one its own start time with
        ``"ventilator:<name>"``, the name it was loaded under, e.g.
        ``{"ventilator": 0.0, "ventilator:monitor": 12.5}``. Ventilator data
        that came inside the EIT or EMG file always uses that file's start
        time.

        No samples are removed: every recording keeps its full length. The
        start times are stored in `session.start_times` and are added to
        breath times only when modalities are compared
        (`synchronize_multimodal_breaths`, `link_breaths`), so each
        modality's own results stay on its own clock. Calling this again
        replaces the start times rather than adding to them. See
        `m3resp.synchronization.start_times`.

        Returns ``{modality: {"start_time_seconds": ...}}`` for every loaded
        modality, plus a ``"ventilator:<name>"`` entry for each standalone
        ventilator recording given its own start time.
        """

        if method != "manual_offset":
            raise ValueError("Stage 1 supports only method='manual_offset'")

        configured_offsets = resolve_alignment_offsets(offset_seconds)
        self._check_ventilator_start_keys(configured_offsets)
        resolved_reference = self._resolve_raw_alignment_reference(reference_modality)
        offsets = offsets_relative_to_reference(configured_offsets, resolved_reference)
        self.start_times = {
            modality: float(offset) for modality, offset in offsets.items()
        }

        synchronized: dict[str, Any] = {}
        traces: dict[str, Any] = {}
        for modality in loaded_modalities(self):
            start_time = recording_start_time(self, modality)
            synchronized[modality] = {"start_time_seconds": start_time}
            shift = shared_clock_shift(self, modality)
            for trace_name, before in raw_synchronization_traces(
                self, modality
            ).items():
                traces[trace_name] = {
                    "before": before,
                    "after": shift_trace(before, shift),
                    "start_time_seconds": start_time,
                }
        for key in self.start_times:
            if key.startswith(ventilator_start_key("")):
                synchronized[key] = {"start_time_seconds": self.start_times[key]}

        self.processed["raw_synchronization"] = traces
        self.parameters["raw_alignment"] = {
            "method": method,
            "reference_modality": resolved_reference,
            "requested_reference_modality": reference_modality,
            "offset_seconds": offsets,
            "configured_offset_seconds": configured_offsets,
            "start_time_seconds": dict(self.start_times),
            "synchronized_modalities": sorted(synchronized),
        }
        self._record(
            "synchronize_raw_modalities",
            parameters={
                "method": method,
                "reference_modality": resolved_reference,
                "offset_seconds": offsets,
                "configured_offset_seconds": configured_offsets,
                "start_time_seconds": dict(self.start_times),
            },
        )
        return synchronized

    def slice_emg(
        self, start_seconds: float, end_seconds: float | None = None
    ) -> dict[str, Any]:
        """Keep only the part of the loaded EMG recording between two times.

        Times are in seconds from the start of the recording as it is now
        (after any earlier slicing). ``end_seconds=None`` keeps everything up
        to the end. Ventilator channels recorded in the same file (for
        example airway pressure on a Biopac export) are cut the same way, so
        they stay lined up with the EMG. After slicing, EMG times count from
        the new start, and the EMG start time in `session.start_times` moves
        later by `start_seconds`, so the EMG stays lined up with the other
        modalities.

        Run this before `preprocess_emg`. The removed samples are gone from
        the loaded recording; reload the file to get them back.
        """

        recording = self.emg
        data = recording.data if recording is not None else None
        if not isinstance(data, dict) or "array" not in data:
            raise MissingModalityDataError("slice_emg needs a loaded EMG recording.")
        window = emg_sample_window(recording, start_seconds, end_seconds)
        removed_end_seconds = (
            window.n_samples - window.end_index
        ) / window.sample_frequency

        keep_emg_samples(recording, window.start_index, window.end_index)
        # Ventilator channels from the same file share the EMG's clock, so the
        # same stretch of time is cut off both ends of them, counted at their
        # own sampling rate.
        for item in ventilator_recordings(self):
            if ventilator_clock(item) == "emg":
                cut_seconds_off_ends(item, window.shift_seconds, removed_end_seconds)
        self.start_times["emg"] = (
            self.start_times.get("emg", 0.0) + window.shift_seconds
        )
        summary = {
            "start_seconds": window.start_seconds,
            "end_seconds": window.end_seconds,
            "removed_samples_start": window.start_index,
            "removed_samples_end": window.n_samples - window.end_index,
            "kept_samples": window.end_index - window.start_index,
        }
        self.parameters["emg_slice"] = summary
        self._record("slice_emg", "emg", parameters=summary)
        return summary

    def slice_eit(
        self, start_seconds: float, end_seconds: float | None = None
    ) -> dict[str, Any]:
        """Keep only the part of the loaded EIT recording between two times.

        Times are in seconds from the first frame of the recording as it is
        now (after any earlier slicing), not the time of day stored in the
        file. ``end_seconds=None`` keeps everything up to the end. The kept
        part runs from the first frame at or after `start_seconds` up to,
        but not including, the first frame at or after `end_seconds`.

        The cutting itself is done by `eitprocessing`
        (`EITProcessingAdapter.slice_sequence`): pixel data, global
        impedance, the pressure/flow channels and the vendor's markers are
        all cut to the same frames. A ventilator recording loaded from the
        same EIT file is cut at exactly those frames too, so it stays lined
        up with the EIT. EIT frame times keep their original values; the EIT
        start time in `session.start_times` moves later by the part cut off
        the front, so the EIT stays lined up with the other recordings.

        Run this before `preprocess_eit`. The removed frames are gone from
        the loaded recording; reload the file to get them back. Signals
        already added to `session.signals` when loading keep the full
        recording.
        """

        recording = self.eit
        if recording is None or recording.data is None:
            raise MissingModalityDataError("slice_eit needs a loaded EIT recording.")
        if self.processed_variants["eit"]:
            raise ValueError(
                "slice_eit must run before preprocess_eit: the EIT has already "
                "been preprocessed, and those results would still cover the "
                "whole recording. Reload the file, slice, then preprocess."
            )

        window = frame_window(recording, start_seconds, end_seconds)

        # Ventilator data loaded from the same EIT file has one sample per EIT
        # frame, so it is cut at exactly the same frames.
        on_eit_clock = [
            item
            for item in ventilator_recordings(self)
            if ventilator_clock(item) == "eit"
        ]
        for item in on_eit_clock:
            payload = ventilator_payload(item)
            if (
                payload is not None
                and np.asarray(payload["array"]).shape[-1] != window.n_samples
            ):
                raise ValueError(
                    "slice_eit: a ventilator recording from the EIT file has "
                    f"{np.asarray(payload['array']).shape[-1]} samples but the "
                    f"EIT has {window.n_samples} frames, so they cannot be cut "
                    "at the same frames."
                )

        keep_frames(
            recording, window.start_index, window.end_index, adapter=self.eit_adapter
        )
        for item in on_eit_clock:
            keep_ventilator_samples(item, window.start_index, window.end_index)

        self.start_times["eit"] = (
            self.start_times.get("eit", 0.0) + window.shift_seconds
        )
        summary = {
            "start_seconds": window.start_seconds,
            "end_seconds": window.end_seconds,
            "removed_frames_start": window.start_index,
            "removed_frames_end": window.n_samples - window.end_index,
            "kept_frames": window.end_index - window.start_index,
            "ventilator_recordings_cut": len(on_eit_clock),
        }
        self.parameters["eit_slice"] = summary
        self._record("slice_eit", "eit", parameters=summary)
        return summary

    def slice_ventilator(
        self,
        start_seconds: float,
        end_seconds: float | None = None,
        *,
        name: str | None = None,
    ) -> dict[str, Any]:
        """Keep only the part of standalone ventilator recordings between two
        times.

        "Standalone" means a ventilator or monitor export with a clock of its
        own, loaded with ``load_ventilator(path, source="ventilator")``.
        Ventilator data that came inside the EIT or EMG file is on that
        file's clock and is cut with it by `slice_eit` or `slice_emg`;
        asking to cut it here raises an error, since cutting it alone would
        move it out of line with its host recording.

        `name` cuts only the recording loaded under that name. Without it,
        every standalone ventilator recording is cut.

        Times are in seconds from the start of the recording as it is now
        (after any earlier slicing). ``end_seconds=None`` keeps everything up
        to the end. After slicing, the recording's times count from the new
        start, and its start time moves later by `start_seconds`: it is
        stored under its own ``"ventilator:<name>"`` entry in
        `session.start_times`, so the recording stays lined up with the
        other modalities and no other recording moves.

        Run this before `preprocess_ventilator`. The removed samples are
        gone from the loaded recording; reload the file to get them back.
        """

        named = self.ventilators or (
            {DEFAULT_VENTILATOR_NAME: ventilator_raw(self)}
            if ventilator_raw(self) is not None
            else {}
        )
        if not named:
            raise MissingModalityDataError(
                "slice_ventilator needs a loaded ventilator recording."
            )
        if name is not None:
            if name not in named:
                raise MissingModalityDataError(
                    f"No ventilator recording named {name!r}. Loaded: {sorted(named)}."
                )
            named = {name: named[name]}
        standalone = {
            name: recording
            for name, recording in named.items()
            if ventilator_clock(recording) == VENTILATOR
        }
        if not standalone:
            hosts = sorted(
                {ventilator_clock(recording) for recording in named.values()}
            )
            raise ValueError(
                "slice_ventilator: the loaded ventilator data came inside the "
                f"{' and '.join(h.upper() for h in hosts)} file, so it is on "
                "that file's clock. Cut it together with its host recording "
                f"using {' or '.join(f'slice_{h}' for h in hosts)}. To load a "
                "standalone ventilator export, pass source='ventilator' to "
                "load_ventilator."
            )
        if self.processed_variants[VENTILATOR]:
            raise ValueError(
                "slice_ventilator must run before preprocess_ventilator: the "
                "ventilator data has already been preprocessed, and those "
                "results would still cover the whole recording. Reload the "
                "file, slice, then preprocess."
            )

        windows = {
            key: ventilator_sample_window(
                recording, start_seconds, end_seconds, name=key
            )
            for key, recording in standalone.items()
        }
        for key, window in windows.items():
            # Read the start time before storing anything, so a recording that
            # still used the shared "ventilator" entry keeps its value.
            new_start = (
                recording_start_time(self, VENTILATOR, key) + window.shift_seconds
            )
            keep_ventilator_samples(
                standalone[key], window.start_index, window.end_index
            )
            self.start_times[ventilator_start_key(key)] = new_start
        summary = {
            "start_seconds": float(start_seconds),
            "end_seconds": None if end_seconds is None else float(end_seconds),
            "recordings": {
                key: {
                    "removed_samples_start": window.start_index,
                    "removed_samples_end": window.n_samples - window.end_index,
                    "kept_samples": window.end_index - window.start_index,
                }
                for key, window in windows.items()
            },
        }
        self.parameters["ventilator_slice"] = summary
        self._record("slice_ventilator", VENTILATOR, parameters=summary)
        return summary

    def detect_eit_breaths(self, *, variant: str | None = None, **kwargs: Any) -> Any:
        """Detect EIT breaths and store normalized events.

        Pass `variant=<name>` to detect breaths against a
        `preprocess_eit(..., variant=<name>)` result instead of the default
        `processed["eit"]`; the events are then stored under
        `session.events["eit_breaths:<name>"]` instead of
        `session.events["eit_breaths"]`, so multiple variants' detections
        can coexist.
        """

        if variant is not None:
            data = self.processed_variants["eit"].get(variant)
            if data is None:
                raise MissingModalityDataError(
                    f"No EIT preprocessing variant {variant!r}; call "
                    f"preprocess_eit(variant={variant!r}, ...) first."
                )
            event_key = f"eit_breaths:{variant}"
        else:
            data = self.processed.get("eit") or self._require_raw("eit").data
            event_key = "eit_breaths"
        events = self.eit_adapter.detect_breaths(data, **kwargs)
        self.add_events(event_key, events)
        self._record("detect_eit_breaths", "eit", variant=variant, **kwargs)
        return self.events[event_key]

    def detect_emg_breaths(self, *, variant: str | None = None, **kwargs: Any) -> Any:
        """Detect EMG breaths and store normalized events.

        See `detect_eit_breaths` for what `variant` does.
        """

        if variant is not None:
            data = self.processed_variants["emg"].get(variant)
            if data is None:
                raise MissingModalityDataError(
                    f"No EMG preprocessing variant {variant!r}; call "
                    f"preprocess_emg(variant={variant!r}, ...) first."
                )
            event_key = f"emg_breaths:{variant}"
        else:
            data = self.processed.get("emg") or self._require_raw("emg").data
            event_key = "emg_breaths"
        events = self.emg_adapter.detect_breaths(data, **kwargs)
        self.add_events(event_key, events)
        recorded = dict(kwargs)
        if "baseline" in recorded:
            # The baseline is a full-length array; record that one was used,
            # not its samples.
            recorded["baseline"] = recorded["baseline"] is not None
        self._record("detect_emg_breaths", "emg", variant=variant, **recorded)
        return self.events[event_key]

    def detect_ventilator_breaths(
        self, *, variant: str | None = None, **kwargs: Any
    ) -> Any:
        """Detect ventilator breaths from the volume channel.

        See `detect_eit_breaths` for what `variant` does.

        This promotes what used to be a side effect of `postprocess_emg` into a
        method of its own, so ventilator breaths can be detected without
        running EMG postprocessing first. `postprocess_emg` still populates
        `session.events["ventilator_breaths"]` as before.
        """

        if variant is not None:
            data = self.processed_variants[VENTILATOR].get(variant)
            if data is None:
                raise MissingModalityDataError(
                    f"No ventilator preprocessing variant {variant!r}; call "
                    f"preprocess_ventilator(variant={variant!r}, ...) first."
                )
            event_key = f"ventilator_breaths:{variant}"
        else:
            data = self.processed.get(VENTILATOR)
            if data is None:
                # Detection needs the split channels, which raw data doesn't
                # have - so preprocess on the fly rather than failing, matching
                # how `postprocess_emg` accepts a raw ventilator recording.
                data = self.ventilator_adapter.preprocess(self._require_raw(VENTILATOR))
            event_key = "ventilator_breaths"
        events = self.ventilator_adapter.detect_breaths(data, **kwargs)
        self.add_events(event_key, events)
        self._record("detect_ventilator_breaths", VENTILATOR, variant=variant, **kwargs)
        return self.events[event_key]

    def add_events(self, name: str, events: Any) -> list[Any]:
        """Store a named event list while keeping `session.events` as backing data."""

        self.events[name] = list(events)
        return self.events[name]

    def get_events(self, name: str, default: Any = None) -> Any:
        """Return a named event list from `session.events`."""

        return self.events.get(name, default)

    def postprocess_emg(self, **kwargs: Any) -> Any:
        """Run EMG postprocessing through the adapter."""

        data = self.processed.get("emg") or self._require_raw("emg").data
        events = self.events.get("emg_breaths")
        self.parameters["emg_postprocessing"] = self.emg_adapter.postprocess(
            data,
            events=events,
            **kwargs,
        )
        for parameter in self.emg_adapter.to_parameters(
            self.parameters["emg_postprocessing"]
        ):
            self.parameter_results.add(parameter)
        for flag in self.emg_adapter.to_quality_flags(
            self.parameters["emg_postprocessing"]
        ):
            self.quality.add(flag)
        ventilator_breaths = self._normalize_ventilator_breaths(
            self.parameters["emg_postprocessing"],
            ventilator=kwargs.get("ventilator"),
            ventilator_fs=kwargs.get("ventilator_fs"),
            ventilator_breath_width_seconds=kwargs.get(
                "ventilator_breath_width_seconds",
            ),
        )
        if ventilator_breaths:
            self.add_events("ventilator_breaths", ventilator_breaths)
        self._record("postprocess_emg", "emg", **kwargs)
        return self.parameters["emg_postprocessing"]

    def synchronize_multimodal_breaths(
        self,
        method: str = "manual_offset",
        offset_seconds: float | Mapping[str, float] = 0.0,
        reference_modality: str | None = None,
    ) -> dict[str, Any]:
        """Apply basic Stage 1 alignment to stored breath event lists.

        Named to mirror `synchronize_raw_modalities`: that one shifts raw
        signals in time before processing, this one shifts already-detected
        breath events (`ALIGNMENT_EVENT_LISTS`) in time after detection.
        Previously named `align_modalities`, kept below as an alias.

        Offsets are resolved relative to `reference_modality` (or the
        auto-detected one - see `_resolve_alignment_reference`) before being
        applied, so the reference modality's own events are shifted by zero and
        every other modality moves by its offset *relative to* the reference,
        matching `synchronize_raw_modalities`. `self.parameters["alignment"]`
        keeps both: `offset_seconds` (relative, what was actually applied) and
        `configured_offset_seconds` (the raw per-modality values passed in).

        The start times set by `synchronize_raw_modalities` are added on top,
        so the shifted breaths are on the shared clock. The two add up:
        `offset_seconds` is only for a further correction after detection.
        `start_time_shift_seconds` records what was added for each modality.
        """

        if method != "manual_offset":
            raise ValueError("Stage 1 supports only method='manual_offset'")

        configured_offsets = resolve_alignment_offsets(offset_seconds)
        requested_reference = reference_modality
        resolved_reference, fallback_reference = self._resolve_alignment_reference(
            reference_modality
        )
        # `configured_offsets` are each modality's raw, independently-configured
        # offset. Applying those directly (as this used to) shifts every
        # modality including the reference one by its own offset, so the
        # reference's events move too - the opposite of what naming a reference
        # modality means. Relativizing first, as `synchronize_raw_modalities`
        # already does, is what makes the reference modality's own events stay
        # put (offset 0) and every other modality move by its offset *relative
        # to* the reference.
        offsets = offsets_relative_to_reference(configured_offsets, resolved_reference)
        start_time_shifts = shared_clock_shifts(self)
        total_offsets = {
            modality: offset + start_time_shifts.get(modality, 0.0)
            for modality, offset in offsets.items()
        }
        synchronized: dict[str, Any] = {}
        aligned_event_lists: list[str] = []
        missing_event_lists: list[str] = []
        for name in ALIGNMENT_EVENT_LISTS.values():
            events = self.events.get(name)
            if events is None:
                missing_event_lists.append(name)
                continue
            if not isinstance(events, list):
                continue
            synchronized[name] = align_events_by_modality_offset(events, total_offsets)
            aligned_event_lists.append(name)

        self.processed["synchronized"] = synchronized
        self.parameters["alignment"] = {
            "method": method,
            "reference_modality": resolved_reference,
            "requested_reference_modality": requested_reference,
            "fallback_reference_modality": fallback_reference,
            "offset_seconds": offsets,
            "configured_offset_seconds": configured_offsets,
            "start_time_shift_seconds": start_time_shifts,
            "aligned_event_lists": aligned_event_lists,
            "missing_event_lists": missing_event_lists,
        }
        self._record(
            "synchronize_multimodal_breaths",
            parameters={
                "method": method,
                "reference_modality": resolved_reference,
                "offset_seconds": offsets,
            },
        )
        return synchronized

    def align_modalities(
        self,
        method: str = "manual_offset",
        offset_seconds: float | Mapping[str, float] = 0.0,
        reference_modality: str | None = None,
    ) -> dict[str, Any]:
        """Deprecated alias for `synchronize_multimodal_breaths` - kept for
        backward compatibility with existing calling code."""

        return self.synchronize_multimodal_breaths(
            method, offset_seconds, reference_modality=reference_modality
        )

    def link_breaths(self, *, time_tolerance: float = 0.5) -> list[LinkedBreath]:
        """Link breaths across modalities into `LinkedBreath` objects (Milestone 2.5).

        Prefers the aligned breath lists produced by
        `synchronize_multimodal_breaths` (``self.processed["synchronized"]``)
        over the per-modality event lists in ``self.events``. A list that was
        not aligned there is moved onto the shared clock here, using the
        start times from `synchronize_raw_modalities`, so breaths are always
        matched on one time axis.
        """

        synchronized = self.processed.get("synchronized")
        if not isinstance(synchronized, dict):
            synchronized = {}
        start_time_shifts = shared_clock_shifts(self)

        def _breaths(name: str) -> list[BreathEvent] | None:
            events = synchronized.get(name)
            if events is None:
                events = self.events.get(name)
                if isinstance(events, list):
                    events = align_events_by_modality_offset(events, start_time_shifts)
            return events

        self.linked_breaths = link_breaths_by_time(
            {
                "eit": _breaths("eit_breaths"),
                "emg": _breaths("emg_breaths"),
                VENTILATOR: _breaths("ventilator_breaths"),
            },
            time_tolerance=time_tolerance,
        )
        self._record("link_breaths", parameters={"time_tolerance": time_tolerance})
        return self.linked_breaths

    def compute_multimodal_parameters(
        self,
        *,
        delay_pairs: Sequence[tuple[str, str]] | None = None,
        duration_pairs: Sequence[tuple[str, str]] | None = None,
        anchor: str = "start",
    ) -> list[ParameterResult]:
        """Compute timing-delay/duration-difference/event-agreement
        `ParameterResult`s from `self.linked_breaths` (plan_stage2.md Sec 21).

        Call `link_breaths` first; an empty `self.linked_breaths` yields an
        empty result rather than raising. Results are added to
        `self.parameter_results` and also returned.
        """

        results = compute_multimodal_parameters(
            self.linked_breaths,
            delay_pairs=delay_pairs,
            duration_pairs=duration_pairs,
            anchor=anchor,
        )
        for parameter in results:
            self.parameter_results.add(parameter)
        self._record(
            "compute_multimodal_parameters",
            parameters={"anchor": anchor, "n_linked_breaths": len(self.linked_breaths)},
        )
        return results

    def run_pipeline(
        self, name: str, *, config: Mapping[str, Mapping[str, Any]] | None = None
    ) -> M3Session:
        """Run a named, built-in `Pipeline` preset against this session.

        This is a different mechanism from the module-level
        ``m3resp.run_pipeline(spec, session=...)``, which executes a fully
        custom declarative step-list spec (the Stage 1 pipeline engine in
        ``m3resp.workflows``). ``session.run_pipeline(name)`` instead runs one
        of the small, built-in presets registered in ``m3resp.presets``
        (``"eit"``, ``"emg"``, ``"multimodal"``), which simply call this
        session's own already-instrumented methods in sequence - see
        ``m3resp.presets.base`` for the rationale.
        """

        from m3resp.presets import get_pipeline

        pipeline_cls = get_pipeline(name)
        return pipeline_cls().run(self, config=config)

    def export_summary(
        self, output_dir: str | Path, *, processing_run_id: str | None = None
    ) -> Path:
        """Export the session summary to disk.

        ``processing_run_id`` (typically `PipelineResult.processing_run_id`)
        links a written parameter-array archive to the `ProcessingRun` that
        produced it when a `DataModelRecorder` is attached; omit it for a
        manual export with no associated pipeline run.
        """

        output_path = export_session_summary(
            self, output_dir, processing_run_id=processing_run_id
        )
        self._record("export_summary", parameters={"output_dir": str(output_path)})
        return output_path

    def _require_raw(self, modality: str) -> Any:
        if modality not in self.raw:
            raise MissingModalityDataError(
                f"No raw {modality.upper()} data loaded. Call load_{modality} first."
            )
        return self.raw[modality]

    def _extend_typed_collections_from_eit(self, preprocessed: dict[str, Any]) -> None:
        for signal in self.eit_adapter.to_signals(preprocessed):
            self.signals.add(signal)
        for parameter in self.eit_adapter.to_parameters(preprocessed):
            self.parameter_results.add(parameter)
        for flag in self.eit_adapter.to_quality_flags(preprocessed):
            self.quality.add(flag)

    def _record(
        self,
        action: str,
        modality: str | None = None,
        parameters: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        record_parameters = parameters or kwargs
        provenance_record = record(action, modality, **record_parameters)
        self.provenance.append(provenance_record)
        if self.datamodel is not None:
            self.datamodel.record_provenance(provenance_record)

    def _resolve_alignment_reference(
        self,
        reference_modality: str | None,
    ) -> tuple[str, str | None]:
        if reference_modality is not None:
            return normalize_modality(reference_modality), None
        if self.events.get("ventilator_breaths"):
            return VENTILATOR, None
        return "eit", "eit"

    def _check_ventilator_start_keys(self, offsets: Mapping[str, float]) -> None:
        """Refuse a ``"ventilator:<name>"`` start time that would be ignored:
        one for a recording that is not loaded, or for ventilator data that
        came inside the EIT or EMG file (which uses that file's start time)."""

        prefix = ventilator_start_key("")
        for key in offsets:
            if not key.startswith(prefix):
                continue
            name = key[len(prefix) :]
            recording = self.ventilators.get(name)
            if recording is None:
                raise ValueError(
                    f"Start time {key!r}: no ventilator recording named "
                    f"{name!r}. Loaded: {sorted(self.ventilators) or 'none'}."
                )
            clock = ventilator_clock(recording)
            if clock != VENTILATOR:
                raise ValueError(
                    f"Start time {key!r}: ventilator recording {name!r} came "
                    f"inside the {clock.upper()} file, so it always uses the "
                    f"{clock!r} start time. Give that one instead."
                )

    def _resolve_raw_alignment_reference(self, reference_modality: str | None) -> str:
        if reference_modality is not None:
            return normalize_offset_key(reference_modality)
        if VENTILATOR in self.raw or "vent" in self.raw:
            return VENTILATOR
        if "eit" in self.raw:
            return "eit"
        if "emg" in self.raw:
            return "emg"
        return "eit"

    def _normalize_ventilator_breaths(
        self,
        postprocessing: Any,
        *,
        ventilator: Any | None,
        ventilator_fs: float | None,
        ventilator_breath_width_seconds: float | None,
    ) -> list[BreathEvent]:
        detections = (
            postprocessing.get("computed", {})
            .get("event_detection", {})
            .get("detect_ventilator_breath", [])
            if isinstance(postprocessing, dict)
            else []
        )
        if detections is None:
            return []

        fs = infer_ventilator_fs(ventilator, ventilator_fs)
        width_seconds = (
            0.0
            if ventilator_breath_width_seconds is None
            else float(ventilator_breath_width_seconds)
        )
        duration_seconds = infer_ventilator_duration(ventilator, fs)
        return [
            normalize_ventilator_breath(
                detection,
                fs=fs,
                width_seconds=width_seconds,
                duration_seconds=duration_seconds,
            )
            for detection in iter_ventilator_detections(detections)
        ]


def _coerce_metadata(
    metadata: SessionMetadata | dict[str, Any] | None,
) -> SessionMetadata:
    if metadata is None:
        return SessionMetadata()
    if isinstance(metadata, SessionMetadata):
        return metadata
    return SessionMetadata(attributes=dict(metadata))
