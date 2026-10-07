"""The central Stage 1 M3Resp session object."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from m3resp.adapters.eitprocessing_adapter import EITProcessingAdapter
from m3resp.adapters.resurfemg_adapter import ReSurfEMGAdapter
from m3resp.adapters.ventilator_adapter import (
    VentilatorAdapter,
    infer_ventilator_duration,
    infer_ventilator_fs,
    iter_ventilator_detections,
    normalize_ventilator_breath,
    primary_channel,
)
from m3resp.core.exceptions import MissingModalityDataError, VariantAlreadyExistsError
from m3resp.core.metadata import SessionMetadata
from m3resp.core.provenance import ProvenanceRecord, record
from m3resp.data.collections import (
    IntervalDataCollection,
    ParameterResultCollection,
    PixelMaskCollection,
    QualityReport,
    SignalCollection,
)
from m3resp.data.events import BreathEvent, reuse_matching_breaths
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
from m3resp.modalities.names import VENTILATOR
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
    breath_reference_modality,
    offsets_relative_to_reference,
    raw_reference_modality,
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
from m3resp.synchronization.sync_methods import (
    MANUAL,
    NONE,
    clock_key,
    recording_keys,
    warn_if_not_synchronized,
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


def _eit_event_key(variant: str | None) -> str:
    """Return ``eit_breaths`` or ``eit_breaths:<variant>`` for a named variant."""

    return "eit_breaths" if variant is None else f"eit_breaths:{variant}"


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
        """Create a session for recordings, processing results and their history.

        Args:
            eit_adapter: EIT loading and processing methods. ``None`` uses
                ``EITProcessingAdapter``.
            emg_adapter: EMG loading and processing methods. ``None`` uses
                ``ReSurfEMGAdapter``.
            metadata: Recording description as ``SessionMetadata`` or a
                dictionary. ``None`` creates empty metadata.
            allow_overwrite: Allow preprocessing calls to replace a result
                stored under the same variant name.
            ventilator_adapter: Ventilator loading and processing methods.
                ``None`` creates an adapter using this session's EIT and EMG
                loaders for files that contain ventilator channels.
        """

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
        # How each recording was placed on the shared clock ("manual",
        # "none", ...), keyed like `start_times`. A recording missing here was
        # never synchronized, and steps that compare it with a recording on
        # another clock warn. See `m3resp.synchronization.sync_methods`.
        self.sync_methods: dict[str, str] = {}
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
        # Signals and results shared by EIT, EMG and ventilator processing.
        self.signals = SignalCollection()
        self.parameter_results = ParameterResultCollection()
        # Results with one value per breath (or other interval), e.g. EIT
        # TIV and EELI, and the masks that select a region of EIT pixels.
        # Their times are on each recording's own clock, like `events`.
        self.interval_data = IntervalDataCollection()
        self.pixel_masks = PixelMaskCollection()
        self.quality = QualityReport()
        # Breaths matched across modalities by `link_breaths`.
        self.linked_breaths: list[LinkedBreath] = []
        self.metadata = _coerce_metadata(metadata)
        self.provenance: list[ProvenanceRecord] = []
        # The workflow engine records each step's settings and timing here.
        self.processing_history = ProcessingHistory()
        # Attach a recorder to save processing history and results in the data model.
        self.datamodel: DataModelRecorder | None = None

    def load_eit(
        self, path: str | Path, vendor: str | None = None, **kwargs: Any
    ) -> Any:
        """Load EIT data and store it under `raw["eit"]`.

        A newly loaded recording has not been synchronized yet: any EIT start
        time and synchronization record from before are cleared.

        Args:
            path (str | Path): The EIT file to read.
            vendor (str | None): The EIT device maker, passed to
                `eitprocessing` (for example ``"draeger"``).
            **kwargs (Any): Passed on to the EIT file reader.

        Returns:
            Any: The loaded `eitprocessing` Sequence (`session.eit.data`).
        """

        recording = load_eit_recording(
            path,
            vendor=vendor,
            adapter=self.eit_adapter,
            **kwargs,
        )
        self.eit = recording
        self.raw["eit"] = recording
        self._forget_synchronization("eit")
        self._record("load_eit", "eit", path=str(path), vendor=vendor)
        return recording.data

    def load_emg(self, path: str | Path, **kwargs: Any) -> Any:
        """Load EMG data and store it under `raw["emg"]`.

        A newly loaded recording has not been synchronized yet: any EMG start
        time and synchronization record from before are cleared.

        Args:
            path (str | Path): The EMG file to read.
            **kwargs (Any): Passed on to the EMG file reader.

        Returns:
            Any: The loaded EMG data (`session.emg.data`), a dictionary with
                the samples under ``"array"`` and the file details under
                ``"metadata"``.
        """

        recording = load_emg_recording(path, adapter=self.emg_adapter, **kwargs)
        self.emg = recording
        self.raw["emg"] = recording
        self._forget_synchronization("emg")
        self._record("load_emg", "emg", path=str(path))
        return recording.data

    def load_ventilator(
        self, path: str | Path, *, name: str | None = None, **kwargs: Any
    ) -> Any:
        """Load ventilator data and store it under `raw["ventilator"]`.

        Mirrors `load_eit`/`load_emg`. The recording is additionally stored
        under the legacy `raw["vent"]` key, pointing at the same object.

        A newly loaded standalone recording has not been synchronized yet: its
        own start time and synchronization record from before are cleared.
        Ventilator data from the EIT or EMG file follows that file's clock.

        Args:
            path (str | Path): The file the ventilator data arrives in: the
                multi-channel export shared with the sEMG, or an EIT ``*.bin``
                carrying ventilator waveforms beside its impedance frames.
                `VentilatorAdapter` picks by file suffix.
            name (str | None): Files this recording alongside any already
                loaded, for a study where more than one instrument recorded
                ventilator data - a ventilator export and the EIT file's own
                Medibus channels, say, each with its own airway pressure.
                Without a name the recording is the primary one, which is what
                `session.ventilator` and `raw["ventilator"]` point at.
                Preprocess a named recording with
                `preprocess_ventilator(name=...)`, which qualifies its channel
                keys so the two airway pressures stay distinct in
                `session.signals`.
            **kwargs (Any): Passed on to the ventilator file reader. Use
                ``source="eit"``/``"emg"``/``"ventilator"`` to choose the file
                type, and ``ventilator_channels=`` to select which channels to
                read from a ``*.bin`` (see
                `m3resp.adapters.ventilator_adapter`).

        Returns:
            Any: The loaded ventilator data (`VentilatorRecording.data`), a
                dictionary with the samples under ``"array"`` and the file
                details under ``"metadata"``.
        """

        recording = load_ventilator_recording(
            path, adapter=self.ventilator_adapter, **kwargs
        )
        key = name or DEFAULT_VENTILATOR_NAME
        self.ventilators[key] = recording
        if key == DEFAULT_VENTILATOR_NAME or self.ventilator is None:
            self.ventilator = recording
            set_ventilator_raw(self.raw, recording)
        if ventilator_clock(recording) == VENTILATOR:
            self._forget_synchronization(ventilator_start_key(key))
        self._record("load_ventilator", VENTILATOR, path=str(path), name=key)
        return recording.data

    def primary_ventilator_name(self) -> str | None:
        """The name of the recording `session.ventilator` points at.

        Returns:
            str | None: The name, or None when no ventilator recording is
                loaded.
        """

        if DEFAULT_VENTILATOR_NAME in self.ventilators:
            return DEFAULT_VENTILATOR_NAME
        return next(iter(self.ventilators), None)

    def get_ventilator(self, name: str | None = None) -> VentilatorRecording:
        """A loaded ventilator recording by name, or the primary one.

        Args:
            name (str | None): The name the recording was loaded under (see
                `load_ventilator`). None gives the primary recording.

        Returns:
            VentilatorRecording: The loaded recording.
        """

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
        """Preprocess the loaded EIT recording and store the results.

        Results are stored in ``session.processed_variants['eit']`` under the
        variant name, or ``'default'`` when omitted. The default result is
        also stored in ``session.processed['eit']``. Upstream preprocessing
        adds signals, rate parameters, values per breath and quality flags
        to the session's collections. Per-breath results reuse breaths
        stored under the matching variant's event key.

        Args:
            variant (str | None): Name to store this result under. None
                stores it as ``"default"``.
            overwrite (bool): Replace a result already stored under the same
                name. ``session.allow_overwrite`` also permits replacement.
            **kwargs (Any): Passed on to `EITProcessingAdapter.preprocess`
                (for example ``filter_mode``). ``preprocess=`` replaces the
                whole step with a function of your own; its output is stored
                directly in the processed-result dictionaries.

        Returns:
            Any: The preprocessing result, also stored in
                `session.processed_variants["eit"]`.

        Raises:
            MissingModalityDataError: If an EIT recording has yet to be loaded.
            VariantAlreadyExistsError: If the variant already exists and
                replacement is disabled.
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
            self._extend_typed_collections_from_eit(
                result, event_key=_eit_event_key(variant)
            )
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

        Args:
            variant (str | None): Name to store this result under. None
                stores it as ``"default"``.
            overwrite (bool): Replace a result already stored under the same
                name.
            **kwargs (Any): Passed on to `ReSurfEMGAdapter.preprocess`.
                ``preprocess=`` replaces the whole step with a function of
                your own.

        Returns:
            Any: The preprocessing result. The default step gives a
                dictionary that includes the ``"filtered"`` and
                ``"envelope"`` signals, the ``"channel"`` used and the
                sampling rate ``"fs"`` in Hz.
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

        Unlike its EIT/EMG siblings this runs native code rather than an
        upstream library: nothing in `eitprocessing`/`resurfemg` preprocesses
        ventilator data, which is why these channels used to be consumed
        unfiltered.

        Args:
            name (str | None): Which loaded recording to preprocess when a
                study recorded ventilator data on more than one instrument
                (see `load_ventilator`). A non-primary recording's channel
                keys are qualified with its name - ``pressure__pod`` rather
                than ``pressure`` - so its airway pressure does not collide
                with the primary recording's in `session.signals`. None
                preprocesses the primary recording.
            variant (str | None): Name to store this result under. Defaults
                to `name`, so each recording lands in its own slot rather
                than overwriting, and to ``"default"`` when `name` is None.
            overwrite (bool): Replace a result already stored under the same
                name.
            **kwargs (Any): Passed on to `VentilatorAdapter.preprocess`:
                ``lowpass_hz`` sets the cut-off in Hz, with
                ``lowpass_hz=None`` skipping the filter, and ``preprocess=``
                replaces the whole step with a function of your own. See
                `m3resp.adapters.ventilator_adapter` for the defaults.

        Returns:
            Any: The preprocessing result, a dictionary with one filtered
                signal per channel (pressure, flow, volume) and the sampling
                rate ``"fs"`` in Hz.
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
        recording_name = name or primary
        for signal in self.ventilator_adapter.to_signals(result):
            # Which loaded recording this came from, so its synchronization
            # can be looked up later (e.g. by the data model recorder). The
            # channel name alone cannot tell: a `__pod` suffix may mean a second
            # recording or a pod channel inside the first one.
            if recording_name is not None:
                signal.metadata.setdefault("recording", recording_name)
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

        No samples are removed: every recording keeps its full length. The
        start times are stored in `session.start_times` and are added to
        breath times only when modalities are compared
        (`synchronize_multimodal_breaths`, `link_breaths`), so each
        modality's own results stay on its own clock. Calling this again
        replaces the start times rather than adding to them. See
        `m3resp.synchronization.start_times`.

        Args:
            method (str): How the start times are found. Only
                ``"manual_offset"`` is supported.
            offset_seconds (float | Mapping[str, float]): Start time of each
                modality in seconds, for example ``{"emg": 5.0}`` for "EMG
                started 5 s after EIT", or ``{"emg": -5.0}`` for "EMG started
                5 s before EIT". A single number is the EMG start time.
                ``"ventilator"`` is the start time of the standalone ventilator
                recordings (those loaded with ``source="ventilator"``). When
                two of them started at different moments, give one its own
                start time with ``"ventilator:<name>"``, the name it was loaded
                under, e.g. ``{"ventilator": 0.0, "ventilator:monitor": 12.5}``.
                Ventilator data that came inside the EIT or EMG file always
                uses that file's start time.
            reference_modality (str | None): The recording that start times
                are counted from, so its own start time is always 0. A
                modality name or ``"ventilator:<name>"``. When None: the
                ventilator if one is loaded, otherwise EIT, otherwise EMG.

        Returns:
            dict[str, Any]: ``{modality: {"start_time_seconds": ...}}`` for
                every loaded modality, plus a ``"ventilator:<name>"`` entry for
                each standalone ventilator recording given its own start time.
        """

        if method != "manual_offset":
            raise ValueError("Stage 1 supports only method='manual_offset'")

        configured_offsets = resolve_alignment_offsets(offset_seconds)
        self._check_ventilator_start_keys(configured_offsets)
        resolved_reference = raw_reference_modality(self, reference_modality)
        offsets = offsets_relative_to_reference(configured_offsets, resolved_reference)
        self.start_times = {
            modality: float(offset) for modality, offset in offsets.items()
        }
        for key in self._synchronizable_keys():
            self.sync_methods[key] = MANUAL

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

        Args:
            start_seconds (float): Start of the part to keep, in seconds.
            end_seconds (float | None): End of the part to keep, in seconds.
                None keeps everything up to the end.

        Returns:
            dict[str, Any]: The times used and the number of samples removed
                from the start and the end and kept. Also stored in
                `session.parameters["emg_slice"]`.
        """

        recording = self.emg
        if (
            recording is None
            or not isinstance(recording.data, dict)
            or "array" not in recording.data
        ):
            raise MissingModalityDataError("slice_emg needs a loaded EMG recording.")
        window = emg_sample_window(recording, start_seconds, end_seconds)
        if window.sample_frequency is None:
            raise ValueError("slice_emg: the EMG recording has no sampling rate.")
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
        the loaded recording; reload the file to get them back.

        Args:
            start_seconds (float): Start of the part to keep, in seconds.
            end_seconds (float | None): End of the part to keep, in seconds.
                None keeps everything up to the end.

        Returns:
            dict[str, Any]: The times used, the number of frames removed from
                the start and the end and kept, and how many ventilator
                recordings were cut with the EIT. Also stored in
                `session.parameters["eit_slice"]`.
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

        Times are in seconds from the start of the recording as it is now
        (after any earlier slicing). ``end_seconds=None`` keeps everything up
        to the end. After slicing, the recording's times count from the new
        start, and its start time moves later by `start_seconds`: it is
        stored under its own ``"ventilator:<name>"`` entry in
        `session.start_times`, so the recording stays lined up with the
        other modalities and no other recording moves.

        Run this before `preprocess_ventilator`. The removed samples are
        gone from the loaded recording; reload the file to get them back.

        Args:
            start_seconds (float): Start of the part to keep, in seconds.
            end_seconds (float | None): End of the part to keep, in seconds.
                None keeps everything up to the end.
            name (str | None): Cut only the recording loaded under this name.
                None cuts every standalone ventilator recording.

        Returns:
            dict[str, Any]: The times used and, under ``"recordings"``, the
                number of samples removed from the start and the end and kept
                for each recording that was cut. Also stored in
                `session.parameters["ventilator_slice"]`.
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
        """Detect EIT breaths and store them as ``BreathEvent`` objects.

        Breaths matching an existing per-breath EIT result by modality and
        exact start and end time reuse that result's breath object, including
        its identifier. Detection is recorded in the session's history.

        Args:
            variant (str | None): Detect breaths in the
                ``preprocess_eit(..., variant=<name>)`` result. Events are
                stored under ``session.events['eit_breaths:<name>']``.
                ``None`` uses the default processed result, falling back to
                the raw recording, and stores events under ``'eit_breaths'``.
            **kwargs (Any): Passed on to `EITProcessingAdapter.detect_breaths`.
                ``detector=`` replaces the detection with a function of your
                own.

        Returns:
            list[BreathEvent]: Detected breaths in detector order, also stored
                in ``session.events``. Times use the input recording's clock.

        Raises:
            MissingModalityDataError: If the named preprocessing variant is
                unavailable, or default processed EIT data and a raw recording
                are both unavailable.
        """

        if variant is not None:
            data = self.processed_variants["eit"].get(variant)
            if data is None:
                raise MissingModalityDataError(
                    f"No EIT preprocessing variant {variant!r}; call "
                    f"preprocess_eit(variant={variant!r}, ...) first."
                )
        else:
            data = self.processed.get("eit") or self._require_raw("eit").data
        event_key = _eit_event_key(variant)
        events = self.eit_adapter.detect_breaths(data, **kwargs)
        # The same breaths may already be stored with TIV/EELI values; store
        # those breath objects, so each value points to its stored breath.
        events = reuse_matching_breaths(
            events,
            (
                interval
                for result in self.interval_data.for_modality("eit")
                for interval in result.intervals
            ),
        )
        self.add_events(event_key, events)
        self._record("detect_eit_breaths", "eit", variant=variant, **kwargs)
        return self.events[event_key]

    def detect_emg_breaths(self, *, variant: str | None = None, **kwargs: Any) -> Any:
        """Detect EMG breaths and store normalized events.

        Args:
            variant (str | None): Detect breaths in the `preprocess_emg`
                result with this name, stored under
                `session.events["emg_breaths:<name>"]`. See
                `detect_eit_breaths`.
            **kwargs (Any): Passed on to `ReSurfEMGAdapter.detect_breaths`
                (for example ``baseline=``). ``detector=`` replaces the
                detection with a function of your own.

        Returns:
            Any: The detected breaths, a list of `BreathEvent`.
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

        This promotes what used to be a side effect of `postprocess_emg` into a
        method of its own, so ventilator breaths can be detected without
        running EMG postprocessing first. `postprocess_emg` still populates
        `session.events["ventilator_breaths"]` as before.

        Args:
            variant (str | None): Detect breaths in the
                `preprocess_ventilator` result with this name, stored under
                `session.events["ventilator_breaths:<name>"]`. See
                `detect_eit_breaths`. When None and the ventilator has not
                been preprocessed, it is preprocessed here with the default
                settings.
            **kwargs (Any): Passed on to `VentilatorAdapter.detect_breaths`
                (for example ``breath_width_seconds``). ``detector=``
                replaces the detection with a function of your own.

        Returns:
            Any: The detected breaths, a list of `BreathEvent`.
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
        """Store a named event list while keeping `session.events` as backing data.

        Args:
            name (str): The name to store the events under, for example
                ``"emg_breaths"``. An existing list with this name is
                replaced.
            events (Any): The events, for example a list of `BreathEvent`.

        Returns:
            list[Any]: The stored events.
        """

        self.events[name] = list(events)
        return self.events[name]

    def get_events(self, name: str, default: Any = None) -> Any:
        """Return a named event list from `session.events`.

        Args:
            name (str): The name the events were stored under, for example
                ``"emg_breaths"``.
            default (Any): What to return when no events have this name.

        Returns:
            Any: The stored events, or `default`.
        """

        return self.events.get(name, default)

    def postprocess_emg(self, **kwargs: Any) -> Any:
        """Run EMG postprocessing through the adapter.

        Uses the default EMG preprocessing result and the breaths in
        `session.events["emg_breaths"]`. Any ventilator breaths found during
        postprocessing are stored in `session.events["ventilator_breaths"]`.

        Args:
            **kwargs (Any): Passed on to `ReSurfEMGAdapter.postprocess`, for
                example ``ventilator``, ``ventilator_fs`` (Hz) and
                ``ventilator_breath_width_seconds``.

        Returns:
            Any: The postprocessing result, a dictionary with the values
                computed (``"computed"``), those skipped (``"skipped"``) and
                the settings used (``"settings"``). Also stored in
                `session.parameters["emg_postprocessing"]`.
        """

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
        auto-detected one - see
        `m3resp.synchronization.alignment.breath_reference_modality`) before
        being applied, so the reference modality's own events are shifted by zero and
        every other modality moves by its offset *relative to* the reference,
        matching `synchronize_raw_modalities`. `self.parameters["alignment"]`
        keeps both: `offset_seconds` (relative, what was actually applied) and
        `configured_offset_seconds` (the raw per-modality values passed in).

        The start times set by `synchronize_raw_modalities` are added on top,
        so the shifted breaths are on the shared clock. The two add up:
        `offset_seconds` is only for a further correction after detection.
        `start_time_shift_seconds` records what was added for each modality.

        Args:
            method (str): How the breaths are shifted. Only
                ``"manual_offset"`` is supported.
            offset_seconds (float | Mapping[str, float]): Extra shift of each
                modality's breaths in seconds, on top of the start times, for
                example ``{"emg": 0.2}``. A single number is the EMG shift.
            reference_modality (str | None): The modality whose breaths are
                not shifted by an extra offset. When None: the ventilator if
                there are ventilator breaths, otherwise EIT.

        Returns:
            dict[str, Any]: The shifted breath lists, keyed by list name (for
                example ``"emg_breaths"``). Also stored in
                `session.processed["synchronized"]`.
        """

        if method != "manual_offset":
            raise ValueError("Stage 1 supports only method='manual_offset'")

        configured_offsets = resolve_alignment_offsets(offset_seconds)
        requested_reference = reference_modality
        resolved_reference, fallback_reference = breath_reference_modality(
            self, reference_modality
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
        for modality, name in ALIGNMENT_EVENT_LISTS.items():
            if name in aligned_event_lists:
                self.sync_methods[clock_key(self, modality)] = MANUAL

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
        backward compatibility with existing calling code.

        Args:
            method (str): See `synchronize_multimodal_breaths`.
            offset_seconds (float | Mapping[str, float]): See
                `synchronize_multimodal_breaths`.
            reference_modality (str | None): See
                `synchronize_multimodal_breaths`.

        Returns:
            dict[str, Any]: The shifted breath lists, keyed by list name.
        """

        return self.synchronize_multimodal_breaths(
            method, offset_seconds, reference_modality=reference_modality
        )

    def skip_synchronization(self) -> list[str]:
        """Use the loaded recordings as they are, without synchronizing them.

        For recordings that really did start at the same moment - for
        example when one trigger started every device. Each loaded recording
        that has not been synchronized is recorded as ``"none"`` in
        `session.sync_methods`, so steps that compare recordings skip their
        missing-synchronization warning, while the choice stays visible in
        the provenance log and the exported summary. Recordings already synchronized keep their record.
        Breath lists added directly with `add_events`, without a loaded
        recording, are covered too. Recordings loaded later are not.

        Returns:
            list[str]: The recordings marked ``"none"``.
        """

        skipped = [
            key for key in self._synchronizable_keys() if key not in self.sync_methods
        ]
        for key in skipped:
            self.sync_methods[key] = NONE
        self._record("skip_synchronization", parameters={"recordings": skipped})
        return skipped

    def link_breaths(self, *, time_tolerance: float = 0.5) -> list[LinkedBreath]:
        """Link breaths across modalities into `LinkedBreath` objects (Milestone 2.5).

        Prefers the aligned breath lists produced by
        `synchronize_multimodal_breaths` (``self.processed["synchronized"]``)
        over the per-modality event lists in ``self.events``. A list that was
        not aligned there is moved onto the shared clock here, using the
        start times from `synchronize_raw_modalities`, so breaths are always
        matched on one time axis.

        Warns (`UnsynchronizedDataWarning`) when the breaths come from
        recordings on different clocks and one of them was never
        synchronized; see `skip_synchronization`.

        Args:
            time_tolerance (float): Largest time difference, in seconds, at
                which two breaths from different modalities are still linked.
                The closest match wins; a breath with no match gets a
                `LinkedBreath` of its own.

        Returns:
            list[LinkedBreath]: The linked breaths. Also stored in
                `session.linked_breaths`.
        """

        synchronized = self.processed.get("synchronized")
        if not isinstance(synchronized, dict):
            synchronized = {}
        start_time_shifts = shared_clock_shifts(self)
        warn_if_not_synchronized(
            self,
            [
                modality
                for modality, name in ALIGNMENT_EVENT_LISTS.items()
                if synchronized.get(name) or self.events.get(name)
            ],
            action="link_breaths",
        )

        def _breaths(name: str) -> list[BreathEvent] | None:
            events = synchronized.get(name)
            if events is not None:
                return events
            stored = self.events.get(name)
            if not isinstance(stored, list):
                return stored
            breaths: list[BreathEvent] = stored
            return align_events_by_modality_offset(breaths, start_time_shifts)

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

        Args:
            delay_pairs (Sequence[tuple[str, str]] | None): Modality pairs to
                compute the per-breath timing delay for, for example
                ``[("eit", "emg")]``. Positive means the second modality's
                breath comes later. None uses every pair of modalities found
                in the linked breaths.
            duration_pairs (Sequence[tuple[str, str]] | None): Modality pairs
                to compute the per-breath duration difference for. None uses
                every pair of modalities found in the linked breaths.
            anchor (str): Which point of each breath the delay is measured
                between: ``"start"``, ``"peak"`` or ``"end"``.

        Returns:
            list[ParameterResult]: The timing delays and duration differences
                per breath (in seconds), and one event-agreement result per
                delay pair.
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

        Args:
            name (str): The preset to run: ``"eit"``, ``"emg"`` or
                ``"multimodal"``.
            config (Mapping[str, Mapping[str, Any]] | None): Settings for each
                step, keyed by step name, for example
                ``{"preprocess": {"high_pass_hz": 20.0}}`` for the ``"emg"``
                preset. None uses the
                preset's defaults.

        Returns:
            M3Session: This session, with the results of every step stored.
        """

        from m3resp.presets import get_pipeline

        pipeline_cls = get_pipeline(name)
        return pipeline_cls().run(self, config=config)

    def export_summary(
        self, output_dir: str | Path, *, processing_run_id: str | None = None
    ) -> Path:
        """Export the session summary to disk.

        Args:
            output_dir (str | Path): The folder to write the files to. It is
                created if it does not exist.
            processing_run_id (str | None): Typically
                `PipelineResult.processing_run_id`. Links a written
                parameter-array archive to the `ProcessingRun` that produced
                it when a `DataModelRecorder` is attached; omit it for a
                manual export with no associated pipeline run.

        Returns:
            Path: The folder the files were written to.
        """

        output_path = export_session_summary(
            self, output_dir, processing_run_id=processing_run_id
        )
        self._record("export_summary", parameters={"output_dir": str(output_path)})
        return output_path

    def _synchronizable_keys(self) -> list[str]:
        """The `sync_methods` keys a synchronization call covers: every loaded
        recording with a clock of its own, plus the clock of any breath list
        added without a loaded recording (e.g. through `add_events`)."""

        keys = recording_keys(self)
        for modality, name in ALIGNMENT_EVENT_LISTS.items():
            if self.events.get(name):
                key = clock_key(self, modality)
                if key not in keys:
                    keys.append(key)
        return keys

    def _forget_synchronization(self, key: str) -> None:
        """Clear a recording's start time and synchronization record, for a
        recording that was just (re)loaded."""

        self.start_times.pop(key, None)
        self.sync_methods.pop(key, None)

    def _require_raw(self, modality: str) -> Any:
        if modality not in self.raw:
            raise MissingModalityDataError(
                f"No raw {modality.upper()} data loaded. Call load_{modality} first."
            )
        return self.raw[modality]

    def _extend_typed_collections_from_eit(
        self, preprocessed: dict[str, Any], *, event_key: str = "eit_breaths"
    ) -> None:
        """Add EIT signals, rates, values per breath and quality flags to the session.

        Per-breath values reuse matching breaths already stored under
        ``event_key``.
        """

        for signal in self.eit_adapter.to_signals(preprocessed):
            self.signals.add(signal)
        for parameter in self.eit_adapter.to_parameters(preprocessed):
            self.parameter_results.add(parameter)
        for interval_data in self.eit_adapter.to_interval_data(
            preprocessed, stored_breaths=self.get_events(event_key, None)
        ):
            self.interval_data.add(interval_data)
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
