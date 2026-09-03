# Changelog

## Unreleased

### The standard EMG pipeline now removes ECG, band-passes 20-500 Hz, and computes an RMS envelope

The `"emg"` preset was `preprocess_emg` -> `detect_emg_breaths` ->
`postprocess_emg`, with no ECG removal anywhere, so the pipeline advertised as
the standard EMG chain produced an ECG-contaminated envelope - and every breath
detection and amplitude-derived parameter computed from it. The standard chain
is band-pass -> ECG peak detection -> gating -> envelope -> baseline, and the
preset now runs it.

**These are behavior changes: EMG amplitudes, envelopes, and breath detections
will differ from previous runs.**

- `EMGPipeline` (`session.run_pipeline("emg")`) runs `emg.ecg_detect_peaks` +
  `emg.ecg_gating` between preprocessing and breath detection. Gating
  recomputes the envelope from the gated signal, so nothing downstream sees the
  pre-gating envelope.
  - `config={"ecg_detect_peaks": {...}}` / `config={"ecg_gating": {...}}` pass
    keyword arguments to either step; `{"ecg_detect_peaks": {"ecg_channel": n}}`
    points detection at a dedicated reference ECG channel.
  - `config={"ecg_removal": {"ecg_peak_indices": [...]}}` gates already-known
    peaks and skips detection.
  - `config={"ecg_removal": {"enabled": False}}` restores the old behavior. This
    is a data-check/exploratory path only.
- The default EMG band-pass is **20-500 Hz** (`high_pass_hz` was `10.0`).
  `low_pass_hz` still caps at 0.95 x Nyquist below 1053 Hz sampling. The
  high-pass no longer doubles as weak ECG suppression, since gating now owns
  that.
- The default EMG envelope is **RMS**, not ARV. `preprocess_emg` and
  `emg.preprocess` gained `envelope_method` (`"rms"` default, `"arv"`);
  `emg.ecg_gating` gained the same parameter, defaulting to whatever
  preprocessing used so the recomputed envelope cannot silently switch method.
  Pass `envelope_method="arv"` to reproduce `resurfemg`'s `full_rolling_arv`
  exactly.
- New shared primitive `m3resp.processing.rolling_envelope(values,
  window_length=..., method=...)` over the existing `rolling_rms`/`rolling_arv`,
  plus `ENVELOPE_METHODS`.
- `_preprocess_default` itself is unchanged in scope: it still only filters and
  envelopes. ECG removal lives in the preset, not in the adapter primitive, so
  composing the blocks by hand stays possible.

### Signals carry a data *category* alongside their modality

`Signal`, `ParameterResult`, and `QualityFlag` gained a `category` field.
`modality` now means only "which device/technique produced this" (`"eit"`,
`"emg"`, `"ventilator"`); `category` means "what physical quantity is this"
(`"impedance"`, `"airway_pressure"`, `"airflow"`, `"volume"`). They are
independent axes: one device emits several quantities, and one quantity can
come from several devices.

- Physical quantities were removed from the `modality` vocabulary. Pocc results
  that were tagged `modality="pressure"` are now
  `modality="ventilator", category="airway_pressure"`. This value was never
  released on `main`, so no published API changes.
- Query either axis: `for_category(...)` joins `for_modality(...)` on
  `session.signals`, `session.parameter_results`, and `session.quality`.
  `for_modality` is unchanged.
- **Fixes** persisted signal types being recorded incorrectly. Resolving a
  Layer 2 `SignalStream.signal_type` needs both axes, so with only `modality`
  every ventilator signal was stored as `ventilator_pressure` regardless of
  what it measured, and `ventilator_volume` was unreachable. An uncategorized
  ventilator signal is now skipped with a warning instead of guessed.
- **Fixes** derived features being attributed to the wrong signal stream: the
  recorder cached streams by modality alone, so a device's channels overwrote
  one another.
- `parameter_results.csv` gains a `category` column.
- The category vocabulary is open and extensible at runtime via
  `register_category_alias` / `load_category_aliases`, so an externally
  maintained taxonomy can be adopted without vendoring it.

### Ventilator becomes a first-class modality

- Ventilator pipeline steps moved out of the `emg.*` namespace into
  `ventilator.*`, and now declare `modality="ventilator"`, so they are
  discoverable under their own prefix instead of buried among the EMG steps:

  | Was | Now |
  |---|---|
  | `emg.load_ventilator` | `ventilator.load` |
  | `emg.ventilator_channels` | `ventilator.channels` |
  | `emg.detect_ventilator_breath` | `ventilator.detect_breaths` |
  | `emg.normalize_ventilator_breaths` | `ventilator.normalize_breaths` |
  | `emg.ventilator_respiratory_rate` | `ventilator.respiratory_rate` |
  | `emg.find_occluded_breaths` | `ventilator.find_occluded_breaths` |
  | `emg.pocc_intervals` | `ventilator.pocc_intervals` |
  | `emg.pocc_time_product` | `ventilator.pocc_time_product` |
  | `emg.pocc_quality` | `ventilator.pocc_quality` |
  | `emg.detect_non_consecutive_manoeuvres` | `ventilator.detect_non_consecutive_manoeuvres` |

  Every former id keeps working as a **silent alias**, so existing pipeline
  specs compile and run unchanged with no warning. Aliases resolve to the
  current step, so a spec written against an old id still records the new
  `operation_id` in provenance. They are deliberately hidden from
  `available_steps()`/`describe_steps()` and from the "available steps" list in
  `UnknownStepError`, so discovery and any GUI built on it only ever offer
  current names. `register_step` gained an `aliases=` argument and the registry
  exports `STEP_ALIASES`.

  The step *functions* moved too, into a new `m3resp.workflows.steps.ventilator`
  package (`loading.py`/`detection.py`/`normalization.py`/`features.py`/
  `quality.py`, plus its own `_shared.py` mirroring the EIT/EMG packages'
  pattern of a per-modality provenance helper rather than a cross-package
  import). `from m3resp.workflows.steps.emg import pocc_quality` etc. must
  become `from m3resp.workflows.steps.ventilator import pocc_quality`.

  Moving these off the EMG package's shared `_record_step` also **fixes** a
  latent bug: `ventilator.pocc_intervals`/`.pocc_time_product`/`.pocc_quality`/
  `.detect_non_consecutive_manoeuvres` were recording their step-level
  provenance under `modality="emg"` (the EMG helper's hardcoded value) even
  though nothing about them is EMG-specific. They now correctly record
  `modality="ventilator"`.
- New `VentilatorAdapter` (`m3resp.adapters.ventilator_adapter`), completing the
  set alongside `EITProcessingAdapter` and `ReSurfEMGAdapter`. It is the first
  adapter that wraps **no upstream library**: neither `eitprocessing` nor
  `resurfemg` implements ventilator preprocessing, which is why ventilator
  channels were previously used unfiltered. Its defaults are native, built on
  `m3resp.processing.filters` and `m3resp.processing.peaks`.
  - `preprocess()` splits a recording into pressure/flow/volume and returns the
    channels as the ventilator recorded them. Low-passing ventilator waveforms
    is not standard practice, so no filter is applied unless `lowpass_hz` is
    given (clamped below Nyquist); `SUGGESTED_LOWPASS_HZ` (20 Hz) is offered as
    a starting point. When a cutoff is used the unfiltered arrays stay
    available under `"raw"`.
  - `to_signals()` emits `modality="ventilator"` with a per-channel `category`
    (`airway_pressure`/`airflow`/`volume`), so a ventilator's three quantities
    are finally distinguishable. Ventilator data has never reached
    `session.signals` before.
  - `detect_breaths()` returns `BreathEvent`s from the volume channel.
  - Loading delegates to `ReSurfEMGAdapter` unless a loader is injected, since
    ventilator channels usually share the sEMG's file.
- New `M3Session.preprocess_ventilator(variant=..., overwrite=..., **kwargs)`
  and `M3Session.detect_ventilator_breaths(variant=...)`, with the same
  `variant`/`overwrite`/`allow_overwrite` semantics as their EIT/EMG
  counterparts and a `processed_variants["ventilator"]` slot to match.
  - Ventilator signals now reach `session.signals` for the first time.
  - Ventilator breath detection used to be reachable only as a side effect of
    `postprocess_emg`; it is now a method of its own. `postprocess_emg` still
    populates `session.events["ventilator_breaths"]` unchanged.
- New `M3Session.load_ventilator(path, **kwargs)`, mirroring
  `load_eit`/`load_emg`. It stores a new `VentilatorRecording`
  (`m3resp.modalities.ventilator`) on `session.ventilator` and under
  `session.raw["ventilator"]`, and records `load_ventilator` provenance.
- `M3Session(ventilator_adapter=...)` allows a dedicated loader. It defaults to
  the EMG adapter, since ventilator channels usually share the sEMG's file, so
  an injected EMG loader keeps covering both.
- **Breaking:** `session.raw["ventilator"]`/`["vent"]` now hold a
  `VentilatorRecording` rather than the bare `{"array", "metadata"}` payload,
  matching what `raw["eit"]`/`raw["emg"]` have always held. Code reading
  `session.raw["vent"]["array"]` should use `session.ventilator.data["array"]`,
  or `m3resp.synchronization.cropping.ventilator_payload(...)`, which unwraps
  either shape. Assigning a bare dict to `raw["vent"]` still works.
- The `emg.load_ventilator` pipeline step now delegates to the session method,
  so `session.raw` bookkeeping and provenance happen in one place. Its emitted
  `ventilator_raw` artifact is unchanged.

### The ventilator modality canonicalizes to `"ventilator"`

Stage 1 used `"vent"` internally while the docs, `M3Session.link_breaths`, and
`Signal.modality` used `"ventilator"` - so a breath could be tagged
`modality="vent"` while the `LinkedBreath` holding it was keyed `"ventilator"`.
Everything now normalizes to `"ventilator"`.

`"vent"` keeps working everywhere it was previously accepted:

- `session.raw` stores the ventilator recording under **both** keys, pointing
  at the same object, so `session.raw["vent"]` still resolves. Cropping mutates
  that object in place, so the two views cannot drift apart.
- Alignment canonicalizes both the event's modality and the offset keys before
  matching, so events tagged `"vent"` still shift under a `"ventilator"` offset
  and vice versa.
- `sync.estimate_offset` accepts `vent` and `ventilator` as source values, so
  existing pipeline specs keep validating.

Detected ventilator breaths now carry `modality="ventilator"`, and
`session.parameters["alignment"]["offset_seconds"]` is keyed `"ventilator"`.
Code comparing those values against the literal `"vent"` needs updating.

## 0.1.0

- Create the initial Stage 1 M3Resp package skeleton.
- Add `M3Session`, event models, modality adapters, manual synchronization, and
  export helpers.
