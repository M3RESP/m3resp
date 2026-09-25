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

### The Annemijn example is replaced by the multidomain Recording1 example

`examples/annemijn_multimodal/annemijn.pipeline.yaml` is now
`examples/multidomain_recording1/recording1_a2.pipeline.yaml`. It runs the multidomain
results chain (`tools/visualization_tools/paper_results_v2.py`) on
Recording1 (`TestPS3.txt`), window A2 (140-200 s, EIT off), without the
adaptive harmonic stage, which A2 does not need. It gives the notebook's
numbers value for value: 77 ECG peaks, 9 breaths, 9.72 breaths/min, and the
same on/offsets, amplitudes and time products. EIT and FluxMed files are not
used (there is no EIT data in A2).

New options this needed, all off by default so existing pipelines do not
change:

- `emg.preprocess`: `notch_before_bandpass` (notch the raw signal first, as
  the multidomain results chain does) and `envelope_method: median` (median of the absolute
  signal, which ignores short spikes such as heartbeat leftovers).
- `emg.detect_breaths`: `merge_close_peaks_within_width` merges peaks closer
  than the minimum breath width, keeping the higher one.

### Biopac `.txt` files with a leading time column were read one column off

Some Biopac exports start every row with a time column (`min` before `CH1`).
The reader labelled that time column as the first channel, so every channel
was shifted by one and the last channel was dropped. The time column is now
skipped and named in `metadata["skipped_time_column"]`.

### `emg.ecg_wavelet_denoising` keeps the preprocessing envelope method

It always rebuilt the envelope as ARV, whatever preprocessing used. It now
uses the same method as preprocessing (like `emg.ecg_gating`), or the one
given in its new `envelope_method` setting. **Pipelines that combine this step
with the default RMS preprocessing now get an RMS envelope instead of ARV.**

### New steps: `emg.slice` and `ventilator.detect_pressure_breaths`

- `emg.slice` / `session.slice_emg(start_seconds, end_seconds=None)` keeps only
  a time window of the loaded EMG recording, for example to leave out a stretch
  where another device disturbed the EMG. Ventilator channels read from the same
  file (such as airway pressure on a Biopac export) are cut the same way, so
  they stay lined up. A window outside the recording raises `ValueError`.
- `ventilator.detect_pressure_breaths` finds spontaneous breaths as dips in the
  airway pressure, for recordings without a ventilator volume channel. It
  writes `ventilator_breath_indices`, so `ventilator.respiratory_rate` and
  `ventilator.normalize_breaths` work on its output. Missing pressure samples
  warn and never hold a breath.

### Respiratory rate with fewer than two breaths warns instead of crashing

`respiratory_rate_from_indices` (used by the `emg.respiratory_rate` and
`ventilator.respiratory_rate` steps) crashed with `index -1 is out of bounds`
when fewer than two breaths were found. It now warns and returns NaN and an
empty breath-to-breath array.

### EMG breath detection accepts breaths from 0.5 s wide (was 1.0 s)

The default `min_breath_width_seconds` of `emg.detect_breaths`,
`session.detect_emg_breaths()` and `run_pipeline("emg")` is now **0.5 s**. At
1.0 s the `emg_data_synth_quiet_breathing` file (about 22 breaths/min) gave
**0** breaths; at 0.5 s it gives 154 (22.0 breaths/min, against 154
ventilator breaths).

**This is a behavior change: EMG breath counts, and everything computed from
them, can differ from previous runs.** Pipelines that set
`min_breath_width_seconds` themselves are not affected. Pass
`min_breath_width_seconds=1.0` to get the old behavior.

### EMG preprocessing no longer analyses channel 0 by default

`preprocess_emg()` (and so `run_pipeline("emg")` and the `emg.preprocess` step)
used channel 0 whenever no `channel` was given. In the
`emg_data_synth_quiet_breathing` file, channel 0 is the ECG, so the breathing
analysis ran on the heart signal.

**This is a behavior change: code that relied on the silent channel 0 either
gets a different channel or an error.**

- When `channel` is not given, it is picked from the channel names: a
  recording with one channel uses it, and a channel named ECG/EKG is never
  picked. If exactly one other channel is left, that one is used.
- If the names do not show which channel is the breathing muscle (for example
  `emg_0` and `emg_1` from the synthetic generator), an `UnresolvedChannelError`
  lists the channels and asks for `channel=`.
- A `channel` you pass is always used as given.

### The EMG preset no longer crashes when saving the respiratory rate

`session.run_pipeline("emg")` and `session.postprocess_emg()` failed with
`ValueError: ... inhomogeneous shape` on every recording with at least two
detected breaths. The respiratory rate is a pair (median rate, rate of each
breath) with two different shapes, and it was saved as one value.

- The pair is now saved as two `ParameterResult`s: `respiratory_rate` (the
  median, a single number) and `respiratory_rate_breath_to_breath` (one value
  per breath, NaN for outlier breaths). Both are in breaths/min.
- A new test runs the EMG preset from start to end on a made-up recording.

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
