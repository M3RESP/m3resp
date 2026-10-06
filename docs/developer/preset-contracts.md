# Preset contracts

Presets run fixed sequences of processing operations on a loaded session.
For custom YAML/JSON step sequences, see [Workflows](../workflows.md).

## `Preset` (`src/m3resp/presets/base.py`)

```python
class Preset(ABC):
    name: str

    @abstractmethod
    def run(
        self, session: M3Session, *, config: PresetConfig | None = None
    ) -> M3Session:
        ...
```

`PresetConfig` is `Mapping[str, Mapping[str, Any]]`: keyword arguments
grouped by the configuration keys defined by each preset (e.g.
`{"preprocess": {...}, "detect_breaths": {...}}`).

A concrete `Preset.run` calls session methods or registered steps in a
fixed order. Those operations store signals, measurements, quality flags
and provenance on the session. The supplied session is updated and returned.

## Built-in presets and the registry

| Preset | `name` | Calls |
|---|---|---|
| `EITPreset` | `"eit"` | `session.preprocess_eit(...)`, `session.detect_eit_breaths(...)` |
| `EMGPreset` | `"emg"` | `session.preprocess_emg(...)`, `emg.ecg_detect_peaks` + `emg.ecg_gating`, a moving baseline (`session.emg_adapter.moving_baseline`), `session.detect_emg_breaths(...)`, `session.postprocess_emg(...)` |
| `MultimodalPreset` | `"multimodal"` | `session.synchronize_raw_modalities(...)`, `session.synchronize_multimodal_breaths(...)` |

`presets/registry.py` maps each `name` to its class via `register_preset`/
`get_preset`. `M3Session.run_preset(name, config=...)` looks the preset up
and calls it:

```python
session.run_preset("eit")
session.run_preset("emg", config={"preprocess": {"variant": "native"}})
session.run_preset("multimodal")
```

`EMGPreset` calls the registered steps `emg.ecg_detect_peaks` and
`emg.ecg_gating`, which update the session's signals and provenance.
Its default processing order is:
band-pass -> ECG peak detection -> gating -> envelope -> baseline -> breath
detection -> postprocessing. The baseline is the quiet level of the envelope
that the breath-detection threshold is measured against, so it is computed
before breaths are detected; set it with
`config={"baseline": {"window_seconds": ..., "step_seconds": ..., "percentile": ...}}`
(defaults: 30 s window, 1 s step, 33rd percentile).
`config={"ecg_removal": {"enabled": False}}` skips ECG removal; cardiac
activity can then remain in the envelope and derived measurements. Pass
`config={"ecg_detect_peaks": {"ecg_channel": n}}` when a dedicated reference
ECG channel was recorded, or
`config={"ecg_removal": {"ecg_peak_indices": [...]}}` to gate already-known
peaks and skip detection. Supplied peaks and nonempty `ecg_detect_peaks`
settings are mutually exclusive and raise `TypeError` when combined.

## Choosing how to run processing

- `m3resp.run_workflow(spec, session=...)` (module-level) runs a fully
  custom YAML/JSON step-list spec built from individually composable steps
  (`eit.mdn_filter`, `emg.ecg_gating`, ...), documented in
  [Workflows](../workflows.md).
  Most of these steps take a `session` binding and populate the typed
  collections and record provenance through the same `M3Session._record()`
  session method the presets use (see `_record_step` in each modality's
  `_shared.py`) - `eit.roi_amplitude_lungspace`, `emg.ecg_gating`, and so on
  all do this. The exception is the small set of pure per-breath feature
  steps (e.g. `emg.time_to_peak`, `emg.amplitude`) that operate on
  extracted arrays and return calculated values. Use this for custom or
  batch workflows where the sequence of operations varies per project.
- `session.run_preset("eit" | "emg" | "multimodal", config=...)` (a method
  on `M3Session`, this page) runs one of the small, built-in
  presets, each a fixed sequence of processing operations. Use this for
  running one modality with default behavior or configured options.

## Adding a new preset

1. Add a class in `presets/*.py` implementing `Preset.run`, calling only
   existing session methods or registered processing steps.
2. Register it: `register_preset("my_name", MyPreset)` in
   `presets/registry.py`.
3. Expose the operations' configurable options through `config` and document
   the configuration keys used by the preset.
