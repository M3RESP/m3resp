# Preset contracts

A {term}`preset` runs a fixed list of session methods on a loaded session. A {doc}`workflow file <../workflows>` runs a list of steps that you choose.

## The Preset class

Each preset is a subclass of `Preset` in `src/m3resp/presets/base.py`:

```python
class Preset(ABC):
    name: str

    @abstractmethod
    def run(self, session: M3Session, *, config: PresetConfig | None = None) -> M3Session:
        ...
```

`PresetConfig` is a dictionary of dictionaries. Each key names one operation of the preset, and its value holds the keyword arguments of that operation. `run` changes the session and returns it.

## The built-in presets

| Class | Name | Operations, in order | `config` keys |
|---|---|---|---|
| `EITPreset` | `"eit"` | `preprocess_eit`, `detect_eit_breaths` | `preprocess`, `detect_breaths` |
| `EMGPreset` | `"emg"` | `preprocess_emg`, the steps `emg.ecg_detect_peaks`, `emg.ecg_gating` and `emg.envelope`, `emg_adapter.moving_baseline`, `detect_emg_breaths`, `postprocess_emg` | `preprocess`, `ecg_detect_peaks`, `ecg_gating`, `ecg_removal`, `envelope`, `baseline`, `detect_breaths`, `postprocess` |
| `MultimodalPreset` | `"multimodal"` | `synchronize_raw_modalities`, `synchronize_multimodal_breaths` | `synchronize_raw`, `align` |

`session.run_preset(name, config=...)` finds the preset by its name and runs it:

```python
session.run_preset("eit")
session.run_preset("emg", config={"preprocess": {"channel": 1}})
```

See {doc}`../how-to/emg-processing-settings` for the settings of the `"emg"` preset.

## Rules of the EMG preset

- After the ECG removal, the step `emg.envelope` calculates the envelope of the gated signal. It uses the window and the method that `preprocess_emg` used. `"envelope": {"window_seconds": ..., "method": ...}` changes them.
- The baseline comes after this envelope, so it gives the quiet level of the gated signal.
- The baseline comes before the breath detection, because the detection measures the envelope against the baseline. The default baseline is the 33rd percentile in a window of 7.5 s, with a step of 0.2 s.
- `"ecg_removal": {"enabled": False}` turns the ECG removal off.
- `"ecg_removal": {"ecg_peak_indices": [...]}` gives the R peaks and skips the peak detection. If you also give settings in `"ecg_detect_peaks"`, the preset raises `TypeError`.

## Preset or workflow file

| Use | When |
|---|---|
| `session.run_preset(...)` | You process one modality with the default steps and some changed settings |
| `run_workflow(spec, session=...)` or `run_spec(...)` | You choose the steps, their order and their settings, for example for one project or a batch of recordings |

Most workflow steps store their results in the session and add a provenance record, as the session methods do. Some EMG feature steps, for example `emg.amplitude` and `emg.time_to_peak`, only calculate values from arrays.

## Add a preset

1. Add a subclass of `Preset` in a new file in `src/m3resp/presets/`. Use only session methods and registered steps in `run`.
2. Register it with `register_preset("my_name", MyPreset)` in `presets/registry.py`.
3. Give each operation a `config` key, and list the keys in the docstring of the class.

Continue with {doc}`testing`.
