# Tutorial: EMG-only processing

This walks through loading, preprocessing, detecting breaths, and
postprocessing a single EMG recording using `M3Session` directly. For the
same processing expressed as a declarative YAML spec, see
`examples/emg_full_preprocessing/emg-full.pipeline.yaml` and
[../pipelines.md](../pipelines.md).

## Step by step

```python
from m3resp import M3Session

session = M3Session()

session.load_emg(
    "data/source/synthetic/20260610_153009/m3resp_multimodal_1_emg.Poly5"
)

session.preprocess_emg()      # bandpass filter + envelope -> session.signals
session.detect_emg_breaths()  # -> session.events["emg_breaths"]
session.postprocess_emg()     # features + quality -> parameter_results/quality

session.export_summary("results/emg-only/")
```

After this:

- `session.signals` has the raw/filtered/envelope EMG `Signal`s (see
  [../concepts/signals.md](../concepts/signals.md)).
- `session.events["emg_breaths"]` has the detected `BreathEvent`s.
- `session.parameter_results` has amplitude/AUC/pseudo-slope/time-to-peak/
  respiratory-rate `ParameterResult`s (see
  [../concepts/parameters.md](../concepts/parameters.md)).
- `session.quality` has the native `resurfemg` clinical quality checks
  (Pocc prerequisites, SNR, baseline crossing, etc.) as `QualityFlag`s.
- `results/emg-only/` has the structured export files (see
  [export-results.md](export-results.md)).

## The one-call preset

```python
session.run_pipeline("emg")
```

Calls `preprocess_emg()`, `detect_emg_breaths()`, then `postprocess_emg()`
in sequence. Pass `config={"preprocess": {...}, "detect_breaths": {...}, "postprocess": {...}}`
to override any call's keyword arguments. See
[../developer/pipeline-contracts.md](../developer/pipeline-contracts.md).

## ECG removal and other advanced operations

`preprocess_emg`'s default path handles the common bandpass + envelope
case. For ECG-contaminated EMG (e.g. diaphragm EMG near the heart), ECG
removal (gating, wavelet denoising, or the native estimated-ECG-subtraction
alternative), custom baselines, and Pocc-specific quality checks are exposed
individually on `ReSurfEMGAdapter` (see
[../developer/adapters.md](../developer/adapters.md)) and as composable
steps in the declarative pipeline engine - see "ECG-removal alternatives" in
[../pipelines.md](../pipelines.md) for the full comparison and when to use
each.

Any `resurfemg.postprocessing` function not covered by a named wrapper is
still reachable without leaving the adapter boundary:

```python
session.emg_adapter.run_postprocessing_function(
    "quality_assessment", "some_function_name", *args, **kwargs
)
```
