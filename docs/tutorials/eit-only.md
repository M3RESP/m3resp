# Tutorial: EIT-only processing

This walks through loading, preprocessing, detecting breaths, and exporting
a single EIT recording using `M3Session` directly. For the same processing
expressed as a declarative YAML spec (every individual operation, not just
the default preset), see `examples/eit_full_preprocessing/eit-full.pipeline.yaml`
and [../pipelines.md](../pipelines.md).

## Step by step

```python
from m3resp import M3Session

session = M3Session()

session.load_eit(
    "data/source/synthetic/20260610_153009/m3resp_multimodal_1_eit_draeger.bin",
    vendor="draeger",
)

session.preprocess_eit()      # filters + derives TIV/EELI/rate -> signals/parameter_results/quality
session.detect_eit_breaths()  # -> session.events["eit_breaths"]

session.export_summary("results/eit-only/")
```

After this:

- `session.signals` has the raw/filtered/processed EIT `Signal`s (see
  [../concepts/signals.md](../concepts/signals.md)).
- `session.parameter_results` has TIV/EELI/rate `ParameterResult`s (see
  [../concepts/parameters.md](../concepts/parameters.md)).
- `session.quality` has any EIT quality flags.
- `session.events["eit_breaths"]` has the detected `BreathEvent`s.
- `results/eit-only/` has the structured export files (see
  [export-results.md](export-results.md)).

## The one-call preset

The three processing calls above (minus loading/exporting) are also
available as a single named preset:

```python
session.run_pipeline("eit")
```

This calls `session.preprocess_eit()` then `session.detect_eit_breaths()` in
sequence - identical behavior, just a shorter call for the common case. Pass
`config={"preprocess": {...}, "detect_breaths": {...}}` to override either
call's keyword arguments. See
[../developer/pipeline-contracts.md](../developer/pipeline-contracts.md).

## Choosing what gets computed

`preprocess_eit` owns filtering and derived-parameter output: keyword
arguments control filter mode (`mdn`, `lowpass`, `bandpass`, `none`),
breath-interval detection (`breath_min_duration_seconds`), and which optional
parameters to compute on top of that (rates, TIV, EELI, pixel TIV) - see
`EITProcessingAdapter.preprocess` in
[../developer/adapters.md](../developer/adapters.md) for the exact mapping.
Note that rate detection runs before filtering when `filter_mode="mdn"`,
since MDN filtering is itself parameterized by the detected rates - so
splitting filtering and parameter computation into separate calls isn't a
clean before/after split.

`detect_eit_breaths` only normalizes breath detections into `BreathEvent`s;
its keyword arguments are passed straight through to the detector (or to a
custom `detector=callable`) and don't affect what `preprocess_eit` computes.
A custom `preprocess=callable` bypasses the typed-collection conversion
entirely, for algorithms not yet covered by the adapter.
