# Tutorial: multimodal EIT + EMG (+ ventilator)

This walks through loading EIT and EMG (and, optionally, ventilator) data
into one session, synchronizing them, processing each modality, linking
their breaths, and computing cross-modality timing parameters. For the same
processing (plus ventilator) expressed as a declarative YAML spec, see
`examples/multimodal_full/multimodal-full.pipeline.yaml`,
`examples/multimodal_example/multimodal.pipeline.yaml`, or
`examples/annemijn_multimodal/annemijn.pipeline.yaml`, and
[../pipelines.md](../pipelines.md).

## Step by step

```python
from m3resp import M3Session

session = M3Session()

data_dir = "data/source/synthetic/20260610_153009"
session.load_eit(f"{data_dir}/m3resp_multimodal_1_eit_draeger.bin", vendor="draeger")
session.load_emg(f"{data_dir}/m3resp_multimodal_1_emg.Poly5")

# Align raw signals onto a common time axis before per-modality processing.
session.synchronize_raw_modalities(
    method="manual_offset",
    offset_seconds={"eit": 0.0, "emg": 0.0},
    reference_modality="eit",
)

# Process each modality independently.
session.preprocess_eit()
session.detect_eit_breaths()
session.preprocess_emg()
session.detect_emg_breaths()
session.postprocess_emg()

# Align the detected event lists (in case synchronize_raw_modalities alone
# wasn't enough - e.g. a manually estimated offset changed after loading).
session.synchronize_multimodal_breaths(method="manual_offset", offset_seconds={"emg": 0.0})

# Match breaths across modalities into LinkedBreath objects.
linked = session.link_breaths(time_tolerance=0.5)

# Compute cross-modality timing parameters from the linked breaths.
multimodal_parameters = session.compute_multimodal_parameters()

session.export_summary("results/multimodal/")
```

## What each new call adds

- `session.link_breaths(time_tolerance=0.5)` returns `list[LinkedBreath]`
  (also stored on `session.linked_breaths`) - each one groups the EIT/EMG/
  ventilator breaths that occurred close together in time. A breath with no
  match in another modality still appears, with only its own slot filled.
  See [../concepts/synchronization.md](../concepts/synchronization.md).
- `session.compute_multimodal_parameters()` turns those links into
  `ParameterResult`s: a per-breath `eit_to_emg_delay` (signed timing delay,
  seconds), a per-breath `eit_emg_duration_difference` (breath duration
  difference, seconds), and an aggregate `eit_emg_event_agreement` (fraction
  of linked breaths where both modalities matched). If a ventilator breath
  list was also linked, the same three parameters are produced for every
  other observed modality pair (`eit`/`ventilator`, `emg`/`ventilator`).
  Results are added to `session.parameter_results` alongside the
  per-modality parameters, so they export to the same `parameter_results.csv`
  - see [export-results.md](export-results.md).

```python
for p in multimodal_parameters:
    if p.name == "eit_to_emg_delay":
        print(p.breath_id, p.value, "s")  # signed delay, EMG relative to EIT
```

To compare a specific anchor point instead of breath-start (e.g. peak
inspiration), pass `anchor="peak"`:

```python
session.compute_multimodal_parameters(anchor="peak")
```

To restrict which modality pairs get computed (skipping a pairing you don't
care about), pass `delay_pairs`/`duration_pairs` explicitly:

```python
session.compute_multimodal_parameters(delay_pairs=[("emg", "eit")], duration_pairs=[])
```

## The one-call preset for the synchronization half

```python
session.run_pipeline("multimodal")
```

Calls `synchronize_raw_modalities()` then `synchronize_multimodal_breaths()` - run this
after the per-modality `"eit"`/`"emg"` presets so their breath events
already exist, then call `session.link_breaths()` and
`session.compute_multimodal_parameters()` directly (there is no preset for
those two yet since they're commonly parameterized per study). See
[../developer/pipeline-contracts.md](../developer/pipeline-contracts.md).
