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

# Step 1 of 3: shift the raw signals onto a common time axis. This trims
# samples off the start of a recording, so everything downstream - including
# breath detection - works on the shifted signals.
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

# Step 2 of 3: shift the detected breath times by a further hand-entered
# offset. This does not replace step 1 - the two offsets add up. Despite the
# name it does not derive an offset from the breaths themselves; it applies
# the offset you give it to the breath lists. Use it when the offset estimate
# changed after loading, and leave it out otherwise.
session.synchronize_multimodal_breaths(method="manual_offset", offset_seconds={"emg": 0.0})

# Step 3 of 3: match breaths across modalities into LinkedBreath objects.
# Neither step above does this matching - an EIT breath and an EMG breath are
# paired here, by how close their times are once both shifts have been applied.
linked = session.link_breaths(time_tolerance=0.5)

# Measure breath timing across modalities. These read only the breath
# start/end times, never the signal values inside a breath.
multimodal_parameters = session.compute_multimodal_parameters()

session.export_summary("results/multimodal/")
```

## The three timing steps

The three steps above do different things and none replaces another:

| Step | What it moves | When |
|---|---|---|
| `synchronize_raw_modalities` | The raw signals, by trimming samples off the start | Before processing |
| `synchronize_multimodal_breaths` | The detected breath times, by a further offset that adds to the first | After detection, only if the offset estimate changed |
| `link_breaths` | Nothing - it pairs EIT with EMG breaths by how close their times are | Last |

Neither synchronization step pairs breaths across modalities, and neither
works out an offset from the breaths themselves: both take a hand-entered
offset (`method="manual_offset"` is currently the only method either accepts).

## What each new call adds

- `session.link_breaths(time_tolerance=0.5)` returns `list[LinkedBreath]`
  (also stored on `session.linked_breaths`) - each one groups the EIT/EMG/
  ventilator breaths that occurred close together in time. A breath with no
  match in another modality still appears, with only its own slot filled.
  See [../concepts/synchronization.md](../concepts/synchronization.md).
- `session.compute_multimodal_parameters()` turns those links into
  `ParameterResult`s. All three are measures of breath *timing* - they use
  only breath start/end times, never the signal values within a breath:
  - `eit_to_emg_delay` (per breath, seconds, signed): the lag between the
    two modalities' breaths - electromechanical coupling time when measured
    between EMG effort and EIT volume change. A physiological quantity.
  - `eit_emg_duration_difference` (per breath, seconds): how much longer one
    modality's breath is than the other's.
  - `eit_emg_event_agreement` (aggregate, fraction): how often both
    modalities found a breath at all. A quality check on detection and
    synchronization, not an outcome measure.

  If a ventilator breath list was also linked, the same three are produced
  for every other observed modality pair (`eit`/`ventilator`,
  `emg`/`ventilator`). Results are added to `session.parameter_results`
  alongside the per-modality parameters, so they export to the same
  `parameter_results.csv` - see [export-results.md](export-results.md).

  A cross-modality measure that reads signal *values* rather than breath
  times - an EMG-effort-to-EIT-pendelluft coupling index, say - is a
  separate computation, not an extension of this one. See
  [../concepts/parameters.md](../concepts/parameters.md).

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
