# Tutorial: multimodal EIT + EMG (+ ventilator)

This walks through loading EIT and EMG (and, optionally, ventilator) data
into one session, synchronizing them, processing each modality, linking
their breaths, and computing cross-modality timing parameters. For the same
processing (plus ventilator) expressed as a declarative YAML spec, see
`examples/multimodal_full/multimodal-full.pipeline.yaml`
or `examples/multimodal_example/multimodal.pipeline.yaml`, and
[../pipelines.md](../pipelines.md).

## Step by step

```python
from m3resp import M3Session

session = M3Session()

data_dir = "data/source/synthetic/20260610_153009"
session.load_eit(f"{data_dir}/m3resp_multimodal_1_eit_draeger.bin", vendor="draeger")
session.load_emg(f"{data_dir}/m3resp_multimodal_1_emg.Poly5")

# Step 1 of 3: put the recordings on a common clock by giving each one a
# start time, in seconds from the start of the reference recording. A
# negative value means that recording started earlier, a positive value that
# it started later. No samples are removed: every recording keeps its full
# length, and each modality's own results (breath times, signals) stay on
# its own clock. The start times are added when breaths are compared across
# modalities in steps 2 and 3. Skipping this step makes link_breaths warn
# that the recordings were never synchronized; if they really did start
# together, call session.skip_synchronization() instead.
session.synchronize_raw_modalities(
    method="manual_offset",
    offset_seconds={"eit": 0.0, "emg": 0.0},
    reference_modality="eit",
)

# Process each modality independently.
session.preprocess_eit()
session.detect_eit_breaths()
session.preprocess_emg(channel=1)  # channel 1 is the breathing muscle signal
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
| `synchronize_raw_modalities` | Nothing yet - it stores each recording's start time in `session.start_times`, which steps 2 and 3 add to the breath times | Before processing |
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
  - `eit_to_emg_delay` (per breath, seconds, signed): the EMG breath anchor
    minus the EIT breath anchor. Read it as a check on detection and
    alignment, not as an outcome measure. With the default `anchor="start"`
    the two sides are not the same kind of landmark: the EIT start is a
    detected breath start, while the EMG start is built from the envelope
    peak by subtracting a fixed half-window (`half_window_seconds`, 0.5 s by
    default), so changing that setting shifts the delay by the same amount.
    With `anchor="peak"` it compares the EIT breath middle against the EMG
    envelope peak. Calling this electromechanical coupling time would need
    the EMG anchor to be diaphragm activation onset and the EIT anchor the
    start of volume change, defined consistently and validated against each
    other; that is separate work.
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
