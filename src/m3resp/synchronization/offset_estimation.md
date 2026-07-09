# Multimodal offset estimation

`m3resp.synchronization.offset_estimation` estimates the constant time offset
that aligns two devices recording the same subject on **independent clocks** —
e.g. a Draeger EIT device (~50 Hz frame rate) and a Biopac amplifier (2 kHz)
carrying airway pressure (Paw) and diaphragm sEMG. The two files share no common
timestamp, so the offset between them has to be recovered from the signals
themselves before the recordings can be cropped onto a common window.

This module was extracted from the interactive marimo viewer
(`tools/visualization_tool/multimodal_vis.py`), which lets you find the offset by
hand. The module makes the same logic reusable, testable, and available as a
declarative pipeline step.

> **Estimate vs. apply.** `M3Session.synchronize_raw_modalities`
> (`session.sync_raw`) *applies* a known offset by cropping. This module
> *finds* the offset. They are complementary.

---

## Offset convention

An **offset** is the timestamp, **on the reference clock**, at which the target
recording's `t = 0` occurs. Equivalently, a sample at target-relative time `t`
maps to reference time `offset + t`.

For the EIT + Biopac case the reference clock is the Biopac timeline and the
target is the EIT recording, so `offset` = "Biopac seconds at EIT t=0" — the same
number the marimo viewer shows as **Effective offset**.

---

## The two estimators

The dataset gives two independent alignment anchors, and they are meant to be
used together — a coarse absolute anchor followed by a fine relative refinement.

### 1. Interference anchor (absolute) — `estimate_offset_from_interference`

Many EIT devices inject interference into a simultaneously-recorded sEMG **only
while the EIT device is running**. When the EIT device stops (in this dataset the
last ~2 minutes of the Biopac recording is EIT-off), the sEMG high-frequency
power drops sharply. That edge pins the EIT recording's **end** onto the Biopac
clock; subtracting the known EIT duration yields the offset directly — no manual
guessing.

How it works:

1. Emphasise high frequencies with a first difference of the sEMG, then smooth
   the magnitude twice (a short window to form a power measure, a longer one to
   suppress burst-to-burst variation).
2. Decimate to `detection_rate_hz`, estimate an "interference present" plateau
   level (median before `end − plateau_guard_seconds`) and an "absent" level
   (median over the final `tail_seconds`).
3. If the plateau exceeds the absent level by `min_power_ratio`, find the
   strongest sustained downward step within the last `search_window_seconds`.
   Accept it if it drops by at least `min_drop_fraction` of the plateau→absent
   range.
4. `offset = edge_time − reference_duration_seconds`.

The detection thresholds are heuristics: **always confirm the returned
`edge_time_seconds` against the `power_time` / `power_values` trace** (the marimo
viewer plots exactly this). Smoothing biases the detected edge slightly *late*
(≈1 s at the default 2 s smoothing window); the cross-correlation refinement
removes that residual.

### 2. Cross-correlation refinement (relative) — `refine_offset_by_crosscorrelation`

EIT global impedance and airway pressure both track the breath cycle.
Cross-correlating them over an overlap window snaps the alignment to within a
breath. Both signals are mapped onto a shared uniform grid over their overlap,
z-scored, and cross-correlated within `± max_lag_seconds`; the peak absolute
correlation gives the lag correction.

Because breathing is quasi-periodic (~3–5 s), cross-correlation only
disambiguates reliably **once a coarse anchor has placed you within one breath** —
run it on top of the interference anchor, not on its own. An optional `stretch`
factor lets you account for slow clock drift
(`reference_t = base_offset + target_t · stretch`).

---

## Public API

All functions are numpy-only and importable from `m3resp.synchronization`.

| Object | Purpose |
| --- | --- |
| `estimate_offset_from_interference(emg_values, sample_frequency, reference_duration_seconds, …)` | Interference-edge anchor on plain arrays. Returns `InterferenceOffsetResult`. |
| `refine_offset_by_crosscorrelation(target_time, target_values, reference_time, reference_values, base_offset_seconds, …)` | Cross-correlation refinement on plain arrays. Returns `CrossCorrelationOffsetResult`. |
| `estimate_sync_offset(method=…, emg=…, target=…, reference=…, …)` | Orchestrates both, dispatched by `method`. Returns `SyncOffsetResult`. |
| `interference_power(emg_values, sample_frequency, …)` | The high-frequency power envelope (exposed for plotting). |
| `estimate_offset_from_interference_signal(emg, reference_duration_seconds, …)` | `TimeSeries` wrapper for the interference anchor. |
| `refine_offset_by_crosscorrelation_signals(target, reference, base_offset_seconds, …)` | `TimeSeries` wrapper for the refinement. |

`estimate_sync_offset` methods:

- `"manual"` — return `manual_offset_seconds` unchanged.
- `"interference"` — interference anchor only (falls back to the manual offset if
  no edge is found).
- `"crosscorrelation"` — cross-correlation refinement of the manual offset.
- `"interference+crosscorrelation"` — anchor, then refine (recommended).

Result dataclasses (`InterferenceOffsetResult`, `CrossCorrelationOffsetResult`,
`SyncOffsetResult`) carry both the scalar answer and the diagnostic arrays
(power trace, correlation curve) so you can verify a fit visually.

---

## Programmatic usage

```python
from m3resp.synchronization import estimate_sync_offset
from m3resp.data.timeseries import TimeSeries

# emg_ts:  diaphragm sEMG (full, un-cropped, incl. the EIT-off tail)
# gi_ts:   EIT global impedance
# paw_ts:  airway pressure (Paw)

result = estimate_sync_offset(
    method="interference+crosscorrelation",
    emg=emg_ts,
    target=gi_ts,
    reference=paw_ts,
)
print(result.offset_seconds, result.source)

# Then apply it (Stage-1 cropping is relative to a reference modality):
session.synchronize_raw_modalities(
    method="manual_offset",
    offset_seconds={"eit": 0.0, "vent": -result.offset_seconds, "emg": -result.offset_seconds},
    reference_modality="eit",
)
```

Or call the estimators directly on arrays — see the docstrings for every tuning
knob (`detection_rate_hz`, `search_window_seconds`, `plateau_guard_seconds`,
`tail_seconds`, `min_power_ratio`, `min_drop_fraction`, `max_lag_seconds`,
`grid_rate_hz`, `stretch`, `window`).

---

## Pipeline step: `sync.estimate_offset`

Registered in `m3resp.workflows.steps.sync`. It reads the **raw, un-cropped**
signals straight from the session (`session.emg`, `session.raw["vent"]`,
`session.eit.global_impedance`), so it must run **after the `*.load` steps and
before `session.sync_raw`**.

It writes two context artifacts:

- `estimated_offset_seconds` — the float offset.
- `offset_estimation` — a JSON-friendly summary (per-estimator detail), also
  stored on `session.parameters["offset_estimation"]` for provenance/QA.

Parameters:

| Parameter | Default | Meaning |
| --- | --- | --- |
| `method` | `"interference"` | any `estimate_sync_offset` method |
| `emg_source` / `emg_channel` | `"emg"` / `0` | where the interference-bearing sEMG lives: `"emg"` (dedicated EMG file) or `"vent"` (sEMG shares the Biopac ventilator file) |
| `target_source` / `target_channel` | `"eit"` / `0` | breathing target, normally EIT global impedance |
| `reference_source` / `reference_channel` | `"vent"` / `0` | reference breathing signal for cross-correlation, normally Paw |
| `reference_duration_seconds` | `None` | override for the EIT duration (defaults to the target's duration) |
| `manual_offset_seconds` | `0.0` | fallback / base offset |
| `interference_kwargs`, `crosscorrelation_kwargs` | `None` | pass-through tuning dicts |

> **Match the sources to how you loaded the data.** In this dataset the
> interference-bearing sEMG is channel 2 of the Biopac `.txt`. If that `.txt` is
> loaded as the ventilator modality, set `emg_source: vent`, `emg_channel: 1`
> (0-based). If a dedicated EMG file is loaded, keep `emg_source: emg`.

Spec fragment:

```yaml
steps:
  - uses: eit.load
    with: { file: "@eit_file", vendor: draeger }
  - uses: emg.load_ventilator
    with: { file: "@vent_file" }

  # Estimate the offset from the raw signals (before any cropping)
  - uses: sync.estimate_offset
    with:
      method: interference
      emg_source: vent        # sEMG is in the Biopac ventilator file …
      emg_channel: 1          # … channel 2 (0-based)
```

### Applying the estimate

The pipeline engine binds a step's `with:` values from literals and `@input`
references only — it **cannot** yet bind one step's output (`estimated_offset_seconds`)
into another step's `with:` (`session.sync_raw`'s `offset_seconds` is not part of
its `reads`). So today the flow is:

1. Run `sync.estimate_offset` in the pipeline — it computes and **records** the
   offset (context + `session.parameters`), which is enough for inspection/QA and
   for any custom code reading the context.
2. Apply it either **programmatically** (recipe above) or by reading the reported
   value and putting it in `session.sync_raw`'s `with: offset_seconds`.

A single `manual_offset` crop only trims one end of each modality, so for a Biopac
recording that both *precedes* and *follows* the EIT recording (leading lead-in +
trailing EIT-off tail) you will generally window it in code rather than with one
`sync.sync_raw` offset. Auto-applying the crop is deliberately left out of this
step to avoid an incorrect one-sided trim.

---

## Caveats

- **Heuristic detection.** The interference thresholds assume a clear EIT-off
  region in the tail and interference that is strong relative to the resting
  sEMG. Verify the detected edge against the power trace; tune the knobs if the
  edge is misplaced.
- **Periodicity ambiguity.** Cross-correlation of near-sinusoidal breathing is
  ambiguous by ~half a breath period; only trust it once the interference anchor
  has you within a breath.
- **Constant offset only.** Both estimators assume a single constant offset (plus
  an optional linear `stretch`). They do not model non-linear drift.
- **Needs the un-cropped sEMG.** The interference edge lives in the EIT-off tail;
  estimate before any synchronization crop removes it.
