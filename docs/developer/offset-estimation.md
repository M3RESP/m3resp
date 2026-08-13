# Multimodal offset estimation (marimo-viewer utilities)

`tools/visualization_tools/utils/offset_estimation` estimates the constant
time offset that aligns two devices recording the same subject on
**independent clocks** — e.g. a Draeger EIT device (~50 Hz frame rate) and a
Biopac amplifier (2 kHz) carrying airway pressure (Paw) and diaphragm sEMG.
The two files share no common timestamp, so the offset between them has to be
recovered from the signals themselves before the recordings can be cropped
onto a common window.

**This is a marimo-viewer utility, not part of the installable `m3resp`
package.** The estimators here are protocol-specific — they were tuned
against the Annemijn `eit_emg` dataset's specific acquisition artifact and are
not a robust, general-purpose automatic sync method. `m3resp.synchronization`
itself only supports a manual offset (`sync.estimate_offset`'s `method`
defaults to `"manual"`); use `2_annemijn_multimodal_vis.py` interactively to
find an offset by hand for a given recording, then hardcode the result as
`manual_offset_seconds` in your pipeline spec (see the Annemijn example).

> **Estimate vs. apply.** `M3Session.synchronize_raw_modalities`
> (`session.sync_raw`) *applies* a known offset by cropping. This module
> *finds* the offset, for interactive/manual use only. They are complementary.

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

The Annemijn dataset gives two alignment cues that can be used together — a
recording-protocol-specific coarse anchor followed by a fine relative
refinement.

### 1. Protocol artifact anchor (absolute) — `estimate_offset_from_interference`

In the Annemijn recording, the way the EIT and Biopac acquisition were run left
a visible artifact in the raw diaphragm sEMG: during the EIT acquisition the sEMG
contained stronger high-frequency activity, and after the EIT recording stopped
that activity dropped sharply. This is **not** a general property of EIT devices
or a feature that should be assumed for every dataset. It is a protocol-specific
artifact that can be used only when the raw recording contains a clear EIT-on to
EIT-off transition.

For this dataset, the downward step pins the EIT recording's **end** onto the
Biopac clock; subtracting the known EIT duration yields the offset directly.

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

More technically, the raw sEMG is transformed into a low-rate interference-power
envelope before edge detection:

```text
emg[n]                  raw diaphragm EMG samples
hf[n] = abs(emg[n] - emg[n-1])
power_fast[n] = mean(hf over fast_window_seconds)
power[n] = mean(power_fast over smooth_window_seconds)
power_ds = power sampled at detection_rate_hz
```

The first difference behaves like a simple high-pass operation: slow baseline
drift and the respiratory EMG envelope change little from one sample to the next,
while sharp sample-to-sample activity grows. Taking `abs(...)` turns this into an
energy-like signal without caring about polarity. The two moving averages then
turn the noisy high-frequency activity into an envelope:

- `fast_window_seconds` (`0.5 s` by default) converts sample-level activity into a
  local power estimate.
- `smooth_window_seconds` (`2.0 s` by default) suppresses breath-by-breath EMG
  variation so this recording artifact's on-to-off transition becomes the
  dominant feature.

The detector then estimates two robust baseline levels from medians:

```text
end = last EMG timestamp
high = median(power_ds where time < end - plateau_guard_seconds)
low  = median(power_ds where time > end - tail_seconds)
```

With defaults, `high` is estimated before the final `180 s` guard region, where
this artifact is expected to be present in the Annemijn recording. `low` is
estimated from the last `45 s`, where the artifact is expected to be absent. The
edge search only runs if:

```text
high > low * min_power_ratio
```

This avoids forcing an edge when the tail does not actually look quieter than the
recording plateau. For other datasets, this condition may fail simply because the
artifact was never present; that is expected and should be treated as "no
interference anchor available".

The edge itself is found by scanning candidate times in the last
`search_window_seconds` (`300 s` by default). For each candidate, the algorithm
compares the mean power in a window before the candidate with the mean power in a
window after it:

```text
before = mean(power_ds[i - step_window : i])
after  = mean(power_ds[i : i + step_window])
drop   = before - after
```

`step_window_seconds` is `20 s` by default. The candidate with the largest
positive `drop` is the strongest sustained downward step. It is accepted only if:

```text
drop > min_drop_fraction * (high - low)
```

With the default `min_drop_fraction = 0.25`, the detected step must explain at
least a quarter of the observed plateau-to-tail decrease. Once accepted,
`edge_time_seconds` is interpreted as the EIT stop time on the Biopac clock, and
the EIT start time is:

```text
offset_seconds = edge_time_seconds - eit_duration_seconds
```

That offset is the absolute anchor used by the combined synchronization mode when
this protocol-specific artifact is present.

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

### Combined mode: `interference+crosscorrelation`

The combined mode is sequential. It does **not** average the two estimators:

1. Run `estimate_offset_from_interference` on the raw sEMG trace. This returns an
   absolute offset by detecting the protocol-specific EIT-off artifact edge in
   the Biopac clock and subtracting the EIT duration.
2. Use that offset as `base_offset_seconds` for
   `refine_offset_by_crosscorrelation`.
3. Cross-correlate the target breathing signal, usually EIT global impedance,
   against the reference breathing signal, usually Paw, over the configured
   target-relative window.
4. Add the signed lag correction to the interference offset:

   ```text
   final_offset_seconds = interference_offset_seconds + lag_seconds
   ```

Inside the cross-correlation step, the reference signal is first moved into the
target's relative clock:

```text
reference_relative_time = (reference_time - base_offset_seconds) / stretch
```

Both signals are then interpolated onto the same grid, mean-centered,
z-scored, and correlated for lags inside `± max_lag_seconds`. The code chooses
the peak by absolute correlation, so inverse polarity is accepted. This matters
for EIT-vs-Paw data because the traces can breathe together while having
opposite signs.

The returned `lag_seconds` is the correction to apply to the base offset. A
negative lag means "start the Biopac crop earlier relative to the EIT start";
a positive lag means "start it later".

For the Annemijn multimodal example, the current pipeline configuration gives:

- interference edge: `1697.20 s` on the Biopac clock
- EIT duration: `640.98 s`
- interference offset: `1056.22 s`
- cross-correlation correction: `-5.60 s`
- final offset: `1050.62 s`
- peak correlation: about `-0.57`

The negative peak correlation is still useful: the implementation searches the
largest absolute correlation, because opposite polarity still indicates matched
respiratory timing.

---

## Public API

All functions are numpy-only and importable from
`tools.visualization_tools.utils.offset_estimation` (not from
`m3resp.synchronization` - see the note at the top of this file).

| Object | Purpose |
| --- | --- |
| `estimate_offset_from_interference(emg_values, sample_frequency, reference_duration_seconds, …)` | Interference-edge anchor on plain arrays. Returns `InterferenceOffsetResult`. |
| `refine_offset_by_crosscorrelation(target_time, target_values, reference_time, reference_values, base_offset_seconds, …)` | Cross-correlation refinement on plain arrays. Returns `CrossCorrelationOffsetResult`. |
| `interference_power(emg_values, sample_frequency, …)` | The high-frequency power envelope (exposed for plotting). |
| `estimate_offset_from_interference_signal(emg, reference_duration_seconds, …)` | `TimeSeries` wrapper for the interference anchor. |
| `refine_offset_by_crosscorrelation_signals(target, reference, base_offset_seconds, …)` | `TimeSeries` wrapper for the refinement. |

Result dataclasses (`InterferenceOffsetResult`, `CrossCorrelationOffsetResult`)
carry both the scalar answer and the diagnostic arrays (power trace,
correlation curve) so you can verify a fit visually - exactly what the marimo
viewer plots.

---

## Interactive usage (marimo viewer)

```python
from tools.visualization_tools.utils.offset_estimation import (
    estimate_offset_from_interference_signal,
    refine_offset_by_crosscorrelation_signals,
)

# emg_ts:  diaphragm sEMG (full, un-cropped, incl. the artifact-off tail)
# gi_ts:   EIT global impedance
# paw_ts:  airway pressure (Paw)

interference = estimate_offset_from_interference_signal(
    emg_ts, reference_duration_seconds=gi_ts.duration
)
refined = refine_offset_by_crosscorrelation_signals(
    gi_ts, paw_ts, interference.offset_seconds
)
print(refined.refined_offset_seconds)
```

Once you've confirmed the fit against the power/correlation traces (see
`2_annemijn_multimodal_vis.py`), hardcode the resulting offset as
`manual_offset_seconds` in your pipeline spec's `sync.estimate_offset` step -
see `examples/annemijn_multimodal/annemijn.pipeline.yaml` for the actual
values used for this dataset.

Or call the estimators directly on arrays - see the docstrings for every tuning
knob (`detection_rate_hz`, `search_window_seconds`, `plateau_guard_seconds`,
`tail_seconds`, `min_power_ratio`, `min_drop_fraction`, `max_lag_seconds`,
`grid_rate_hz`, `stretch`, `window`).

---

## Caveats

- **Protocol-specific artifact.** The interference anchor is valid only for
  recordings where the specific acquisition setup left a visible high-frequency
  sEMG artifact that disappears when EIT stops. It is not a general EIT-device
  feature.
- **Heuristic detection.** The interference thresholds assume a clear off region
  in the tail and an artifact that is strong relative to the resting sEMG. Verify
  the detected edge against the power trace; tune the knobs if the edge is
  misplaced.
- **Periodicity ambiguity.** Cross-correlation of near-sinusoidal breathing is
  ambiguous by ~half a breath period; only trust it once the interference anchor
  has you within a breath.
- **Constant offset only.** Both estimators assume a single constant offset (plus
  an optional linear `stretch`). They do not model non-linear drift.
- **Needs the un-cropped sEMG.** When this artifact exists, the edge lives in the
  EIT-off tail; estimate before any synchronization crop removes it.
