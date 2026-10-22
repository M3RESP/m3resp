# Offset estimation

When two devices record on separate clocks, the files have no shared time. The time offset between them must come from the signals. M3Resp takes this offset as a number. This page shows how a workflow uses the offset, and two methods that find it.

## The offset

The offset is the time on the source clock at which the target recording starts. A sample at time `t` of the target recording is at time `offset + t` on the source clock.

For example, an EIT device is the target and a Biopac amplifier with EMG and airway pressure is the source. Then the offset is the Biopac time at the first EIT frame.

## Use the offset in a workflow

The step `sync.estimate_offset` stores the offset. The step `sync.apply_estimated_offset` then gives each recording its {term}`start time`:

```yaml
steps:
  - uses: sync.estimate_offset
    with: { method: manual, manual_offset_seconds: 2.0 }
  - uses: sync.apply_estimated_offset
    with: { target_modality: eit, source_modalities: [emg] }
```

The target recording starts at 0 s. Each source recording starts at the negative offset, here at -2.0 s. M3Resp keeps all samples. `"manual"` is the only `method` of `sync.estimate_offset`.

## Find the offset

The team finds the offset with two methods in an interactive marimo notebook, with the code in `tools/visualization_tools/utils/offset_estimation.py`. Both methods suit one recording protocol. Always look at the result in a plot before you use the offset.

### Method 1: the end of the EIT interference in the EMG

In some recordings, the EMG has more high-frequency noise while the EIT device measures. When the EIT recording stops, the noise drops. The time of this drop on the Biopac clock is the end of the EIT recording, so:

```text
offset = time of the drop - duration of the EIT recording
```

The method finds the drop in four steps:

1. Calculate the noise power: the absolute first difference of the EMG, then a moving mean over 0.5 s, then a second moving mean over 2.0 s. Keep 10 values per second.
2. Calculate the noise level during the EIT recording: the median power before the last 180 s. Calculate the level after the EIT recording: the median power of the last 45 s.
3. Stop if the first level is less than 1.25 times the second level. Then the recording has no clear drop.
4. In the last 300 s, find the time with the largest difference between the mean power of 20 s before and 20 s after. Use this time if the difference is at least 25% of the difference between the two levels.

The moving means move the drop about 1 s later. Method 2 corrects this.

### Method 2: cross-correlation of two breathing signals

The EIT global impedance and the airway pressure both follow the breaths. The method moves the pressure to the EIT clock with the offset of method 1. Then it puts both signals on one time grid of 50 Hz, scales them to a mean of 0 and a standard deviation of 1, and calculates the cross-correlation. The search covers lags up to ±5 s.

The lag with the highest absolute correlation is the correction:

```text
final offset = offset of method 1 + lag
```

The method uses the absolute correlation, because the impedance and the pressure can move in opposite directions during a breath. A negative peak correlation still shows matched breaths.

Breaths repeat every 3 to 5 s, so the cross-correlation can match the wrong breath. Use method 2 only after method 1 has put the signals within one breath.

## Limits of the methods

- Method 1 needs a recording where the EIT device adds noise to the EMG. Other recordings can have no drop.
- Method 1 needs the full EMG recording, with the part after the end of the EIT recording. Find the offset before you cut the EMG to a time window.
- Both methods find one constant offset. Method 2 can also correct a linear clock drift with its `stretch` setting. Neither method corrects a drift that changes over time.
- The limits in method 1 (180 s, 45 s, 300 s, 1.25 and 25%) suit one recording protocol. Check them for each new protocol.

Continue with {doc}`processing-defaults`.
