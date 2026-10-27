# Migration guide

These pages help you change analysis code that calls `eitprocessing` or `resurfemg` into M3Resp code. Each page lists the direct calls and the M3Resp calls that do the same work.

```{toctree}
:maxdepth: 1

migration/from-eitprocessing
migration/from-resurfemg
```

## How M3Resp uses the two packages

M3Resp reads and slices EIT data and sums impedance with its own code. The remaining EIT calculations use `eitprocessing`; EMG loading and calculations use `resurfemg`. Functions adapted from both packages are listed in [NOTICE.md](https://github.com/M3RESP/m3resp/blob/main/NOTICE.md). The tests in `tests/regression/` check that M3Resp gives the same results as the direct calls with the same settings.

Both packages stay separate. You can still use them without M3Resp.

## Tasks for all modalities

Some tasks that you did by hand have one M3Resp call:

| Task | M3Resp call |
|---|---|
| Keep a note of the settings and the package versions | `session.provenance`, or a `DataModelRecorder` for a full record. See {doc}`../concepts/provenance`. |
| Shift the times of one recording to the clock of another | `session.synchronize_raw_modalities(offset_seconds=...)`. See {doc}`../how-to/synchronize-recordings`. |
| Put a signal on the time points of another signal | `interpolate_to_time(signal, other_signal.time)`. See {doc}`../concepts/synchronization`. |
| Find the breaths of two modalities that belong together | `session.link_breaths(time_tolerance=0.5)`, or `link_breaths_by_time(...)` for your own lists of breaths |
| Calculate the time between the breaths of two modalities | `session.compute_breath_timing_parameters()`, after `session.link_breaths()`. See {ref}`breath-timing-parameters`. |
| Write the results to CSV and JSON files | `session.export_summary(output_dir)`. See {doc}`../how-to/export-results`. |

`interpolate_to_time` uses linear interpolation. The two signals must be on the same clock. To lower the sampling rate, filter the signal with a low-pass filter first.

`session.link_breaths` only matches breaths. Detect the breaths of each modality first.

## Use clock times of the devices

If you know the clock time at which each device started, `compute_offsets_from_timestamps` gives the start time of each recording:

```python
from m3resp import compute_offsets_from_timestamps

# start of each recording, in seconds after midnight
offsets = compute_offsets_from_timestamps("eit", {"eit": 36000.0, "emg": 35998.0})
print(offsets)  # {'eit': 0.0, 'emg': -2.0}

session.synchronize_raw_modalities(offset_seconds=offsets, reference_modality="eit")
```

## Use a function of your own

The session methods can run a function of your own in place of a processing step:

| Argument | Session methods |
|---|---|
| `preprocess=` | `preprocess_eit` |
| `detector=` | `detect_eit_breaths`, `detect_emg_breaths` |
| `postprocess=` | `postprocess_emg` |

A `detector` function gets the processed data and gives a list of breaths. M3Resp changes each breath into a `BreathEvent`. A breath can be a `(start_time, end_time)` pair or a dictionary:

```python
def my_detector(data, **kwargs):
    return [(1.0, 4.0), (4.0, 7.5)]

breaths = session.detect_eit_breaths(detector=my_detector)
```

Continue with {doc}`migration/from-eitprocessing`.
