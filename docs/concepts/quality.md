# Quality flags

A {term}`quality flag` is the result of one quality check: did a signal, a breath or a time window pass the check? M3Resp stores each flag as a `QualityFlag` object in `session.quality`.

## Look at the quality flags

```python
for flag in session.quality:
    print(flag.name, flag.passed, flag.severity)

failed_flags = session.quality.failed()
```

To find flags, use `failed`, `for_modality`, `for_category` or `for_signal`.

## What a quality flag holds

| Field | Contents |
|---|---|
| `name` | The name of the check |
| `passed` | `True` if the check passed, `False` if it failed |
| `severity` | How serious a failure is: `"info"`, `"warning"`, `"error"` or `"critical"` |
| `value` | The measured value |
| `threshold` | The limit of the check. With `value`, it shows why the check passed or failed. |
| `message` | A short text about the result |
| `modality`, `category`, `signal_name` | The device, the physical quantity and the signal that the check is about |
| `breath_id`, `start_time`, `end_time` | The breath or the time window that the check is about. With none of these, the flag is about the whole signal. |
| `metadata` | Other details, as a dictionary |

If a signal is good except for a noisy period, make one flag per time window, with `start_time` and `end_time`. The same `signal_name` connects the flags to the signal.

```python
from m3resp import QualityFlag

noise = QualityFlag(
    name="noise_check",
    passed=False,
    severity="warning",
    modality="emg",
    signal_name="envelope",
    start_time=42.0,
    end_time=47.5,
    message="Movement artefact",
)
```

## Checks that could not run

- The EMG quality steps of a workflow give no flag for a check that they could not run. So their flags always show a real result.
- `postprocess_emg` gives a failed flag with severity `"warning"` for a check that it could not run, for example a check that needs ventilator data. The flag has `metadata["skipped"] == True`, and `message` gives the reason.

## Where quality flags come from

`preprocess_eit` and `postprocess_emg` add flags to `session.quality`. Some workflow steps add flags too, for example `ventilator.pocc_quality`. `session.export_summary` writes the flags to `quality_flags.csv`.

If you use the stored data model (see {doc}`provenance`), M3Resp stores the quality flags in the results of a workflow run as `QualityAnnotation` records.

Continue with {doc}`synchronization`.
