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
| `threshold` | The limit value for passing or failing the check. Taken together with `value`, it shows why the check passed or failed. |
| `message` | An explanation in words, for example why the check could not run (`"Needs at least two ventilator breaths."`). Most checks leave it empty (`None`). |
| `modality`, `category`, `signal_name` | The device, the physical quantity and the signal that the check is about |
| `breath_id`, `breath_ids`, `start_time`, `end_time` | The breath, the breaths or the time window that the check is about. With none of these, the flag is about the whole signal. |
| `metadata` | Other details, as a dictionary |

The severity is a fixed label that each check sets. The measured value does not change it, and M3Resp does not use it to stop or skip any step.

| Severity | Meaning |
|---|---|
| `"info"` | Shows a property of the data. A failure alone does not make the result wrong. |
| `"warning"` | Look at the data before you use the result. |
| `"error"` | The result is probably wrong. |
| `"critical"` | The data cannot be used. |

The current checks use `"info"` and `"warning"` only.

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

A flag about one breath sets `breath_id`, as a string. A flag about several breaths together sets `breath_ids`, as a list of strings, for example ten breaths in a stable period, or breaths with gaps between them. Set only one of the two fields: both fields, or an empty `breath_ids` list, give a `ValueError`. The steps that check each breath use the position of the breath in its list as the id (`"0"`, `"1"`, ...).

```python
stable = QualityFlag(
    name="stable_stretch",
    passed=True,
    severity="info",
    modality="eit",
    breath_ids=["3", "4", "5", "6"],
)
```

In `quality_flags.csv`, `breath_ids` is a Python list (`['3', '4', '5', '6']`), and `ast.literal_eval` reads it back. In `summary.json`, it is a JSON list.

## Checks that could not run

- The EMG quality steps of a workflow give no flag for a check that they could not run: if a flag exists, the check was performed. Their flags only exist for real results.
- `postprocess_emg` gives a failed flag with severity `"warning"` for a check that it could not run, for example a check that needs ventilator data. The flag has `metadata["skipped"] == True`, and `message` gives the reason.

## Where quality flags come from

`preprocess_eit` and `postprocess_emg` add flags to `session.quality`. Some workflow steps add flags too, for example `ventilator.pocc_quality`. `session.export_summary` writes the flags to `quality_flags.csv`.

If you use the stored data model (see {doc}`provenance`), M3Resp stores the quality flags in the results of a workflow run as `QualityAnnotation` records. Each flag becomes one record. The record keeps the time window of the flag in seconds, and its breaths in `QualityAnnotation.breath_ids`: a list with one id for a flag about one breath, or the full list for a flag about several breaths.

Continue with {doc}`synchronization`.
