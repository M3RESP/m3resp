# How to read a ventilator file in another format

This guide reads a ventilator file that M3Resp has no reader for, for example a CSV export of the ventilator. You write one function that reads the file and register it for the file extension. Then `session.load_ventilator(...)` uses this function for every file with that extension.

M3Resp reads the ventilator waveforms of Draeger and Timpel EIT files (`.bin`) and of the multi-channel files that also hold the EMG. For these files, you need no reader of your own.

## Write the reader

The reader gets the path of the file and returns a dictionary with two keys:

- `"array"`: the samples, with one row per channel (channels, samples).
- `"metadata"`: a dictionary with `"fs"` (the samples per second, in Hz), `"labels"` (one name per row) and `"units"` (one unit per row).

This reader reads a CSV file with a header line and the columns time (s), airway pressure (cmH2O), flow (L/min) and volume (mL):

```python
import numpy as np


def read_ventilator_csv(path, **kwargs):
    data = np.loadtxt(path, delimiter=",", skiprows=1)
    time = data[:, 0]
    return {
        "array": data[:, 1:].T,  # one row per channel
        "metadata": {
            "fs": 1 / np.median(np.diff(time)),
            "labels": ["Paw", "Flow", "Volume"],
            "units": ["cmH2O", "L/min", "mL"],
        },
    }
```

M3Resp finds each channel by its label, in upper or lower case. `"Paw"`, `"airway pressure"` and `"pressure"` are the airway pressure, `"flow"` is the flow and `"volume"` is the volume. `"Pes"`, `"Pl"` and `"Pga"` are the esophageal, transpulmonary and gastric pressures. For a label of your device, add it with `register_channel_alias(...)`, for example `register_channel_alias("P_insp", "airway_pressure")`. When no label is known, rows 0, 1 and 2 are the airway pressure, the flow and the volume.

M3Resp keeps the units of the file and does not convert them.

## Register the reader

Register the reader once at the start of the script, before you load a file:

```python
from m3resp import M3Session
from m3resp.adapters.ventilator_adapter import register_ventilator_loader

register_ventilator_loader(".csv", read_ventilator_csv)

session = M3Session()
session.load_ventilator("patient01_ventilator.csv")
session.preprocess_ventilator()
breaths = session.detect_ventilator_breaths()
```

The registration holds until Python stops, for every session. `unregister_ventilator_loader(".csv")` removes it. With `source="eit"` or `source="emg"`, `session.load_ventilator(...)` uses the built-in readers also for a registered extension.

## Choose the clock of the channels

A file with its own clock has the default `source="ventilator"`. Give its start time with the key `"ventilator"` in {doc}`synchronize-recordings`.

Use `register_ventilator_loader(".ext", reader, source="eit")` or `source="emg"` only when the format holds the ventilator data of an EIT or EMG recording. The channels then have the clock of that recording. With `source="eit"`, the reader returns an `EITRecording` or an `EITFileData` with the ventilator waveforms in `signals`.
