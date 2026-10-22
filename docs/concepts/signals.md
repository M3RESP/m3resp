# Signals

A {term}`signal` is one time series with a {term}`sampling rate` and a unit, for example the global impedance of an EIT recording or the envelope of an EMG channel. M3Resp stores each signal as a `Signal` object in `session.signals`.

## Look at a signal

```python
signal = session.signals.for_modality("eit")[0]
print(signal.name)               # Global impedance (raw)
print(signal.sample_frequency)   # 20.0
print(signal.unit)               # a.u.
print(signal.values.shape)       # (1200,)
print(signal.time[:3])           # [0.   0.05 0.1 ]
```

## What a signal holds

| Field | Contents |
|---|---|
| `values` | The measured values, as a NumPy array |
| `time` | The time of each sample on the clock of its recording, in seconds |
| `sample_frequency` | The sampling rate, in Hz |
| `unit` | The unit of the values, for example `"µV"` |
| `name` | The name of the signal |
| `modality` | The device that recorded it: `"eit"`, `"emg"`, `"ventilator"`, or another name |
| `category` | The physical quantity, for example `"impedance"`, `"airway_pressure"`, `"airflow"` or `"volume"` |
| `channel` | The channel of the device, for example an EMG channel name |
| `source` | The package or loader that made the signal, for example `"resurfemg"` |
| `processing_state` | How far the processing is: `"raw"`, `"intermediate"`, `"processed"` or `"derived"` |
| `derived_from` | For a derived signal: the processing state of the signals that it was calculated from |
| `method` | The calculation that made the signal |
| `metadata` | Other details, as a dictionary |

The first axis of `values` is time, so its length is equal to the length of `time`. `values` can have more axes. For example, the EIT pixel data has one 32 × 32 image per time point: time × rows × columns.

`Signal` adds the fields from `modality` to `method` to the basic time series, `TimeSeries`. A `TimeSeries` has only `values`, `time`, `sample_frequency`, `unit`, `name` and `metadata`.

## Units

M3Resp writes each unit in one form, so units from different devices can be compared. For example, `"uV"` becomes `"µV"`:

```python
import numpy as np

from m3resp import Signal

signal = Signal(values=np.zeros(3), time=np.arange(3) / 10, sample_frequency=10.0, unit="uV")
print(signal.unit)  # µV
```

To add a spelling of your own, use `register_unit_alias` in `m3resp.data.units`. To read a list of spellings from a YAML or JSON file, use `load_unit_aliases`.

## Modality and category

The `modality` is the device that recorded the signal. The `category` is the physical quantity. These are two separate fields, because one device records several quantities, and one quantity can come from several devices:

- A ventilator records airway pressure, flow and volume.
- Airway pressure can come from a ventilator, from a separate pressure monitor or from an EIT device.

Airway pressure from an EIT device has `modality="eit"` and `category="airway_pressure"`. You can search on each field:

```python
session.signals.for_modality("ventilator")       # all ventilator signals
session.signals.for_category("airway_pressure")  # airway pressure from all devices
```

`for_category` is also available on `session.parameter_results` and `session.quality`.

Both fields accept any name. M3Resp changes a known other name of a category to its main name: `category="paw"` is stored as `"airway_pressure"`, and `for_category("paw")` also works. M3Resp keeps an unknown name as it is. To add a name of your own, use `register_category_alias` or `load_category_aliases` in `m3resp.data.categories`.

## Processing state

| `processing_state` | Meaning |
|---|---|
| `"raw"` | The signal as the device or the loader gave it |
| `"intermediate"` | After one or more processing steps, for example a filter or a slice to a {term}`time window`. More steps follow. |
| `"processed"` | The final signal of a channel, ready for the calculation of parameters |
| `"derived"` | Calculated from other signals, for example the difference of two signals. `derived_from` gives the state of those signals. |

Two signals can have the same channel and the same processing state, for example two filtered signals that use different filters. The field `method` then tells them apart. M3Resp writes `method` as `"<package>.<function>"`, for example `"resurfemg.moving_baseline"` or `"eitprocessing.RateDetection"`.

## Where signals come from

`preprocess_eit`, `preprocess_emg` and `preprocess_ventilator` add signals to `session.signals`. Many workflow steps also add signals, for example `emg.ecg_gating` adds the EMG signal after ECG removal.

Continue with {doc}`events-and-breaths`.
