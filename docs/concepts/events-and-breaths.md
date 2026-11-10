# Events, intervals and breaths

M3Resp has three objects for things that happen at a time, beside the continuous {term}`signals <signal>`. All modalities use the same three objects.

| Object | What it is | Example |
|---|---|---|
| `Event` | Something at a single instant: `time` | An R peak, a blood gas sample |
| `Interval` | Something that lasts for a period, from `start_time` to `end_time` | An occlusion manoeuvre, a noisy period, an intervention |
| `BreathEvent` | One {term}`breath`: an interval with an {term}`extremum` inside it | A breath found in EIT, EMG or ventilator data |

All times are in seconds.

## Make an event and an interval

```python
from m3resp import Event, Interval

sample = Event("arterial_blood_gas_draw", "ventilator", 125.0, label="before PEEP step 2")
occlusion = Interval("ventilator", 10.0, 12.5, name="occlusion")
print(occlusion.duration)  # 2.5
```

- `name` says what kind of event or interval it is. `label` is a note of your own that tells two events of the same kind apart. M3Resp does not use `label`.
- For an `Interval`, give `modality`, `start_time` and `end_time` in this order, and all other fields by name. An interval that ends before it starts raises `ValueError`.
- An interval with equal start and end times is allowed. It means that only one time point is known.

## Breaths

A `BreathEvent` has all fields of an `Interval`. Its `name` is always `"breath"`. It also has `extremum_time` (in seconds) and `extremum_index` (the sample number):

```python
from m3resp import BreathEvent, Interval

breath = BreathEvent("eit", 1.0, 2.0, extremum_time=1.5)
print(breath.duration)                # 1.0
print(isinstance(breath, Interval))   # True
```

The extremum is the highest or the lowest point of the breath in the signal. Its meaning depends on the signal:

| Signal | Extremum |
|---|---|
| EIT impedance | The highest impedance: the end of inspiration |
| EMG envelope | The highest muscle activity |
| Ventilator volume | The highest volume |
| Airway pressure of an occlusion | The lowest pressure |

EIT, EMG and ventilator breath detection all give `BreathEvent` objects. So you can compare the breaths of different modalities (see {doc}`synchronization`).

The default EMG breath detection finds only the highest point of the {term}`envelope`. So the start time, the end time and the extremum time of an EMG breath are equal, and `metadata["boundaries_measured"]` is `False`. The step `emg.onoffpeak_baseline_crossing` measures the start and the end of each EMG breath, and keeps them in its own result.

An occlusion (Pocc) is also a `BreathEvent`, with `metadata["event_type"] == "pocc"`: it is a breath effort against a closed airway, with the lowest pressure as its extremum.

## Sample numbers

The fields `sample_index` (for an event), `start_index`, `end_index` and `extremum_index` give positions in a signal. Each recording can have signals with different sampling rates, so `signal_name` and `sample_frequency` say which signal the positions belong to. When an event does not come from a signal, the three fields are `None`.

## The identifier of an event

Each event, interval and breath gets a random `id` when M3Resp makes it. The `id` stays the same when M3Resp shifts the event to the shared clock, and when it writes the event to a file and reads it back. When you run the breath detection two times on the same data, the breaths of the two runs have the same times but different `id`s. M3Resp ignores the `id` when it compares two events, so the breaths of both runs count as equal.

A result can point to one event with its `id`, for example with `ParameterResult.event_id`. The results per breath of M3Resp still point to a breath by its position in the list.

## Convert other objects

`coerce_interval` and `coerce_breath_event` make an M3Resp object from a dictionary, from a `(start_time, end_time)` pair, or from an object with `start_time` and `end_time`, for example an `eitprocessing` interval. Give `name=` and `modality=` if the input does not have them.

```python
from m3resp import coerce_interval

interval = coerce_interval((10.0, 12.5), name="occlusion", modality="ventilator")
```

- An input with the name `"breath"`, or with an extremum, becomes a `BreathEvent`.
- An input with an extremum and another name raises `ValueError`, because an `Interval` has no extremum.
- `coerce_breath_event` raises `ValueError` for an input with another name than `"breath"`. So an occlusion or a noisy period never becomes a breath by mistake.
- `coerce_breath_event` reads the extremum from `extremum_time` and `extremum_index`, then from the older names `peak_time` and `peak_index` (with a `UserWarning`), then from the `middle_time` of `eitprocessing`.

## Values per interval or event

Some results have one value per breath or per event, for example the TIV of each breath or the amplitude of each R peak. `IntervalData` and `EventData` keep each value next to its interval or event:

```python
from m3resp import IntervalData

tiv = IntervalData(
    name="tidal_impedance_variation",
    modality="eit",
    intervals=session.get_events("eit_breaths"),
    values=[1.5, 1.7, 1.6],
    unit="a.u.",
)
```

| Field | Contents |
|---|---|
| `name` | What the values are |
| `modality` | The device of the values. This can be another device than the device of the intervals: EIT values can belong to breaths from the ventilator data. |
| `intervals` (`IntervalData`) or `events` (`EventData`) | The intervals or events, in order |
| `values` | One value per interval or event, in the same order: a number, an array (for example a pixel map) or an M3Resp object. `None` if the intervals are the result. |
| `category`, `unit` | The physical quantity and the unit, as on a {doc}`parameter <parameters>` |
| `method`, `metadata` | The calculation and other details |

If the number of values is not equal to the number of intervals, M3Resp raises `ValueError`. A NaN value stays next to its breath. `to_dict()` gives the content as lists and dictionaries for a JSON file.

The EIT steps store their values per breath in `session.interval_data`:

| Step | Name | One value per breath |
|---|---|---|
| `eit.continuous_tiv` | `continuous_tivs` | TIV |
| `eit.eeli` | `continuous_eelis` | EELI |
| `eit.pixel_tiv` | `pixel_tivs` | A {doc}`pixel map <pixel-maps>` of TIV |
| `eit.pixel_eeli` | `pixel_eelis` | A {doc}`pixel map <pixel-maps>` of EELI |
| `eit.pixel_breaths` | `pixel_breaths` | The start, middle and end time of the breath in each pixel |

`preprocess_eit` also stores `continuous_tivs`, `continuous_eelis` and `pixel_tivs`, and `pixel_eelis` with `compute_pixel_eeli=True`. The breaths of these values have the times of the EIT recording. A breath with the same start and end time as a breath in `session.events["eit_breaths"]` is the same object.

### Breaths of another modality

Four steps can use breaths of any modality: `eit.continuous_tiv`, `eit.eeli`, and `eit.pixel_tiv` and `eit.pixel_eeli` with the timing `continuous`. Connect the breaths to the input `breaths`. It takes a list of `BreathEvent`s or an `IntervalData` of breaths, for example `breath_intervals` of `eit.detect_breaths`, `emg_breath_events` of `emg.detect_breaths` or `ventilator_breath_events` of `ventilator.normalize_breaths`:

```yaml
- uses: eit.eeli
  in: { signal: global_impedance, breaths: ventilator_breath_events }
```

Check the meaning of the breath times first. TIV is the impedance at the end of inspiration (`extremum_time`) minus the impedance at the start of the breath, and EELI is the impedance at the end of the breath. The values are a TIV and an EELI only when the breath times mark these moments in the impedance, as the EIT breaths and the breaths of the EIT device do. The default EMG breaths have the same start, extremum and end time: the highest muscle activity. So their TIV is 0, and their EELI is the impedance at the highest muscle activity. The ventilator breaths of `ventilator.normalize_breaths` are a window of 0.5 s around the volume peak, so their EELI is near the end of inspiration.

Each modality keeps its breath times on its own clock. The step moves the breaths to the clock of the EIT recording with the start times of the session (see {doc}`../how-to/synchronize-recordings`), and reads each time at the nearest sample. If the recordings were never synchronized, the step gives a warning. A breath that lies partly outside the EIT data, or in a gap of its time axis (for example between two joined files), gets NaN. The result keeps the given breath objects, with their own times, and `metadata["breath_modality"]` gives their modality. The EIT figures (`save_eit_figures`, `plot_eit_processing_summary`) show each value on the EIT clock.

The pixel breath search takes EIT breaths only: `eit.pixel_breaths`, and `eit.pixel_tiv` and `eit.pixel_eeli` with the timing `pixel` and no `pixel_breaths` input. Breaths of another modality give a `ValueError`, because the search windows of the pixel breaths lie between the ends of inspiration of the impedance breaths. See {doc}`pixel-maps`.

## Where the breaths are stored

The session stores each list of breaths or events in `session.events`, under a name:

| Name | Filled by |
|---|---|
| `"eit_breaths"` | `detect_eit_breaths` |
| `"emg_breaths"` | `detect_emg_breaths` |
| `"ecg_peaks"` | The `"emg"` preset and the step `emg.ecg_detect_peaks` |
| `"ventilator_breaths"` | `detect_ventilator_breaths`, `postprocess_emg` with ventilator data, or the step `ventilator.normalize_breaths` |
| `"pocc_breaths"` | The step `ventilator.pocc_intervals`: the occlusions |

To store or read a list yourself:

```python
session.add_events("interventions", [occlusion])
print(session.get_events("interventions"))
```

`ventilator.detect_breaths` finds breaths as the highest points of the volume. An occlusion moves no air, so it is only in `"pocc_breaths"`. `ventilator.detect_pressure_breaths` finds breaths as the lowest points of the airway pressure, so with that step an occlusion can be in both lists.

Continue with {doc}`parameters`.
