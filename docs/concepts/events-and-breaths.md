# `Event` and `BreathEvent`

`m3resp.core.events` defines two timestamped types shared across modalities.

## `Event`

A single-instant, timestamped event from one modality:

```python
@dataclass
class Event:
    name: str
    modality: str
    time: float
    sample_index: int | None = None
    label: str | None = None
    confidence: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
```

## `BreathEvent`

Unlike `Event`, which occurs at a single `time`, a `BreathEvent` spans an
interval (`start_time` to `end_time`) - one breath, not an instant. EIT,
EMG, and ventilator breath detectors all produce this same type rather than
each having their own `Breath` class.

```python
@dataclass
class BreathEvent:
    modality: str
    start_time: float
    end_time: float
    peak_time: float | None = None
    start_index: int | None = None
    peak_index: int | None = None
    end_index: int | None = None
    sample_frequency: float | None = None
    signal_name: str | None = None
    source: str | None = None
    confidence: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time
```

`start_time`/`end_time`/`peak_time` are always the authoritative,
real-world times - they don't need to be recomputed from an index. The
`*_index` fields are optional sample-index positions into the signal that
produced this breath; `sample_frequency` and `signal_name` identify which
time axis those indices are relative to, since different signals have
different start times, durations, and sampling rates.

## Where breath/event lists live

There is deliberately no `BreathCollection` type: breaths live in
`session.events`, a `dict[str, list[BreathEvent]]` populated by
`session.detect_eit_breaths()` (`"eit_breaths"`),
`session.detect_emg_breaths()` (`"emg_breaths"`), and any ventilator breaths
normalized during `postprocess_emg` (`"ventilator_breaths"`). Use
`session.add_events(name, events)`/`session.get_events(name)` for direct
access. This predates the typed-collection work and is depended on
throughout Stage 1 - introducing a second container would fork that API
rather than reconcile with it.

To match breaths across modalities into a single physiological event, see
[`LinkedBreath`](synchronization.md).
