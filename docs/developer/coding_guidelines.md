# Coding guidelines

This file collects small design-pattern decisions that came up during review
and are meant to be reused rather than re-litigated per function. If you are
about to introduce a new one-off convention, check here first, and if what
you need isn't covered, add it here once it's settled instead of leaving the
precedent buried in a PR discussion.

## Optional secondary outputs: use `captures`, not a boolean return-shape toggle

Don't give a function a boolean flag that changes what type it returns (e.g.
`return_details: bool = False` making the function return either a value or
a tuple). This forces an `@overload` pair to type it correctly, is easy to
call wrong (`result = f(...)` silently works whether or not the flag was
set, so a caller who forgot to unpack a tuple gets a confusing downstream
error instead of an immediate one), and is generally considered poor
practice (see the discussion on
[PR #24](https://github.com/M3RESP/m3resp/pull/24), which surveyed
alternatives including always-returning-a-tuple, attaching properties to
the output object, and an OOP detector class).

Instead, accept an optional `captures: dict[str, Any] | None = None`
parameter and fill it in-place via the shared `capture_value` helper
(`m3resp.processing.filters.capture_value`):

```python
def detect_peaks(
    ...,
    captures: dict[str, Any] | None = None,
) -> np.ndarray:
    ...
    capture_value(captures, "properties", properties)
    return peak_indices
```

The function's return type never changes shape. A caller who doesn't need
the extra output just omits `captures`; one who does passes a dict and
reads it back afterwards. This is the convention already used throughout
`m3resp.processing` (`filters.py`, `peaks.py`); follow it for any new
function with optional diagnostic/intermediate outputs rather than
inventing a new toggle.

## Pick fields by producer semantics, not by inspecting the value

Some types offer more than one field to express what looks like the same
kind of information (for example `ParameterResult.breath_id` vs.
`breath_ids`, or `event_id` vs. `start_time`/`end_time`). When a type does
this, which field a given piece of code uses must be decided once, at
implementation time, based on what the *producing algorithm* actually
computed, never dynamically, based on inspecting the value itself (e.g.
"use `breath_id` if the list has exactly one element").

Concretely: an algorithm whose method is "aggregate a value over a set of
breaths" always populates `breath_ids`, even in a run where that set
happens to contain only one breath, because the claim being made
("this is an aggregate over a breath-set") doesn't change with how many
elements the set has. `breath_id` is reserved for algorithms that are
inherently single-breath by design. See the discussion on
[PR #24](https://github.com/M3RESP/m3resp/pull/24) for the full reasoning.

## Name the shared primitive by what it computes; name the domain quantity by its literature term

When a generic math primitive in `m3resp.processing` (shared across
modalities) underlies a domain-specific, literature-named quantity in one
modality's adapter, name each thing for its own audience rather than giving
the shared primitive the domain term. The primitive's name should describe
what it mathematically does; the domain-specific name stays wherever it's
actually user- or literature-facing (dict keys, registered step names,
docstrings of the modality-specific functions that consume it).

Concretely: `m3resp.processing.metrics.window_integral` computes "integrate
signal-minus-baseline over a window" - a primitive already reused by both
EMG and ventilator step code for different purposes. In EMG, that
computation produces the quantity respiratory-EMG literature calls
"time-product"/ETP (Electrical Time Product), and that name is preserved
everywhere it's domain-facing: the `"time_product"` dict key, the
registered `emg.time_product` step, and the quality-assessment functions
that take `time_products` as an argument. Naming the shared primitive
itself `time_product` would have been the actual mistake, since ventilator
code calls it for something that isn't ETP at all. See the discussion on
[PR #24](https://github.com/M3RESP/m3resp/pull/24) for the full reasoning.

## Two ways to specify one value: both default to `None`, raise if both are set

When a function offers two parameters that specify the same underlying
value in different units or forms (e.g. `gate_width_seconds` /
`gate_width_samples`, `min_peak_width_samples` / `min_peak_width_s`), give
both a default of `None` rather than giving one of them a concrete default
like `1`. Resolve the effective value only after checking both: if both are
`None`, fall back to the real default; if both are explicitly set, raise
`ValueError` rather than silently letting one win.

A concrete non-`None` default on one of the two breaks this: it makes it
impossible to tell whether the caller explicitly passed that value or just
left it at default, so you can neither warn nor error correctly when the
other parameter is also set - you'd get false positives whenever the
default happens to differ from what the caller passed for the other one.
`None`-by-default on both sides is what makes "was this explicitly set?"
answerable at all. See `ecg_gating`'s `gate_width_seconds`/
`gate_width_samples` handling and `detect_emg_breath_peaks`'s
`min_peak_width_samples`/`min_peak_width_s` handling for the pattern in
practice, and the discussion on
[PR #24](https://github.com/M3RESP/m3resp/pull/24) for the full reasoning.
