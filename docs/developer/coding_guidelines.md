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
