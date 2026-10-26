# How to add a step of your own

You can add a calculation of your own as a workflow {term}`step`. Then you can use it in a workflow file as the built-in steps.

## Write the step

Write a Python function and register it with `@register_step`. Give the name of the step, the results that it reads (`reads`) and the names of its results (`writes`):

```python
from m3resp.workflows import register_step


@register_step(
    "my.threshold",
    reads={"signal": "global_impedance"},  # argument name: name of the result to read
    writes=("above",),                     # names of the results that it returns
)
def my_threshold(signal, *, threshold: float = 0.5) -> dict:
    return {"above": signal.values > threshold}
```

- The function gets the results as keyword arguments, and returns a dictionary with one entry per name in `writes`.
- A step with no results returns `{}` or `None`.
- Keyword arguments with a default value (here `threshold`) are the {term}`settings <setting>` of the step.
- Import `eitprocessing` or `resurfemg` inside the function, so M3Resp works also without these packages.

## Use the step

Import the Python file with the step, then run the workflow from Python:

```python
from m3resp.workflows import run_workflow

import my_steps  # the file with my_threshold

result = run_workflow("my-analysis.workflow.yaml")
print(result.value("above"))
```

In the workflow file:

```yaml
  - uses: my.threshold
    with: { threshold: 100.0 }
```

## Write files to the results folder

To write files, read the results folder of the run from `_resolved_output_dir`. It is `None` if the workflow file has no `outputs.dir`. With `outputs.timestamped: true`, it is the subfolder of this run, so all files of one run go to one folder:

```python
@register_step(
    "export.my_table",
    reads={"session": "session", "output_dir": "_resolved_output_dir"},
    writes=("result_path",),
)
def my_table(session, output_dir, **kwargs) -> dict:
    if output_dir is None:
        raise ValueError("export.my_table needs 'outputs.dir' in the workflow file.")
    ...
```

## Record each call in the provenance

To add a record to `session.provenance` at each call, register the step with `record=True`. The step must read `session`. It returns its own details, for example its settings, under the result name `"_provenance"`:

```python
@register_step(
    "my.smooth",
    reads={"session": "session", "signal": "global_impedance"},
    writes=("smoothed",),
    modality="eit",
    record=True,
)
def my_smooth(session, signal, *, window_seconds: float = 1.0) -> dict:
    smoothed = ...
    return {
        "smoothed": smoothed,
        "_provenance": {"parameters": {"window_seconds": window_seconds}},
    }
```

- After each call, M3Resp adds one record with the name and the modality of the step, its `reads` and `writes`, and the details from `"_provenance"`.
- The record also has `upstream_version`: the installed version of the first package in `optional_packages`, or `None` for a step without one.
- M3Resp removes `"_provenance"` from the results.
- This works in the same way when you call the function directly and when a workflow runs it.
