# How to check a workflow file

Check a {term}`workflow` file before a long run. The checks run no step and read no data.

## Check the file

```bash
m3resp validate my-analysis.workflow.yaml
```

This checks that the file is correct: each step exists, each input of a step comes from an earlier step, and each {term}`setting` has the correct type. M3Resp lists all problems in one report.

## Check that the files and packages are there

```bash
m3resp validate --readiness my-analysis.workflow.yaml
```

This also checks that the input files exist and that the packages of each step (`eitprocessing`, `resurfemg`) are installed. A missing file is an error. A missing package is a warning.

## See the steps that will run

```bash
m3resp run --dry-run my-analysis.workflow.yaml
```

This prints the steps in order, with their inputs, settings and results, and runs nothing.

## Check from Python

```python
from m3resp import load_spec
from m3resp.workflows import validate_workflow

spec = load_spec("my-analysis.workflow.yaml")
report = validate_workflow(spec, readiness=True)
print(report.is_valid)    # True if the file has no errors
print(report.as_dict())   # all problems
```

`is_valid` shows only the problems of the file itself. Look at `report.readiness` for the files and packages.

## Use the exit code in a script

| Exit code | Meaning |
|---|---|
| 0 | No problems |
| 2 | The file has an error, or M3Resp cannot read it |
| 3 | An input file is missing (with `--readiness`) |

See {ref}`workflow-command-line` for all exit codes.
