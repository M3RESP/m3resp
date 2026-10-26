# Get help

This page shows what to do when a workflow fails or when you have a question about M3Resp.

## Read the error message

The command line prints a short message and an exit code when a workflow fails. To see the full Python error, add `--debug`:

```bash
m3resp run my.workflow.yaml --debug
```

The {ref}`workflow-command-line` section gives the meaning of each exit code.

## Check the workflow file

`m3resp validate` checks the file before a run. With `--readiness`, it also checks that each input file exists:

```bash
m3resp validate my.workflow.yaml --readiness
```

See {doc}`how-to/check-a-workflow`.

## Ask a question

Open an issue on [GitHub](https://github.com/M3RESP/m3resp/issues) and add the label "Question". Give the command or the code that you used, and the full error message.

## Report a problem

Open an issue on [GitHub](https://github.com/M3RESP/m3resp/issues) with the "Bug report" form. The form asks for the versions of Python, M3Resp, `eitprocessing` and `resurfemg`. To show them:

```bash
python --version
python -m pip show m3resp eitprocessing resurfemg
```

:::{warning}
Do not attach patient recordings to an issue. An issue is public. To show the problem, use the synthetic recordings in `tests/data`, or make a synthetic recording with `m3resp.synthetic`.
:::

The run manifest of a workflow run (`run_manifest.json`) records the M3Resp version, the Python version and each step. Attach it to the issue if the problem occurs in a workflow run.

## Problems in the EIT or EMG processing

M3Resp uses `eitprocessing` for EIT and `resurfemg` for EMG. Report a problem in these calculations on the [M3Resp issue list](https://github.com/M3RESP/m3resp/issues) too.

Continue with {doc}`cite`.
