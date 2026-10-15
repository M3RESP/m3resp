# Contributing

M3Resp connects EIT, EMG and ventilator processing in one workflow. This file shows where a change belongs, how to set up the code, and how to work on the documentation.

Read and follow our [Code of Conduct](CODE_OF_CONDUCT.md).

## Where a change belongs

| Change | Place |
|---|---|
| How EIT signals are processed | `eitprocessing` |
| How EMG signals are processed | `ReSurfEMG` (`resurfemg`) |
| Synchronization, export, the session, the workflow engine, and anything that combines modalities | M3Resp |

If you do not know where a change belongs, open an issue and ask.

## Kinds of contributions

- Questions: open an issue with the label "Question".
- Bug reports: open an issue with the "Bug report" form. Give the versions of M3Resp, Python, `eitprocessing` and `resurfemg`, your operating system, and a short example. Use synthetic data in the example, never patient recordings.
- Code changes: open a pull request. For a change larger than a small fix, open an issue first, so that we can agree on the approach.
- Documentation: changes to the documentation are welcome. See the documentation section below.

## Set up the code

The recordings in `tests/data` are stored with [Git LFS](https://git-lfs.com). Install `git-lfs` and run `git lfs install` once, before you clone the repository. If you cloned without it, run `git lfs pull`. Without Git LFS, the files are small text files, and the tests that read them fail.

Install M3Resp with the developer tools, and install the pre-commit hooks once:

```bash
pip install -e ".[all,dev]"
pre-commit install
```

The hooks run the Ruff checks and the Ruff formatter at each commit. To run them on all files:

```bash
pre-commit run --all-files
```

## Check a change

Before you open a pull request, run the tests and the type check:

```bash
pytest
mypy src
```

Write docstrings and comments as `AGENTS.md` describes. See the [testing page](docs/developer/testing.md) for the groups of tests.

## Documentation

Install the documentation tools and build the website:

```bash
python -m pip install -e ".[all,docs]"
python -m sphinx -W --keep-going -b html docs docs/_build/html
```

With uv:

```bash
uv sync --extra all --extra docs
uv run python -m sphinx -W --keep-going -b html docs docs/_build/html
```

Open `docs/_build/html/index.html` to see the result. The documentation workflow runs the same strict build for each pull request. Fix all warnings, broken links and wrong references before the merge.

Each page has one of four types. Put a new page in the folder of its type, and add it to the `toctree` of that folder:

| Type | Folder | Purpose |
|---|---|---|
| Tutorial | `docs/tutorials_src/` | Teaches a new user one complete analysis, step by step |
| How-to guide | `docs/how-to/` | Shows how to do one task, for a user who knows the basics |
| Concept | `docs/concepts/` | Explains what a thing is and why it works that way |
| Reference | `docs/workflows.md`, `docs/api/` | Describes each step, function and class |

Tutorials are Python scripts in Sphinx-Gallery format. The documentation build runs each script and puts the printed output and the figures on the page, in `docs/tutorials/`. The build sorts the scripts by file name, so start the name with a number (`10_`, `20_`, ...). When you add a tutorial:

1. Start with a docstring that has the title and says what the reader will get.
2. Use the synthetic recordings in `tests/data`. Read the folder from `M3RESP_EXAMPLE_DATA`, as the other tutorials do. Write output files only to a temporary folder.
3. Show the output of each step. End with a "Recap" and a "Next" section.
4. Add each plotting package that the tutorial needs to the `docs` extra.
5. Run the strict documentation build. Check the page and the `.py` and notebook downloads.

The API page of each function and class lists the tutorials that use it.

Read the Docs builds the default branch as `latest`. In the Read the Docs project settings, you can make a release tag the `stable` version and give each release its own numbered version. An administrator of the organization must import the repository and turn on the previews of pull requests before the website is available.
