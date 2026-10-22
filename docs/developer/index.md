# Developer guide

These pages explain how the M3Resp code is organized and how to add to it. Read the [contribution guide](https://github.com/M3RESP/m3resp/blob/main/CONTRIBUTING.md) before you open a pull request.

## Set up the code

The recordings in `tests/data` are stored with [Git LFS](https://git-lfs.com), so install `git-lfs` before you clone the repository. Install M3Resp from the source folder with the EIT and EMG packages and the developer tools. Then run the tests:

```bash
python -m pip install -e ".[all,dev,docs]"
pytest
```

## Pages

| Page | Contents |
|---|---|
| {doc}`architecture` | The two kinds of data objects, the folders of the code, and the folder for each kind of new code |
| {doc}`adapters` | How M3Resp calls `eitprocessing` and `resurfemg`, and changes their results into M3Resp objects |
| {doc}`preset-contracts` | What a preset does, and how it differs from a workflow file |
| {doc}`testing` | The test folders, the comparison tests with the two packages, the synthetic test data, and how to run the tests |
| {doc}`coding_guidelines` | Rules for the arguments, return values and names of functions |
| {doc}`offset-estimation` | The tools that find the time offset between two recordings |
| {doc}`processing-defaults` | The default settings of the EIT, EMG and ventilator processing, side by side |
| {doc}`migration` | How to change code that calls `eitprocessing` or `resurfemg` into M3Resp code |
| {doc}`project-history` | The stages of the project and the planned changes |

```{toctree}
:hidden:
:maxdepth: 1

architecture
adapters
preset-contracts
testing
coding_guidelines
offset-estimation
processing-defaults
migration
project-history
```

Continue with {doc}`architecture`.
