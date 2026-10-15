# Getting started

## Install M3Resp

M3Resp works with Python 3.12, 3.13 and 3.14. In the M3Resp source folder, install the base package:

```bash
python -m pip install -e .
```

Then install the integration for each {term}`modality` that you use:

```bash
python -m pip install -e ".[eit]"       # EIT processing
python -m pip install -e ".[emg]"       # EMG processing
python -m pip install -e ".[all]"       # EIT and EMG processing
```

The base package reads and checks workflow files, synchronizes recordings, makes synthetic EIT data and exports results.

## Run a workflow

A {term}`workflow` file is a YAML or JSON file with a list of processing steps. Run the example workflow from the M3Resp source folder:

```bash
m3resp run examples/multimodal_example/multimodal.workflow.yaml
```

The example uses the synthetic recordings in `tests/data` and writes the results to `output/multimodal-summary`. M3Resp reads the file paths in a workflow file from the folder of that file.

To list all processing steps:

```bash
m3resp steps
```

## Use Python

Run the same workflow from Python:

```python
from m3resp import load_spec, run_spec

spec = load_spec("examples/multimodal_example/multimodal.workflow.yaml")
result = run_spec(spec)
print(result.status)               # succeeded
print(result.resolved_output_dir)  # the folder with the result files
```

To run each step yourself, use a {term}`session`:

```python
import os

from m3resp import M3Session

session = M3Session()
session.load_eit(os.path.join("tests", "data", "m3resp_multimodal_1_eit_draeger.bin"), vendor="draeger")
session.run_preset("eit")
print(len(session.events["eit_breaths"]))  # 13
```

Continue with the {doc}`tutorials <tutorials/index>`, or read how a [workflow file](workflows.md) is built.
