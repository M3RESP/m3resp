# How to write a workflow file

A {term}`workflow` file is a YAML or JSON file with a list of {term}`steps <step>`. See {doc}`../workflows` for all keys of a workflow file.

## Start from an example

Copy an example from the `examples` folder of the M3Resp source code, and change it:

| Example | Use it for |
|---|---|
| `examples/multimodal_example/multimodal.workflow.yaml` | A short EIT, EMG and ventilator workflow |
| `examples/eit_full_preprocessing/eit-full.workflow.yaml` | All EIT steps |
| `examples/emg_full_preprocessing/emg-full.workflow.yaml` | All EMG and ventilator steps |
| `examples/multimodal_full/multimodal-full.workflow.yaml` | All steps of the three modalities |

## Give the input files

Put the file paths under `inputs`, and refer to them with `@` in the steps:

```yaml
inputs:
  eit_file: ../data/subject01.bin

steps:
  - uses: eit.load
    with: { file_path: "@eit_file", vendor: draeger }
```

M3Resp reads a relative path from the folder of the workflow file.

## Add a step

To list the steps and their settings:

```bash
m3resp steps
m3resp describe eit.detect_breaths
```

Add the step with `uses`, and its {term}`settings <setting>` with `with`. If the step must read another result than its default input, set it with `in`:

```yaml
  - uses: eit.detect_breaths
    in: { signal: global_impedance }
    with: { min_duration_s: 0.6666666667 }
```

## Run the same step twice

Each step writes its results under fixed names. To keep the results of both runs, rename the results of one run with `out`:

```yaml
  - uses: eit.global_impedance
    in: { signal: raw_eit }
    out: { global_impedance: unfiltered_global_impedance }

  - uses: eit.global_impedance
    in: { signal: filtered_eit }
```

## Choose the result files

Set the folder and the files in `outputs`:

```yaml
outputs:
  dir: ../../output/my-analysis
  mode: automatic
  timestamped: true
  structured_export: true
```

With `timestamped: true`, each run writes to its own subfolder with the date and time. See {ref}`workflow-outputs` for all settings.

## Use strict checks

Set `schema_version: 1` at the top of the file. M3Resp then refuses unknown keys and settings that are not real booleans, so a typing error stops the run. Give each step an `id`, so other tools can find the step also after you add or move steps:

```yaml
schema_version: 1

steps:
  - id: detect_breaths
    uses: eit.detect_breaths
```

Then check the file before you run it: {doc}`check-a-workflow`.
