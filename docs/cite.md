# Cite M3Resp

If you use M3Resp in your research, cite M3Resp and the packages that your analysis used.

## Cite the software

The file [CITATION.cff](https://github.com/M3RESP/m3resp/blob/main/CITATION.cff) has the citation data of M3Resp. On GitHub, the button "Cite this repository" gives the same data as APA and BibTeX text:

```bibtex
@software{m3resp,
  title   = {M3Resp},
  author  = {{M3Resp contributors}},
  version = {0.2.1},
  year    = {2026},
  url     = {https://github.com/M3RESP/m3resp},
  license = {Apache-2.0}
}
```

Change `version` and `year` to the version that you used. `python -m pip show m3resp` shows the version.

## Cite the modality packages

| Your analysis uses | Also cite |
|---|---|
| EIT | [eitprocessing](https://github.com/EIT-ALIVE/eitprocessing) |
| EMG | [ReSurfEMG](https://github.com/resurfemg-org/ReSurfEMG) |

Each package gives its citation data in its own repository.

## Report the analysis

Give the settings of each step in the methods section or in the supplementary material. A workflow file has all steps and their settings, so you can publish the file itself. The run manifest (`run_manifest.json`) records the M3Resp version and the Python version of the run.

See {doc}`concepts/provenance` for the records that M3Resp keeps.
