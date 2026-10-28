This product includes software from third-party open-source projects.
The full text of each project's license is available in its own repository.

## ReSurfEMG

```
ReSurfEMG
  Copyright 2022 The Netherlands eScience Center, University of Twente

  This product includes software developed at
  The Netherlands eScience Center (https://www.esciencecenter.nl/)
  For the 'development of a software tool for automated surface EMG' project.
```

Source: https://github.com/resurfemg-org/ReSurfEMG  
License: Apache License, Version 2.0

Portions of `src/m3resp/processing/peaks.py`, `src/m3resp/processing/windows.py`,
`src/m3resp/processing/metrics.py`, `src/m3resp/processing/intervals.py`,
`src/m3resp/processing/filters.py`, and `src/m3resp/processing/ventilator.py`
are derived from ReSurfEMG; see the
per-module "Provenance" notes in those files for the specific functions and
upstream revision.


## eitprocessing
```
eitprocessing
  This product includes eitprocessing, software developed by Netherlands eScience 
  Center, Erasmus MC.
```

Source: https://github.com/EIT-ALIVE/eitprocessing  
License: Apache License, Version 2.0

Portions of `src/m3resp/processing/filters.py`,
`src/m3resp/processing/windows.py`, `src/m3resp/processing/metrics.py`,
`src/m3resp/processing/peaks.py`, `src/m3resp/data/pixel_maps.py`,
`src/m3resp/eit/io/binreader.py`, `src/m3resp/eit/io/draeger.py`,
`src/m3resp/eit/io/sentec.py`, `src/m3resp/eit/io/timpel.py`,
`src/m3resp/eit/io/read_file.py`, `src/m3resp/eit/io/join.py`,
`src/m3resp/eit/processing/impedance.py`, `src/m3resp/eit/processing/slicing.py`,
`src/m3resp/eit/processing/rate_detection.py`, `src/m3resp/eit/processing/breath_detection.py`,
`src/m3resp/eit/processing/mdn_filter.py`,
`src/m3resp/visualization/rate_detection.py`,
`src/m3resp/processing/slicing.py`, and
`src/m3resp/adapters/eitprocessing_adapter/_slicing.py` are derived from eitprocessing; see the per-module "Provenance"
notes in those files for the specific functions and upstream revision.

## Test data

`tests/data/sentec_healthy_volunteer_1b.zri` and `tests/data/timpel_healthy_volunteer_1_60-120s.txt` come from "Test data for the eitprocessing software package", version 1.0, by Peter Somhorst (2025), https://doi.org/10.5281/zenodo.17423608.

License: Creative Commons Attribution 4.0 International (CC BY 4.0), https://creativecommons.org/licenses/by/4.0/

- `sentec_healthy_volunteer_1b.zri` is the original file, unchanged.
- `timpel_healthy_volunteer_1_60-120s.txt` holds lines 3001 to 6000 of `timpel_healthy_volunteer_1.txt`, which is 60 s to 120 s of the recording at 50 Hz. The lines are unchanged.
