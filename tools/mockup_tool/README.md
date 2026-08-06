# Stage 3 GUI mockup

A single self-contained HTML file that mocks up the Stage 3 desktop GUI —
Prepare / Design / Check / Results — for demoing the intended workflow before
the real GUI is built. No backend: `stage3_gui_mockup.html` is fully
self-contained and needs no server, and all state is in-memory only (a page
reload resets it).

## Open it

```bash
open tools/mockup_tool/stage3_gui_mockup.html        # macOS
xdg-open tools/mockup_tool/stage3_gui_mockup.html    # Linux
```

Or just double-click it — no server required.

## What's real vs. illustrative

The mockup is grounded in the actual m3resp step registry and example data
wherever possible, and says so wherever it isn't:

- **Real**: the default Design-tab workflow graph is `describe_steps()`
  output for the real step registry (ports, types, ids); the default
  15-node/45-step graph mirrors `examples/*/multimodal-full.pipeline.yaml`;
  the EIT dynamic/minute images and impedance curves are computed from the
  actual `.bin` recording under `data/source/synthetic/` via
  `eitprocessing.datahandling.loading.load_eit_data`; the Check/Results tabs
  reuse real output artifacts from a past `multimodal-full-summary` run
  (`run_manifest.json`, `parameter_results.csv`, `quality_flags.csv`, …).
- **Illustrative**: EMG/ventilator waveforms, quality-flag numbers, and
  anything else that would require a live backend are seeded synthetic data
  standing in for the real thing — called out inline (e.g. in data-preview
  pop-ups) wherever that's the case.

## Features covered

- **1 · Prepare** — per-modality load ("Load file…" opens a real file
  picker), raw synchronization offsets, a multi-window "Working window"
  slicer that saves named sequences into a persistent library independent of
  whatever dataset is currently loaded.
- **2 · Design** — a node-graph editor over the real step registry: drag
  nodes, drag-to-connect ports, add/remove nodes and edges, multi-select
  (Shift/Ctrl-click or drag a box) with bulk delete, zoom in/out and fit-to-
  view, click any "Available data" row to preview it in a pop-up, Clear
  board / Auto layout (reset to the default workflow), Validate / Compile /
  Run pipeline (log + problems panel).
- **3 · Check** — quality flags and run report, gated behind "Confirm &
  continue".
- **4 · Results** — parameters, saved values, and plots.

## Developing

`stage3_gui_mockup.html` is generated — don't hand-edit it. The real source
lives under `src/`, split into files under ~300 lines each:

```
src/
  data/   *.json      — the two large embedded blobs (EIT frame/mask data,
                         Results-tab figure PNGs as base64), pulled out of
                         the JS since they're inert data, not code
  css/    *.css        — one file per section of the original stylesheet
  html/   *.html       — one file per tab (shell, prepare, design, check,
                         results)
  js/     core/, design/, prepare/, results/
                        — one file per section of the original script,
                         grouped by the tab/feature they belong to
```

Edit the files under `src/`, then regenerate the distributable file:

```bash
node build.js
```

`build.js` has no dependencies (Node built-ins only) and works by
concatenating the `src/` files back together **in the same order they
appeared in the original single file** — this matters because a few pieces
of state are declared early precisely so later-defined code can read them
(e.g. `ZOOM`, `savedSequences`). If you add a new source file, insert it
into `build.js`'s ordered file list at the point that preserves that
ordering, not just alphabetically.

The two big data blobs are injected via a `/*__INJECT_DATA__:filename__*/`
marker comment in the JS source (see `src/js/prepare/eit-workspace.js` and
`src/js/results/plots.js`) — `build.js` replaces the marker with the raw
contents of the matching file in `src/data/`.

The generated `stage3_gui_mockup.html` stays committed to the repo and
remains a single, double-click-openable file with no build step required
to *view* it — the build step is only needed after editing the source.
