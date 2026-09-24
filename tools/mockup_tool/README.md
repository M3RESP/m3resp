# Stage 3 GUI mockup

A static HTML mockup of the Stage 3 desktop GUI — Prepare / Design / Check /
Results — for demoing the intended workflow before the real GUI is built. No
backend and no server: it opens straight from disk over `file://`, and all
state is in-memory only (a page reload resets it).

## Open it

```bash
open tools/mockup_tool/stage3_gui_mockup.html        # macOS
xdg-open tools/mockup_tool/stage3_gui_mockup.html    # Linux
```

Or just double-click it — no server required.

`stage3_gui_mockup.html` is a thin shell that links the real content from
`src/`, so the mockup travels as a **folder**, not a single file. To hand it to
someone, send `tools/mockup_tool/` (or a zip of it) — the HTML file on its own
will render blank.

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

**`stage3_gui_mockup.html` is generated — don't hand-edit it.** Edits made
there are silently lost the next time anyone runs the build. It is a thin shell
of `<link>`/`<script src>` tags anyway; all the real content lives under
`src/`, split into files under ~300 lines each:

```
src/
  css/    *.css        — one file per section of the original stylesheet
  html/   *.html       — one file per tab (shell, prepare, design, check,
                         results)
  js/     data/        — the two large embedded blobs (EIT frame/mask data,
                         Results-tab figure PNGs as base64) as plain .js
                         declaring `EIT` / `REVIEW`; inert data, not code
          core/, design/, prepare/, results/
                        — one file per section of the original script,
                         grouped by the tab/feature they belong to
```

Because CSS and JS are linked rather than inlined, **editing them needs no
build step** — change a file under `src/css/` or `src/js/` and reload the
browser. Re-run the build only after editing `src/html/**`, or after
adding/removing/reordering a source file:

```bash
node build.js
```

`build.js` has no dependencies (Node built-ins only). It emits the `<link>` and
`<script src>` tags from one ordered file list, and inlines `src/html/**` (there
is no way to link an HTML fragment over `file://`). That ordering **is** the
module system here: there are no imports, just top-level declarations that
later files read (`EIT`, `REVIEW`, `ZOOM`, `savedSequences`), so a new source
file must be inserted where declaration-before-use still holds, not just
alphabetically.

Nothing needs a server: the page uses classic scripts and never calls
`fetch()`, so it works over `file://`. That is also why the data blobs are
`.js` declaring a global rather than `.json` loaded at runtime — fetching JSON
from `file://` is CORS-blocked. It is likewise why there are no ES modules:
`type="module"` is CORS-blocked over `file://` too.

The generated `stage3_gui_mockup.html` stays committed, so viewing the mockup
never requires a build step.

### Verifying a change

The mockup has no test suite; the practical check is that the page still
renders the same DOM it did before, with no console errors:

```bash
node build.js
google-chrome --headless=new --disable-gpu --dump-dom \
  "file://$PWD/stage3_gui_mockup.html" > /tmp/after.html
```

Diff that against the same dump taken before your change. Strip `<script>`,
`<style>`, `<link>` and comment nodes from both sides first, so the comparison
is about rendered content rather than how it got loaded. An unchanged post-JS
DOM is the signal that a refactor was behaviour-preserving.
