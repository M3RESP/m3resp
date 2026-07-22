# Signal slicer

Trim signal `.txt` files down to a shorter window and save the slice as a new
`.txt`. Two formats are auto-detected:

- **plain** — one float sample per line, no header (e.g. `*_emg.txt`).
- **biopac** — a BIOPAC AcqKnowledge export: a metadata header
  (`N msec/sample`, `M channels`, per-channel label/unit lines), a per-channel
  sample-count row, then tab-separated multi-channel data. The header is
  preserved and the counts row is recomputed, so the sliced file stays a valid
  export. Every channel is sliced together (ragged trailing channels are kept).

## Files

- `slice_signal.py` — core library + CLI. `read_file(path)` returns a
  `SignalFile` with `.slice(start, end, unit=...)` and `.save(path)`.
- `slicer.py` — marimo app on top of the library: pick a file, drag a window
  (by seconds or sample index), choose a preview channel, and save.

## CLI

```bash
# Keep the first 60 seconds (biopac files carry their own sampling rate)
python slice_signal.py input.txt out_first60s.txt --start 0 --end 60 --unit seconds

# Plain files have no rate on disk, so pass --fs when slicing by seconds
python slice_signal.py input_emg.txt out.txt --start 0 --end 60 --unit seconds --fs 2000

# Or slice by sample/row index (works for either format, no --fs needed)
python slice_signal.py input.txt out.txt --start 0 --end 10000
```

`--start` is inclusive, `--end` is exclusive; omit either to run to the edge.

## Marimo app

```bash
marimo edit tools/signal_slicer/slicer.py   # interactive
marimo run  tools/signal_slicer/slicer.py   # read-only app
```

The sampling-rate field is only used for plain files; biopac files use the rate
declared in their header (`0.5 msec/sample` → 2000 Hz). The preview is
downsampled for responsiveness; the saved slice is always full resolution.
