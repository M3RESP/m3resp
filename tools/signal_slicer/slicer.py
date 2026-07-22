import marimo

__generated_with = "0.23.10"
app = marimo.App(width="full", app_title="Signal slicer")


@app.cell
def _imports():
    from pathlib import Path

    import marimo as mo
    import numpy as np
    import plotly.graph_objects as go

    from slice_signal import read_file

    return Path, go, mo, np, read_file


@app.cell
def _header(mo):
    mo.md(
        """
        # Signal slicer

        Load a signal `.txt` file, trim it to a chosen window, preview the result,
        and save the slice to a new `.txt`. Both formats are auto-detected:

        * **plain** — one float sample per line (e.g. `*_emg.txt`).
        * **biopac** — a multi-channel AcqKnowledge export with a metadata header;
          the header is preserved and every channel is sliced together.

        Select the window by **seconds** (using the sampling rate) or by **sample
        index**.
        """
    )
    return


@app.cell
def _ui_input(mo):
    input_path = mo.ui.text(
        value=("../../data/source/eit_emg_annemijn/Paw_EMG_ajM3Resp_test.txt"),
        label="Input .txt file",
        full_width=True,
    )
    fs_manual = mo.ui.number(
        value=2000.0,
        start=1.0,
        step=1.0,
        label="Sampling rate (Hz) — used only if the file declares none",
    )
    mo.vstack([input_path, fs_manual])
    return fs_manual, input_path


@app.cell
def _load(Path, fs_manual, input_path, mo, read_file):
    _p = Path(input_path.value).expanduser()
    mo.stop(
        not input_path.value or not _p.is_file(),
        mo.md(f"⚠️ Waiting for a valid input file. Not found: `{input_path.value}`"),
    )
    sig = read_file(_p)
    source = _p
    fs = sig.fs if sig.fs else fs_manual.value

    _dur = f"{sig.n_samples / fs:.1f}s" if fs else "unknown duration"
    _fs_note = (
        f"declared **{sig.fs:g} Hz**"
        if sig.fs
        else f"no rate in file — using **{fs_manual.value:g} Hz**"
    )
    mo.md(
        f"Loaded `{source.name}` — **{sig.fmt}** format, "
        f"**{sig.n_samples:,} samples** × {sig.n_channels} "
        f"channel(s) ({', '.join(sig.channels)}), {_dur}, {_fs_note}."
    )
    return fs, sig, source


@app.cell
def _ui_window(fs, mo, sig):
    unit = mo.ui.radio(
        options=["seconds", "samples"], value="seconds", label="Select window by"
    )
    preview_channel = mo.ui.dropdown(
        options={name: i for i, name in enumerate(sig.channels)},
        value=sig.channels[0],
        label="Preview channel",
    )
    _max_s = sig.n_samples / fs if fs else 0.0
    time_range = mo.ui.range_slider(
        start=0.0,
        stop=round(_max_s, 3),
        step=max(round(_max_s / 1000, 3), 0.001),
        value=[0.0, round(_max_s, 3)],
        label="Time window (s)",
        full_width=True,
    )
    sample_range = mo.ui.range_slider(
        start=0,
        stop=int(sig.n_samples),
        step=1,
        value=[0, int(sig.n_samples)],
        label="Sample window",
        full_width=True,
    )
    mo.vstack(
        [
            mo.hstack([unit, preview_channel], justify="start", gap=2),
            time_range,
            sample_range,
        ]
    )
    return preview_channel, sample_range, time_range, unit


@app.cell
def _compute_slice(fs, mo, sample_range, sig, time_range, unit):
    if unit.value == "seconds":
        _start, _end = time_range.value
        piece = sig.slice(_start, _end, unit="seconds", fs=fs)
    else:
        _start, _end = sample_range.value
        piece = sig.slice(_start, _end, unit="samples")

    _dur = piece.n_samples / fs if fs else 0.0
    mo.md(
        f"**Slice:** {piece.n_samples:,} samples ({_dur:.3f}s) — "
        f"{100 * piece.n_samples / sig.n_samples:.1f}% of the original."
    )
    return (piece,)


@app.cell
def _preview(fs, go, mo, np, piece, preview_channel):
    # Downsample purely for a responsive plot; the saved slice is full-resolution.
    _ch = piece.channel(preview_channel.value)
    _max_points = 8000
    _step = max(1, _ch.size // _max_points)
    _y = _ch[::_step]
    _t = np.arange(0, _ch.size, _step) / fs if fs else np.arange(_y.size)

    _fig = go.Figure(go.Scatter(x=_t, y=_y, mode="lines", line={"width": 1}))
    _fig.update_layout(
        title=f"Slice preview — {piece.channels[preview_channel.value]}",
        xaxis_title="Time (s)" if fs else "Sample",
        yaxis_title="Amplitude",
        height=360,
        margin={"l": 60, "r": 20, "t": 40, "b": 40},
    )
    mo.ui.plotly(_fig)
    return


@app.cell
def _ui_save(mo, source):
    default_out = f"{source.stem}_slice.txt"
    output_path = mo.ui.text(
        value=default_out, label="Output .txt file", full_width=True
    )
    save_button = mo.ui.run_button(label="💾 Save slice")
    mo.vstack([output_path, save_button])
    return output_path, save_button


@app.cell
def _save(Path, fs, mo, output_path, piece, save_button, source):
    mo.stop(
        not save_button.value,
        mo.md("_Adjust the window, then click **Save slice**._"),
    )

    _out = Path(output_path.value).expanduser()
    if not _out.is_absolute():
        _out = source.parent / _out
    piece.save(_out)

    _dur = piece.n_samples / fs if fs else 0.0
    mo.md(f"✅ Saved **{piece.n_samples:,} samples** ({_dur:.3f}s) to\n\n`{_out}`")
    return


if __name__ == "__main__":
    app.run()
