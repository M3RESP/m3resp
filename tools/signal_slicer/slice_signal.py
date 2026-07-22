"""Read a signal ``.txt`` file, slice it, and save the slice.

Two on-disk formats are supported and auto-detected:

* **plain** — one float sample per line, no header (e.g. ``*_emg.txt``). Slicing
  keeps a contiguous run of samples.
* **biopac** — a BIOPAC AcqKnowledge text export: a metadata header
  (``N msec/sample``, ``M channels``, per-channel label/unit lines), a
  per-channel sample-count row, then tab-separated multi-channel data. Slicing
  keeps a contiguous run of rows across every channel and rewrites the header so
  the output is still a valid export.

Use it as a library::

    from slice_signal import read_file

    sig = read_file("Paw_EMG_ajM3Resp_test.txt")
    piece = sig.slice(0, 60, unit="seconds")   # first 60 seconds
    piece.save("first60s.txt")

or from the command line::

    python slice_signal.py input.txt out.txt --start 0 --end 60 --unit seconds
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from itertools import islice
from pathlib import Path

import numpy as np


def _is_float(token: str) -> bool:
    try:
        float(token)
    except ValueError:
        return False
    return True


@dataclass
class SignalFile:
    """A loaded signal plus enough metadata to slice and re-save it.

    Attributes:
        data: ``(n_samples,)`` for plain files, ``(n_samples, n_channels)`` for
            biopac files. Ragged channels are padded with ``NaN``.
        fs: Sampling rate in Hz, or ``None`` if the format does not declare one.
        fmt: ``"plain"`` or ``"biopac"``.
        channels: Channel labels (``["signal"]`` for plain files).
        header_lines: Raw header lines to reproduce verbatim (biopac only).
        has_counts_row: Whether a per-channel sample-count row followed the
            column header (biopac only).
    """

    data: np.ndarray
    fs: float | None
    fmt: str
    channels: list[str]
    header_lines: list[str]
    has_counts_row: bool

    @property
    def n_samples(self) -> int:
        return int(self.data.shape[0])

    @property
    def n_channels(self) -> int:
        return 1 if self.data.ndim == 1 else int(self.data.shape[1])

    def channel(self, index: int = 0) -> np.ndarray:
        """Return one channel as a 1-D array (for previewing)."""
        return self.data if self.data.ndim == 1 else self.data[:, index]

    def slice(
        self,
        start: float | None = None,
        end: float | None = None,
        *,
        unit: str = "samples",
        fs: float | None = None,
    ) -> "SignalFile":
        """Return a new :class:`SignalFile` holding a contiguous window.

        Args:
            start: Window start (inclusive); ``None`` means the beginning.
            end: Window end (exclusive); ``None`` means the end of the signal.
            unit: ``"samples"`` for row indices, ``"seconds"`` for times.
            fs: Sampling rate override, used when ``unit == "seconds"``. Falls
                back to the file's own ``fs``.
        """
        s0, s1 = _resolve_bounds(start, end, unit, fs if fs is not None else self.fs)
        piece = self.data[s0:s1]
        if piece.shape[0] == 0:
            raise ValueError(
                "Selected slice is empty; check start/end (and fs for seconds)."
            )
        return SignalFile(
            data=piece.copy(),
            fs=self.fs,
            fmt=self.fmt,
            channels=list(self.channels),
            header_lines=list(self.header_lines),
            has_counts_row=self.has_counts_row,
        )

    def save(self, path: str | Path, *, fmt: str = "%.6g") -> Path:
        """Write the signal to ``path`` in its original on-disk format."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if self.fmt == "plain":
            np.savetxt(path, self.data, fmt=fmt)
        else:
            _write_biopac(self, path, fmt=fmt)
        return path


def _resolve_bounds(
    start: float | None, end: float | None, unit: str, fs: float | None
) -> tuple[int | None, int | None]:
    if unit not in {"samples", "seconds"}:
        raise ValueError(f"unit must be 'samples' or 'seconds', got {unit!r}")
    if unit == "seconds":
        if fs is None:
            raise ValueError("fs (sampling rate) is required when unit='seconds'")
        s0 = None if start is None else int(round(start * fs))
        s1 = None if end is None else int(round(end * fs))
    else:
        s0 = None if start is None else int(start)
        s1 = None if end is None else int(end)
    return s0, s1


def read_file(path: str | Path) -> SignalFile:
    """Load ``path``, auto-detecting the plain or biopac format."""
    path = Path(path)
    with open(path) as fh:
        first_line = fh.readline()
    tokens = first_line.split()
    if tokens and all(_is_float(t) for t in tokens):
        return _read_plain(path)
    return _read_biopac(path)


def _read_plain(path: Path) -> SignalFile:
    data = np.loadtxt(path, dtype=float)
    channels = (
        ["signal"] if data.ndim == 1 else [f"CH{i + 1}" for i in range(data.shape[1])]
    )
    return SignalFile(
        data=data,
        fs=None,
        fmt="plain",
        channels=channels,
        header_lines=[],
        has_counts_row=False,
    )


def _read_biopac(path: Path) -> SignalFile:
    # The header is short; read just enough lines to parse it before handing the
    # bulk data off to numpy.
    with open(path) as fh:
        head = [line.rstrip("\n") for line in islice(fh, 64)]

    msec_per_sample = float(head[1].split()[0])
    fs = 1000.0 / msec_per_sample if msec_per_sample else None
    n_channels = int(head[2].split()[0])

    channels = [head[3 + 2 * i] for i in range(n_channels)]
    col_header_idx = 3 + 2 * n_channels  # the "CH1  CH2  ..." row
    header_lines = head[: col_header_idx + 1]

    # An optional per-channel sample-count row (all integers) may follow.
    next_line = head[col_header_idx + 1]
    next_tokens = next_line.split()
    has_counts_row = bool(next_tokens) and all(
        _is_float(t) and float(t).is_integer() for t in next_tokens
    )
    data_start = col_header_idx + 1 + (1 if has_counts_row else 0)

    # Load the tab-separated table; a trailing tab yields an extra empty column.
    raw = np.genfromtxt(
        path, delimiter="\t", skip_header=data_start, filling_values=np.nan
    )
    if raw.ndim == 1:
        raw = raw.reshape(-1, 1)
    data = raw[:, :n_channels]

    return SignalFile(
        data=data,
        fs=fs,
        fmt="biopac",
        channels=channels,
        header_lines=header_lines,
        has_counts_row=has_counts_row,
    )


def _write_biopac(sig: SignalFile, path: Path, *, fmt: str) -> None:
    lines = list(sig.header_lines)
    if sig.has_counts_row:
        counts = np.sum(~np.isnan(sig.data), axis=0)
        lines.append("\t".join(str(int(c)) for c in counts) + "\t")

    def _fmt_row(row: np.ndarray) -> str:
        cells = ["" if np.isnan(v) else (fmt % v) for v in row]
        return "\t".join(cells) + "\t"

    body = "\n".join(_fmt_row(row) for row in sig.data)
    path.write_text("\n".join(lines) + "\n" + body + "\n")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("input", type=Path, help="Source .txt signal file")
    parser.add_argument("output", type=Path, help="Destination .txt file for the slice")
    parser.add_argument(
        "--start", type=float, default=None, help="Slice start (inclusive)"
    )
    parser.add_argument("--end", type=float, default=None, help="Slice end (exclusive)")
    parser.add_argument(
        "--unit",
        choices=("samples", "seconds"),
        default="samples",
        help="Interpret start/end as sample indices or seconds (default: samples)",
    )
    parser.add_argument(
        "--fs",
        type=float,
        default=None,
        help="Sampling rate in Hz (needed for seconds if the file has none)",
    )
    parser.add_argument(
        "--fmt", default="%.6g", help="numpy format string per value (default: %%.6g)"
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    sig = read_file(args.input)
    piece = sig.slice(args.start, args.end, unit=args.unit, fs=args.fs)
    out = piece.save(args.output, fmt=args.fmt)
    fs = args.fs or sig.fs
    duration = f", {piece.n_samples / fs:.3f}s" if fs else ""
    channels = "" if piece.n_channels == 1 else f" × {piece.n_channels} channels"
    print(
        f"Read {sig.n_samples} samples ({sig.fmt} format) from {args.input}\n"
        f"Wrote {piece.n_samples} samples{duration}{channels} to {out}"
    )


if __name__ == "__main__":
    main()
