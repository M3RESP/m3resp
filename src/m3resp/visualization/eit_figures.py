"""Save static EIT pipeline figures from a run's context.

These are the headless, file-writing counterparts of the interactive panels
in ``tools/visualization_tools/1_annemijn_pipeline_results.py``: the native
``eitprocessing`` rate-detection figure, the global-impedance/breaths/TIV/EELI
overview, and the per-pixel TIV map. Each figure is produced only when the
context carries the keys it needs, so a pipeline that ran only part of the EIT
chain still writes whatever it can rather than failing.

Wired into the declarative engine by ``outputs.figures: true`` (see
``m3resp.workflows.engine.spec_runner._apply_outputs``); usable directly with
a ``PipelineResult.context.values`` mapping too.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np


def save_eit_figures(
    context_values: Mapping[str, Any], output_dir: str | Path
) -> list[Path]:
    """Write the available EIT pipeline figures into ``output_dir/figures/``.

    Returns the list of files written (empty if the context carries none of
    the required keys). Matplotlib is imported lazily so visualization stays
    an optional dependency of the core package.
    """

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - depends on local extras
        raise ImportError(
            "Saving EIT figures requires matplotlib. Install the EIT optional "
            "dependencies, or install matplotlib directly."
        ) from exc

    figures_dir = Path(output_dir) / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    for builder in (
        _figure_rate_detection,
        _figure_global_impedance,
        _figure_pixel_tiv,
    ):
        path = builder(context_values, figures_dir, plt)
        if path is not None:
            written.append(path)
    return written


def _figure_rate_detection(ctx, figures_dir: Path, plt) -> Path | None:
    """Native ``RateDetection.plotting.plot(...)`` from the rate capture."""

    detector = ctx.get("rate_detector")
    captures = ctx.get("rate_captures")
    if detector is None or not captures:
        return None

    plt.close("all")
    fig = detector.plotting.plot(**captures)
    fig.set_size_inches(11, 4.5)
    path = figures_dir / "rate_detection.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def _figure_global_impedance(ctx, figures_dir: Path, plt) -> Path | None:
    """Global impedance with per-breath TIV segments and EELI reference band.

    Mirrors ``eitprocessing``'s own TIV/EELI example conventions: TIV is the
    "inspiratory" rise ``value_at_middle - value_at_start`` drawn as a vertical
    segment per breath, and EELIs sit on the curve's troughs at each breath's
    end-expiratory sample, with the session mean/median and a +/-1 SD band.
    """

    gi = ctx.get("global_impedance")
    if gi is None:
        return None
    time = np.asarray(getattr(gi, "time", []), dtype=float)
    values = np.asarray(getattr(gi, "values", []), dtype=float)
    if time.size == 0:
        return None
    t0 = time[0]
    t = time - t0

    fig, ax = plt.subplots(figsize=(11, 4.0))
    ax.plot(t, values, color="#185FA5", linewidth=1.0, label="global impedance")

    breaths = getattr(ctx.get("breath_intervals"), "values", None) or []
    if breaths:
        start = np.array([b.start_time - t0 for b in breaths], dtype=float)
        mid = np.array([b.middle_time - t0 for b in breaths], dtype=float)
        peak_idx = np.clip(np.searchsorted(t, mid), 0, t.size - 1)
        start_idx = np.clip(np.searchsorted(t, start), 0, t.size - 1)
        peak_y = values[peak_idx]
        base_y = values[start_idx]
        ax.plot(
            t[peak_idx],
            peak_y,
            linestyle="none",
            marker="v",
            color="#D1495B",
            markersize=6,
            label="detected breaths",
        )
        # Vertical TIV segment (base -> peak) per breath; NaN breaks between
        # breaths so a single call draws every segment.
        seg_x = np.repeat(t[peak_idx], 3)
        seg_x[2::3] = np.nan
        seg_y = np.empty(base_y.size * 3, dtype=float)
        seg_y[0::3] = base_y
        seg_y[1::3] = peak_y
        seg_y[2::3] = np.nan
        ax.plot(seg_x, seg_y, color="#2A9D8F", linewidth=2.0, label="TIV")

    eeli = ctx.get("eeli")
    if eeli is not None:
        eeli_t = np.asarray(getattr(eeli, "time", []), dtype=float) - t0
        eeli_v = np.asarray(getattr(eeli, "values", []), dtype=float)
        if eeli_v.size:
            ax.plot(
                eeli_t,
                eeli_v,
                linestyle="none",
                marker="o",
                color="black",
                markersize=4,
                label="EELIs",
            )
            mean_eeli = float(np.mean(eeli_v))
            median_eeli = float(np.median(eeli_v))
            std_eeli = float(np.std(eeli_v))
            ax.axhspan(
                mean_eeli - std_eeli,
                mean_eeli + std_eeli,
                color="blue",
                alpha=0.12,
                label="EELI +/-1 SD",
            )
            ax.axhline(mean_eeli, color="red", linewidth=1.2, label="EELI mean")
            ax.axhline(median_eeli, color="green", linewidth=1.2, label="EELI median")

    ax.set_xlabel("time (s)")
    ax.set_ylabel("impedance (a.u.)")
    ax.set_title("Global impedance, detected breaths, TIV & EELI")
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    fig.tight_layout()
    path = figures_dir / "global_impedance.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def _figure_pixel_tiv(ctx, figures_dir: Path, plt) -> Path | None:
    """Mean per-pixel TIV map via the native ``PixelMap.plotting.imshow``."""

    pixel_tiv = ctx.get("pixel_tiv")
    if pixel_tiv is None:
        return None
    maps = np.asarray(getattr(pixel_tiv, "values", pixel_tiv), dtype=float)
    if maps.ndim != 3 or maps.size == 0:
        return None
    mean_map = np.nanmean(maps, axis=0)

    try:
        from eitprocessing.plotting.pixelmap import PixelMap
    except ImportError:  # pragma: no cover - depends on local extras
        return None

    plt.close("all")
    fig, ax = plt.subplots(figsize=(5.0, 4.6))
    plt.sca(ax)
    PixelMap(mean_map, label="mean tidal impedance variation").plotting.imshow(
        colorbar=True
    )
    ax.set_title(f"Mean pixel TIV over {maps.shape[0]} breaths")
    fig.tight_layout()
    path = figures_dir / "pixel_tiv.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path
