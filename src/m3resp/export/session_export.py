"""Session export functions."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from m3resp.export.tables import events_to_rows, parameters_to_rows


def export_session_summary(session: Any, output_dir: str | Path) -> Path:
    """Export a minimal CSV/JSON summary for an M3Resp session."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    summary = {
        "metadata": _jsonable(session.metadata),
        "quality": _jsonable(session.quality),
        "parameters": _jsonable(session.parameters),
        "provenance": _jsonable(session.provenance),
    }

    (output_path / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )

    for name, events in session.events.items():
        if events:
            _write_csv(output_path / f"{name}.csv", events_to_rows(events))

    if session.parameters:
        _write_csv(output_path / "parameters.csv", parameters_to_rows(session.parameters))

    return output_path


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return

    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    try:
        json.dumps(value)
    except TypeError:
        return repr(value)
    return value
