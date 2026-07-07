"""Serialize a ``DataModelStore`` to files, per the doc's storage mapping
(Sec 6.5): metadata tables become plain files here; the doc's file-backed
entities (waveforms, frame series) are already written by
``export/session_export.py`` and only referenced by ``DataFile`` rows.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel

from m3resp.datamodel.store import DataModelStore
from m3resp.workflows.utils import write_json

#: Table name -> attribute on DataModelStore, in the order the doc lists them.
_TABLES = (
    "cases",
    "sessions",
    "devices",
    "signal_streams",
    "eit_configurations",
    "emg_channels",
    "ventilator_settings",
    "breaths",
    "clinical_events",
    "data_files",
    "processing_runs",
    "derived_features",
    "quality_annotations",
)


def export_store(store: DataModelStore, out_dir: str | Path) -> dict[str, Path]:
    """Write one JSON file per table to ``out_dir``. Returns table -> path."""

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    written: dict[str, Path] = {}
    for table_name in _TABLES:
        table: dict[str, BaseModel] = getattr(store, table_name)
        rows = [entity.model_dump(mode="json") for entity in table.values()]
        file_path = out_path / f"{table_name}.json"
        write_json(file_path, {"rows": rows})
        written[table_name] = file_path
    return written
