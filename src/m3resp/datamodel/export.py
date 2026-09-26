"""Serialize a ``DataModelStore`` to files, per the doc's storage mapping
(Sec 6.5): metadata tables become plain files here; the doc's file-backed
entities (waveforms, frame series) are already written by
``export/session_export.py`` and only referenced by ``DataFile`` rows.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel

from m3resp.core.exceptions import DataModelValidationError
from m3resp.datamodel.store import DataModelStore
from m3resp.datamodel.validation import validate_store
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


def export_store(
    store: DataModelStore,
    out_dir: str | Path,
    *,
    validate: bool = True,
    require_complete: bool = False,
) -> dict[str, Path]:
    """Write one JSON file per table to ``out_dir``. Returns table -> path.

    The store is checked first with `validate_store`. When a check fails,
    nothing is written - not even ``out_dir`` - and
    `DataModelValidationError` is raised, listing every problem.

    By default only the reference checks run (every record points at records
    that exist, and no time window ends before it starts), which a store
    recorded from a session passes. ``require_complete=True`` also runs the
    completeness checks (units, sampling rate, start time, file checksums),
    for a finished dataset. ``validate=False`` skips the checks and writes the
    store as it is.
    """

    if validate:
        problems = validate_store(store, require_complete=require_complete)
        if problems:
            raise DataModelValidationError(problems)
    elif require_complete:
        raise ValueError(
            "require_complete=True has no effect with validate=False: the "
            "completeness checks are part of the validation. Drop one of them."
        )

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
