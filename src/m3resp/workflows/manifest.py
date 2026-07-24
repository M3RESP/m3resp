"""Deterministic, atomically-written run manifests (Phase 6.3/6.4 of the
pipeline-structure plan).

A manifest is written as soon as a run starts (``status: "running"``) and
updated in place once the run reaches a terminal state, so a crash mid-run
leaves an honestly-incomplete manifest behind rather than nothing, and a
reader can never observe a half-written file: every write goes to a
temporary file in the same directory, then ``os.replace()``s it into place.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

#: Key-name substrings (case-insensitive) whose resolved input value is
#: replaced with a redaction marker in the manifest (Phase 6.3: "resolved
#: inputs with sensitive values redacted").
_SENSITIVE_KEY_MARKERS = ("password", "secret", "token", "credential", "api_key")
_REDACTED = "***redacted***"

if TYPE_CHECKING:
    from m3resp.workflows.engine import PipelineResult
    from m3resp.workflows.spec import PipelineSpec


def is_sensitive_key(name: str) -> bool:
    lowered = name.lower()
    return any(marker in lowered for marker in _SENSITIVE_KEY_MARKERS)


def redact_inputs(inputs: dict[str, Any]) -> dict[str, Any]:
    """Redact sensitive-looking values from a spec's resolved ``inputs``."""

    return {
        key: (_REDACTED if is_sensitive_key(key) else value)
        for key, value in inputs.items()
    }


def sha256_file(path: str | Path) -> str | None:
    """Return the sha256 checksum of an existing file, or ``None`` if it
    cannot be read (missing, not a regular file, permission error)."""

    try:
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 16), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return None


def collect_input_checksums(result: PipelineResult) -> dict[str, str]:
    """Sha256 every existing file referenced by a path-typed step parameter
    across the run's step records (Phase 6.3's "input ... checksums when
    configured"). Skips values that are not existing regular files."""

    checksums: dict[str, str] = {}
    for record in result.step_records:
        for value in record.parameters.values():
            if not isinstance(value, str):
                continue
            path = Path(value)
            if not path.is_file():
                continue
            digest = sha256_file(path)
            if digest is not None:
                checksums[str(path)] = digest
    return checksums


def build_manifest(
    *,
    run_id: str,
    status: str,
    pipeline_name: str,
    spec: PipelineSpec,
    started_at: str | None,
    finished_at: str | None = None,
    duration_seconds: float | None = None,
    output_dir: Path | None = None,
    step_records: tuple[Any, ...] = (),
    diagnostics: tuple[Any, ...] = (),
    warnings: tuple[Any, ...] = (),
    execution_context: Any = None,
    error: dict[str, Any] | None = None,
    checksums: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Build the JSON-safe manifest document (Phase 6.3).

    Called once with ``status="running"`` before execution starts, and once
    more with the terminal ``status`` once the run finishes (succeeded,
    failed, or cancelled) - both calls go through :func:`write_manifest_atomic`.
    """

    return {
        "run_id": run_id,
        "status": status,
        "pipeline_name": pipeline_name,
        "schema_version": spec.schema_version,
        "started_at": started_at,
        "finished_at": finished_at,
        "duration_seconds": duration_seconds,
        "spec": {
            "root": str(spec.root),
            "description": spec.description,
        },
        "inputs": redact_inputs(spec.inputs),
        "execution_context": execution_context.as_dict()
        if execution_context is not None
        else None,
        "output_dir": str(output_dir) if output_dir is not None else None,
        "step_records": [record.as_dict() for record in step_records],
        "diagnostics": [d.as_dict() for d in diagnostics],
        "warnings": [w.as_dict() for w in warnings],
        "error": error,
        "checksums": checksums,
    }


def write_manifest_atomic(path: Path, manifest: dict[str, Any]) -> Path:
    """Write ``manifest`` to ``path`` atomically: a temp file in the same
    directory, then ``os.replace()`` (Phase 6.3/6.4). A reader only ever
    sees the previous complete manifest or the new complete one, never a
    truncated write."""

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    tmp_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    os.replace(tmp_path, path)
    return path
