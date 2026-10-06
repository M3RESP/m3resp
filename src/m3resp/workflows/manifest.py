"""Build run manifests and replace each manifest file in one operation.

The runner writes a running manifest before execution and a terminal state
after execution. Writes use a temporary file in the same directory followed
by os.replace.
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
    from m3resp.workflows.engine import WorkflowResult
    from m3resp.workflows.spec import WorkflowSpec


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


def collect_input_checksums(result: WorkflowResult) -> dict[str, str]:
    """Return SHA-256 checksums keyed by file path from executed-step parameters.

    Examines all string-valued parameters, retaining readable regular files.
    Repeated paths yield one entry; missing or unreadable files are skipped.
    """

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
    workflow_name: str,
    spec: WorkflowSpec,
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
    """Build a run manifest from workflow settings and execution records.

    Args:
        run_id: Execution identifier.
        status: Run state, such as running, succeeded, failed or cancelled.
        workflow_name: Name of the executed workflow.
        spec: Parsed workflow providing root, description, version and inputs.
        started_at: ISO 8601 UTC run start time, or None.
        finished_at: ISO 8601 UTC run finish time, or None.
        duration_seconds: Elapsed execution time in seconds, or None.
        output_dir: Resolved output directory, or None.
        step_records: Records exposing as_dict, in execution order.
        diagnostics: Diagnostics exposing as_dict.
        warnings: Captured warnings exposing as_dict.
        execution_context: Optional version/seed context exposing as_dict.
        error: Optional error details.
        checksums: Optional file paths mapped to SHA-256 checksums.

    Returns:
        dict[str, Any]: Manifest with string paths and dictionaries for the
            supplied records. Sensitive-looking top-level input keys have
            their values replaced with a redaction marker.
    """

    return {
        "run_id": run_id,
        "status": status,
        "workflow_name": workflow_name,
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
