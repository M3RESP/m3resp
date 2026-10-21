"""Rewrite tests/regression/real_data/reference_values.json.

Runs the real-data workflows on every dataset in
tests/regression/real_data/datasets.yaml and stores the collected numbers.
Run it after a change that is meant to alter results, then review the diff of
the JSON file: it shows which numbers moved and by how much.

Usage (from the repository root):

    python scripts/update_real_data_references.py
    uv run python scripts/update_real_data_references.py

The recordings are read from the folder named by M3RESP_REAL_DATA (default:
data/source/Test_PS_27072026).
"""

from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(
    0, os.path.join(os.path.dirname(HERE), "tests", "regression", "real_data")
)

import real_data_helpers as helpers


def main() -> int:
    results = {}
    for name in helpers.load_datasets():
        paths, reason = helpers.dataset_paths(name)
        if paths is None:
            print(f"Cannot update {name}: {reason}", file=sys.stderr)
            return 1
        results[name] = helpers.collect_results(helpers.run_dataset(name, paths))
        print(f"{name}: done")
    helpers.write_reference_values(results)
    print(f"Wrote {helpers.REFERENCE_FILE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
