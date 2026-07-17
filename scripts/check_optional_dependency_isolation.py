"""Smoke test for CI's optional-dependency install matrix

Run once per install in `.github/workflows/tests.yml`'s ``optional-deps``
matrix job, each in its own fresh virtualenv/extras combination (``""``,
``eit``, ``emg``, ``all``) - unlike `tests/test_optional_dependency_absence.py`,
which simulates "package absent" in-process via `sys.modules[name] = None`
inside a single environment that actually has both packages installed, this
script checks the real, as-installed package set.

Usage: ``python scripts/check_optional_dependency_isolation.py eit|emg|all|base``
"""

from __future__ import annotations

import argparse
import importlib.util
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "extra",
        choices=["base", "eit", "emg", "all"],
        help="Which optional-dependency extra this environment was installed with.",
    )
    args = parser.parse_args()

    import m3resp  # noqa: F401 - importing m3resp itself must never require an optional package
    import m3resp.workflows.steps  # noqa: F401 - ensure built-in steps are registered
    from m3resp.workflows.registry import describe_steps

    eitprocessing_installed = importlib.util.find_spec("eitprocessing") is not None
    resurfemg_installed = importlib.util.find_spec("resurfemg") is not None

    expected_eitprocessing = args.extra in ("eit", "all")
    expected_resurfemg = args.extra in ("emg", "all")
    if eitprocessing_installed != expected_eitprocessing:
        print(
            f"eitprocessing installed={eitprocessing_installed}, "
            f"expected={expected_eitprocessing} for extra={args.extra!r}",
            file=sys.stderr,
        )
        return 1
    if resurfemg_installed != expected_resurfemg:
        print(
            f"resurfemg installed={resurfemg_installed}, "
            f"expected={expected_resurfemg} for extra={args.extra!r}",
            file=sys.stderr,
        )
        return 1

    descriptions = describe_steps()
    if not descriptions:
        print("describe_steps() returned no steps", file=sys.stderr)
        return 1

    # Recompute each step's expected capability from its own declared
    # `optional_packages` (Phase 1 metadata) rather than guessing from its
    # name prefix - a `eit.*`/`emg.*` step is not guaranteed to require the
    # matching upstream package (e.g. `emg.ecg_estimated_subtraction` is a
    # native m3resp implementation with no `optional_packages` at all, see
    # docs/developer/adapters.md).
    failures = []
    for description in descriptions:
        missing = [
            package
            for package in description.optional_packages
            if importlib.util.find_spec(package) is None
        ]
        expected_state = "missing_optional_dependency" if missing else "available"
        if description.capability != expected_state:
            failures.append(
                f"{description.name}: state={description.capability!r}, "
                f"expected={expected_state!r} (optional_packages={description.optional_packages!r})"
            )

    # Sanity check that the matrix leg actually varies what it claims to:
    # for the "eit"/"emg"/"all" legs, at least one step in that modality
    # must genuinely require the just-installed package, or this script
    # would pass vacuously (e.g. if every eit.* step became native).
    if args.extra in ("eit", "all") and not any(
        "eitprocessing" in d.optional_packages for d in descriptions
    ):
        print(
            "No step declares eitprocessing as an optional package - test would be vacuous",
            file=sys.stderr,
        )
        return 1
    if args.extra in ("emg", "all") and not any(
        "resurfemg" in d.optional_packages for d in descriptions
    ):
        print(
            "No step declares resurfemg as an optional package - test would be vacuous",
            file=sys.stderr,
        )
        return 1

    if failures:
        print("Unexpected step capability states:", file=sys.stderr)
        for failure in failures:
            print(f"  {failure}", file=sys.stderr)
        return 1

    print(
        f"OK: extra={args.extra!r}, {len(descriptions)} steps described, "
        f"eitprocessing={eitprocessing_installed}, resurfemg={resurfemg_installed}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
