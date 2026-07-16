"""Command-line entry point for M3Resp.

Usage::

    m3resp run <pipeline.yaml> [--dry-run] [--debug]
    m3resp validate <pipeline.yaml> [--readiness] [--json] [--debug]
    m3resp steps [--details] [--json]
    m3resp describe <operation>

Exit codes (Phase 7.2 of the pipeline-structure plan), stable across
releases:

======  ===================================================================
Code    Meaning
======  ===================================================================
0       Success.
1       Usage error (bad arguments, unknown command).
2       Invalid/structurally invalid spec (``validate``, or ``run``
        failing static validation before any step executes).
3       Readiness failure: structurally valid but not runnable here
        (missing optional dependency, missing input file).
4       Execution failure: a step raised (``PipelineExecutionError``).
5       Cancelled (a ``cancellation_token`` stopped the run early).
======  ===================================================================

A traceback is only printed with ``--debug``; otherwise errors print a
short, researcher-readable message.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

EXIT_SUCCESS = 0
EXIT_USAGE_ERROR = 1
EXIT_INVALID_SPEC = 2
EXIT_READINESS_FAILURE = 3
EXIT_EXECUTION_FAILURE = 4
EXIT_CANCELLED = 5


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])

    if args.command is None:
        parser.print_help()
        sys.exit(EXIT_SUCCESS)

    try:
        if args.command == "run":
            sys.exit(_cmd_run(args))
        elif args.command == "validate":
            sys.exit(_cmd_validate(args))
        elif args.command == "steps":
            sys.exit(_cmd_steps(args))
        elif args.command == "describe":
            sys.exit(_cmd_describe(args))
        else:  # pragma: no cover - argparse restricts choices already
            parser.print_help()
            sys.exit(EXIT_USAGE_ERROR)
    except Exception as exc:
        if getattr(args, "debug", False):
            raise
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(_exit_code_for(exc))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="m3resp",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command")

    run_parser = subparsers.add_parser(
        "run", help="Execute a declarative pipeline spec"
    )
    run_parser.add_argument("spec", help="Path to the pipeline spec (YAML/JSON)")
    run_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compile and print the resolved plan without executing any step",
    )
    run_parser.add_argument(
        "--debug", action="store_true", help="Print a full traceback on error"
    )

    validate_parser = subparsers.add_parser(
        "validate", help="Validate a pipeline spec without running it"
    )
    validate_parser.add_argument("spec", help="Path to the pipeline spec (YAML/JSON)")
    validate_parser.add_argument(
        "--readiness",
        action="store_true",
        help="Also check optional-dependency/file-existence readiness",
    )
    validate_parser.add_argument(
        "--json", action="store_true", help="Print the report as JSON"
    )
    validate_parser.add_argument(
        "--debug", action="store_true", help="Print a full traceback on error"
    )

    steps_parser = subparsers.add_parser(
        "steps", help="List all registered pipeline steps"
    )
    steps_parser.add_argument(
        "--details",
        action="store_true",
        help="Show parameters/artifacts/capability per step",
    )
    steps_parser.add_argument(
        "--json", action="store_true", help="Print full step descriptions as JSON"
    )

    describe_parser = subparsers.add_parser(
        "describe", help="Show one operation's full discovery description"
    )
    describe_parser.add_argument(
        "operation", help="Registered step name, e.g. 'eit.load'"
    )

    return parser


def _exit_code_for(exc: Exception) -> int:
    from m3resp.core.exceptions import PipelineSpecError, UnknownStepError
    from m3resp.workflows.lifecycle import PipelineExecutionError

    if isinstance(exc, PipelineExecutionError):
        return EXIT_EXECUTION_FAILURE
    if isinstance(exc, PipelineSpecError | UnknownStepError | OSError):
        # OSError covers a spec path that doesn't exist/can't be read - a
        # problem with what was asked for, same category as a structurally
        # invalid spec, not a step execution failure.
        return EXIT_INVALID_SPEC
    return EXIT_EXECUTION_FAILURE


def _cmd_run(args: argparse.Namespace) -> int:
    from m3resp.workflows.compiler import compile_pipeline
    from m3resp.workflows.spec import load_spec

    if args.dry_run:
        parsed = load_spec(args.spec)
        compiled = compile_pipeline(parsed)
        print(json.dumps(compiled.as_dict(), indent=2, sort_keys=True))
        return EXIT_SUCCESS

    import signal

    from m3resp.workflows.engine import run_spec
    from m3resp.workflows.lifecycle import CancellationToken

    # Ctrl-C cooperatively cancels (finishes the current step, preserves
    # completed work, exits EXIT_CANCELLED) instead of raising a raw
    # KeyboardInterrupt mid-run (Phase 4.5/7.2).
    token = CancellationToken()
    previous_handler = signal.getsignal(signal.SIGINT)

    def _handle_sigint(signum: int, frame: Any) -> None:
        token.cancel()

    signal.signal(signal.SIGINT, _handle_sigint)
    try:
        # A step failure raises PipelineExecutionError - deliberately not
        # caught here, so it reaches main()'s single except block, which
        # prints either a short message or (with --debug) the full
        # traceback, uniformly for every subcommand.
        result = run_spec(args.spec, cancellation_token=token)
    finally:
        signal.signal(signal.SIGINT, previous_handler)

    if result.status == "cancelled":
        print("Pipeline cancelled.", file=sys.stderr)
        return EXIT_CANCELLED
    return EXIT_SUCCESS


def _cmd_validate(args: argparse.Namespace) -> int:
    from m3resp.workflows.compiler import validate_pipeline
    from m3resp.workflows.spec import load_spec

    parsed = load_spec(args.spec)
    report = validate_pipeline(parsed, readiness=args.readiness)

    if args.json:
        print(json.dumps(report.as_dict(), indent=2, sort_keys=True))
    else:
        _print_diagnostics("Structural", report.structural)
        if args.readiness:
            _print_diagnostics("Readiness", report.readiness)
        if report.is_valid and not any(d.severity == "error" for d in report.readiness):
            print("Valid.")

    if not report.is_valid:
        return EXIT_INVALID_SPEC
    if any(d.severity == "error" for d in report.readiness):
        return EXIT_READINESS_FAILURE
    return EXIT_SUCCESS


def _print_diagnostics(label: str, diagnostics: tuple[Any, ...]) -> None:
    if not diagnostics:
        return
    print(f"{label} diagnostics:")
    for diagnostic in diagnostics:
        location = (
            f" (step #{diagnostic.step_position} '{diagnostic.operation_id}')"
            if diagnostic.operation_id is not None
            else ""
        )
        print(
            f"  [{diagnostic.severity}] {diagnostic.code}: {diagnostic.message}{location}"
        )


def _cmd_steps(args: argparse.Namespace) -> int:
    if args.json or args.details:
        from m3resp.workflows.registry import describe_steps

        descriptions = describe_steps()
        if args.json:
            print(
                json.dumps(
                    [d.as_dict() for d in descriptions], indent=2, sort_keys=True
                )
            )
        else:
            for description in descriptions:
                print(f"{description.name}  [{description.capability}]")
                print(f"  {description.summary}")
                for parameter in description.parameters:
                    print(
                        f"    with: {parameter.name} ({parameter.value_type})"
                        f"{' required' if parameter.required else ''}"
                    )
        return EXIT_SUCCESS

    from m3resp import available_steps

    steps = available_steps()
    if not steps:
        print("No steps registered.")
        return EXIT_SUCCESS
    width = max(len(name) for name in steps)
    for name, summary in steps.items():
        print(f"  {name:<{width}}  {summary}")
    return EXIT_SUCCESS


def _cmd_describe(args: argparse.Namespace) -> int:
    from m3resp.workflows.registry import describe_step

    description = describe_step(args.operation)
    print(json.dumps(description.as_dict(), indent=2, sort_keys=True))
    return EXIT_SUCCESS


if __name__ == "__main__":
    main()
