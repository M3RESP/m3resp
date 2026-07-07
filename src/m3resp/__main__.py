"""Command-line entry point for M3Resp.

Usage::

    python -m m3resp run <pipeline.yaml>
    m3resp run <pipeline.yaml>          # after pip install

The ``run`` sub-command loads a declarative pipeline spec, executes every step,
and handles output export — no custom Python script required.
"""

from __future__ import annotations

import sys


def main(argv: list[str] | None = None) -> None:
    args = argv if argv is not None else sys.argv[1:]

    if not args or args[0] in {"-h", "--help"}:
        _print_help()
        return

    command = args[0]

    if command == "run":
        if len(args) < 2:
            print("Error: 'run' requires a pipeline spec path.", file=sys.stderr)
            print("Usage: m3resp run <pipeline.yaml>", file=sys.stderr)
            sys.exit(1)
        _cmd_run(args[1])

    elif command == "steps":
        _cmd_steps()

    else:
        print(f"Error: unknown command '{command}'.", file=sys.stderr)
        _print_help()
        sys.exit(1)


def _cmd_run(spec_path: str) -> None:
    from m3resp.workflows.engine import run_spec

    run_spec(spec_path)


def _cmd_steps() -> None:
    from m3resp import available_steps

    steps = available_steps()
    if not steps:
        print("No steps registered.")
        return
    width = max(len(name) for name in steps)
    for name, summary in steps.items():
        print(f"  {name:<{width}}  {summary}")


def _print_help() -> None:
    print(
        "Usage: m3resp <command> [args]\n"
        "\n"
        "Commands:\n"
        "  run <spec.yaml>   Execute a declarative pipeline spec\n"
        "  steps             List all registered pipeline steps\n"
    )


if __name__ == "__main__":
    main()
