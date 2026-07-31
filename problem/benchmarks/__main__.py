"""Command-line exact audit for contract-native mathematical benchmarks."""

import argparse
import json

from .exact_audit import audit_exact_benchmarks


def main(argv=None):
    """Run selected benchmarks and print either a table or canonical JSON."""
    parser = _argument_parser()
    args = parser.parse_args(argv)
    rows = audit_exact_benchmarks(
        args.benchmark_names or None,
        timeout_seconds=args.timeout_seconds,
    )
    if args.json:
        print(json.dumps(rows, ensure_ascii=False, sort_keys=True))
    else:
        _print_table(rows)
    return 0 if all(row['within_intended_budget'] for row in rows) else 1


def _argument_parser():
    """Keep CLI policy out of the public audit workflow."""
    parser = argparse.ArgumentParser(
        description=(
            'Solve deterministic mathematical benchmarks with their '
            'representation-native Exact solvers.'
        ),
    )
    parser.add_argument(
        'benchmark_names',
        nargs='*',
        help='Stable benchmark names; omit to audit the complete catalog.',
    )
    parser.add_argument(
        '--timeout-seconds',
        type=float,
        default=30 * 60,
        help='Per-instance Exact timeout (default: 1800 seconds).',
    )
    parser.add_argument(
        '--json',
        action='store_true',
        help='Print one JSON array instead of a compact table.',
    )
    return parser


def _print_table(rows):
    """Display the small audit in stable catalog order."""
    for row in rows:
        print(
            f"{row['benchmark_name']}: "
            f"{row['status']} "
            f"objective={row['objective_value']} "
            f"wall={row['wall_seconds']:.3f}s"
        )


if __name__ == '__main__':
    raise SystemExit(main())
