"""Command-line tools for inspecting persisted problem collections."""

from __future__ import annotations

import argparse
import json
import sys

from .validation import (
    ProblemValidationError,
    validate_problem_path,
)


def main(argv=None) -> int:
    """Run one problem command and return its process exit code."""
    parser = _build_parser()
    arguments = parser.parse_args(argv)
    return arguments.handler(arguments)


def _build_parser():
    """Keep command registration separate from validation orchestration."""
    parser = argparse.ArgumentParser(
        prog='python -m problem',
        description='Inspect and validate persisted problem cases.',
    )
    commands = parser.add_subparsers(dest='command', required=True)

    validate_parser = commands.add_parser(
        'validate',
        help='deep-check one problem case or one collection directory',
    )
    validate_parser.add_argument(
        'path',
        nargs='?',
        default='problem/data',
        help='case.json, case directory, or collection root',
    )
    validate_parser.add_argument(
        '--strict',
        action='store_true',
        help='fail when custom semantics cannot be checked',
    )
    validate_parser.add_argument(
        '--json',
        action='store_true',
        dest='json_output',
        help='emit a machine-readable report',
    )
    validate_parser.set_defaults(handler=_run_validate)
    return parser


def _run_validate(arguments):
    """Load, validate and render one report without exposing a traceback."""
    try:
        report = validate_problem_path(
            arguments.path,
            strict=arguments.strict,
        )
    except (
        OSError,
        TypeError,
        ValueError,
        ProblemValidationError,
    ) as error:
        _render_error(error, arguments.json_output)
        return 1

    _render_success(report, arguments.json_output)
    return 0


def _render_success(report, json_output):
    """Print a compact human summary or stable JSON for CI."""
    if json_output:
        print(
            json.dumps(
                report.to_dict(),
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return

    qualification = (
        'fully checked'
        if report.fully_checked
        else 'valid with warnings'
    )
    print(
        f'OK ({qualification}): {report.case_count} case(s), '
        f'{report.artifact_count} artifact(s), '
        f'{report.task_count} task(s), '
        f'{report.best_known_count} best-known witness(es).'
    )
    print(
        'Deep checks: '
        f'{report.validated_artifact_count}/'
        f'{report.artifact_count} artifacts, '
        f'{report.validated_best_known_count}/'
        f'{report.best_known_count} best-known witnesses.'
    )
    if report.representation_counts:
        counts = ', '.join(
            f'{name}={count}'
            for name, count in report.representation_counts
        )
        print(f'Representations: {counts}')
    for warning in report.warnings:
        print(f'WARNING: {warning}')


def _render_error(error, json_output):
    """Send expected user/data failures to stderr with no traceback."""
    if json_output:
        print(
            json.dumps(
                {
                    'ok': False,
                    'error': str(error),
                    'exception_type': type(error).__name__,
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return
    print(f'ERROR: {error}', file=sys.stderr)


if __name__ == '__main__':
    raise SystemExit(main())
