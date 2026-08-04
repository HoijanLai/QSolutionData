"""Strict runtime validation for ``binary-sample-set.v1``."""

from .validation import (
    _validate_binary_sample,
    _validate_json_document,
    _validate_json_object,
    _validate_mapping,
    _validate_non_negative_finite_number,
    _validate_non_negative_integer,
    _validate_object_fields,
)


_REQUIRED_FIELDS = frozenset(
    {
        'schema',
        'problem_id',
        'sample_representation',
        'variable_count',
        'distribution_kind',
        'shots',
        'records',
        'sampler',
        'seed',
        'runtime_seconds',
    }
)
_OPTIONAL_FIELDS = frozenset({'metadata'})
_DISTRIBUTION_KINDS = frozenset(
    {'empirical', 'exact', 'deterministic'}
)
_RECORD_REQUIRED_FIELDS = frozenset({'sample', 'occurrences'})
_RECORD_OPTIONAL_FIELDS = frozenset({'metadata'})
_SAMPLER_REQUIRED_FIELDS = frozenset({'name', 'version'})
_SAMPLER_OPTIONAL_FIELDS = frozenset({'backend'})


def validate_binary_sample_set(
    sample_set,
    *,
    expected_problem_id=None,
    expected_variable_count=None,
):
    """Validate one aggregated binary distribution.

    Empirical and exact distributions use a positive integer ``shots`` as
    their aggregate normalization and must conserve it exactly.  A
    deterministic producer is not shot-based: it uses ``shots: null`` and one
    record with ``occurrences: 1``.

    Record samples must be unique and strictly lexicographically ordered.
    Objective values are intentionally absent; consumers recompute them from
    the canonical source problem.
    """
    _validate_mapping(sample_set, 'sample_set')
    _validate_object_fields(
        sample_set,
        required=_REQUIRED_FIELDS,
        optional=_OPTIONAL_FIELDS,
        label='sample_set',
    )

    _validate_header(
        sample_set,
        expected_problem_id,
        expected_variable_count,
    )
    _validate_distribution(
        sample_set['distribution_kind'],
        sample_set['shots'],
        sample_set['records'],
        sample_set['variable_count'],
    )
    _validate_sampler_identity(sample_set['sampler'])
    _validate_seed(sample_set['seed'])
    _validate_non_negative_finite_number(
        sample_set['runtime_seconds'],
        'runtime_seconds',
    )
    if 'metadata' in sample_set:
        _validate_json_object(
            sample_set['metadata'],
            "sample_set['metadata']",
        )
    _validate_json_document(sample_set, 'sample_set')


def _validate_header(
    sample_set,
    expected_problem_id,
    expected_variable_count,
):
    """Validate stable identity, representation and optional source binding."""
    if sample_set['schema'] != 'binary-sample-set.v1':
        raise ValueError(
            "sample_set schema must be 'binary-sample-set.v1'."
        )
    problem_id = sample_set['problem_id']
    if not isinstance(problem_id, str) or not problem_id:
        raise ValueError('sample_set problem_id must be a non-empty string.')
    if expected_problem_id is not None:
        if (
            not isinstance(expected_problem_id, str)
            or not expected_problem_id
        ):
            raise ValueError(
                'expected_problem_id must be a non-empty string or None.'
            )
        if problem_id != expected_problem_id:
            raise ValueError(
                'sample_set problem_id must match expected_problem_id.'
            )
    if sample_set['sample_representation'] != 'binary-vector.v1':
        raise ValueError(
            "sample_representation must be 'binary-vector.v1'."
        )

    variable_count = sample_set['variable_count']
    _validate_non_negative_integer(variable_count, 'variable_count')
    if expected_variable_count is not None:
        _validate_non_negative_integer(
            expected_variable_count,
            'expected_variable_count',
        )
        if variable_count != expected_variable_count:
            raise ValueError(
                'variable_count must match expected_variable_count.'
            )


def _validate_distribution(kind, shots, records, variable_count):
    """Validate distribution mode, aggregation and canonical record order."""
    if kind not in _DISTRIBUTION_KINDS:
        raise ValueError(f"Unknown distribution_kind '{kind}'.")
    if not isinstance(records, list):
        raise TypeError("sample_set['records'] must be a list.")
    if not records:
        raise ValueError("sample_set['records'] must not be empty.")

    total_occurrences = 0
    previous_sample = None
    for position, record in enumerate(records):
        sample, occurrences = _validate_record(
            record,
            position,
            variable_count,
        )
        if (
            previous_sample is not None
            and sample <= previous_sample
        ):
            raise ValueError(
                'records must contain distinct samples in strict '
                'lexicographic order.'
            )
        previous_sample = sample
        total_occurrences += occurrences

    if kind == 'deterministic':
        _validate_deterministic_distribution(shots, records)
        return

    _validate_positive_integer(shots, 'shots')
    if total_occurrences != shots:
        raise ValueError('sum(occurrences) must equal shots.')


def _validate_record(record, position, variable_count):
    """Validate one closed record and return comparison-friendly values."""
    label = f"sample_set['records'][{position}]"
    _validate_mapping(record, label)
    _validate_object_fields(
        record,
        required=_RECORD_REQUIRED_FIELDS,
        optional=_RECORD_OPTIONAL_FIELDS,
        label=label,
    )
    _validate_binary_sample(
        record['sample'],
        variable_count,
        f"{label}['sample']",
    )
    _validate_positive_integer(
        record['occurrences'],
        f"{label}['occurrences']",
    )
    if 'metadata' in record:
        _validate_json_object(
            record['metadata'],
            f"{label}['metadata']",
        )
    return record['sample'], record['occurrences']


def _validate_deterministic_distribution(shots, records):
    """Define the null-shots exception without weakening other modes."""
    if shots is not None:
        raise ValueError(
            "distribution_kind 'deterministic' requires shots to be null."
        )
    if len(records) != 1 or records[0]['occurrences'] != 1:
        raise ValueError(
            "distribution_kind 'deterministic' requires exactly one record "
            'with occurrences equal to 1.'
        )


def _validate_sampler_identity(sampler):
    """Validate the closed sampler identity object."""
    _validate_mapping(sampler, "sample_set['sampler']")
    _validate_object_fields(
        sampler,
        required=_SAMPLER_REQUIRED_FIELDS,
        optional=_SAMPLER_OPTIONAL_FIELDS,
        label="sample_set['sampler']",
    )
    for field_name in ('name', 'version'):
        value = sampler[field_name]
        if not isinstance(value, str) or not value:
            raise ValueError(
                f"sample_set['sampler']['{field_name}'] must be non-empty."
            )
    if (
        'backend' in sampler
        and not isinstance(sampler['backend'], str)
    ):
        raise TypeError(
            "sample_set['sampler']['backend'] must be a string."
        )


def _validate_seed(seed):
    """Accept any genuine JSON integer seed or an explicit null."""
    if seed is not None and type(seed) is not int:
        raise TypeError('seed must be an integer or null.')


def _validate_positive_integer(value, label):
    """Require a genuine Python integer strictly greater than zero."""
    _validate_non_negative_integer(value, label)
    if value == 0:
        raise ValueError(f'{label} must be positive.')


__all__ = ['validate_binary_sample_set']
