import copy
import math
from collections.abc import Mapping

import numpy as np
import pandas as pd

from ..contracts.validation import _validate_cbqm


SCORE_COLUMNS = (
    'return_score',
    'risk_score',
    'stability_score',
)


def build_portfolio_cbqm(payload, objective_config, problem_id=None):
    """Convert a prepared portfolio payload into the neutral ``cbqm.v1`` model.

    Args:
        payload: Output of :func:`prepare_qubo_inputs`, including aligned assets,
            similarity matrix, variable mapping, structured constraints, and
            fixed-variable names.
        objective_config: Explicit score/similarity objective configuration.
            ``score_coefficients`` and ``similarity_coefficient`` are required;
            supported transforms are ``raw``, ``positive_only``, and
            ``absolute``.
        problem_id: Optional stable identifier. When omitted, an identifier is
            derived from the selected client profile.

    Returns:
        A JSON-serializable dictionary conforming to ``cbqm.v1``. Allocation
        levels become binary variables, asset scores become linear terms,
        similarity becomes cross-asset quadratic terms, and all constraints
        remain explicit.

    Raises:
        TypeError: If payload or configuration containers have invalid types.
        ValueError: If data alignment, coefficients, bounds, or identifiers are
            invalid.
        NotImplementedError: If the requested objective strategy is unavailable.

    Notes:
        This builder does not choose penalties or encode slack variables. Those
        decisions belong to :func:`compile_qubo`.
    """
    _validate_portfolio_payload(payload)
    config = _resolve_objective_config(objective_config)
    resolved_problem_id = _resolve_problem_id(payload, problem_id)

    if config['strategy'] == 'score_similarity':
        variables = _build_cbqm_variables(payload)
        objective = _build_score_similarity_objective(payload, config)
        constraints = _build_cbqm_constraints(payload)
        fixed_values = _build_fixed_values(payload)
        problem = {
            'schema': 'cbqm.v1',
            'problem_id': resolved_problem_id,
            'variables': variables,
            'objective': objective,
            'constraints': constraints,
            'fixed_values': fixed_values,
            'metadata': _build_cbqm_metadata(payload, config),
        }
        _validate_cbqm(problem)
        return problem

    raise NotImplementedError(
        f"Objective strategy '{config['strategy']}' is not implemented."
    )


def _validate_portfolio_payload(payload):
    """Validate portfolio payload."""
    if not isinstance(payload, Mapping):
        raise TypeError('payload must be a mapping.')

    required = {
        'assets',
        'similarity_matrix',
        'variable_mapping',
        'constraints',
        'fixed_variables',
        'metadata',
    }
    missing = sorted(required.difference(payload))
    if missing:
        raise ValueError(f'Missing portfolio payload fields: {missing}')

    assets = payload['assets']
    if not isinstance(assets, pd.DataFrame):
        raise TypeError("payload['assets'] must be a pandas DataFrame.")
    if assets.empty:
        raise ValueError("payload['assets'] must not be empty.")
    asset_columns = {'code', *SCORE_COLUMNS}
    missing_asset_columns = sorted(asset_columns.difference(assets.columns))
    if missing_asset_columns:
        raise ValueError(f'Missing asset columns: {missing_asset_columns}')

    codes = assets['code'].astype('string').str.strip()
    if codes.isna().any() or codes.eq('').any() or codes.duplicated().any():
        raise ValueError('Asset codes must be unique, non-missing strings.')
    for column in SCORE_COLUMNS:
        values = pd.to_numeric(assets[column], errors='coerce')
        if values.isna().any() or not np.isfinite(values).all():
            raise ValueError(f"Asset score column '{column}' must be finite.")

    mapping = payload['variable_mapping']
    if not isinstance(mapping, pd.DataFrame):
        raise TypeError("payload['variable_mapping'] must be a pandas DataFrame.")
    required_mapping = {
        'variable_index',
        'variable_name',
        'asset_index',
        'level_index',
        'code',
        'weight_level',
    }
    missing_mapping = sorted(required_mapping.difference(mapping.columns))
    if missing_mapping:
        raise ValueError(f'Missing variable mapping columns: {missing_mapping}')
    _validate_variable_mapping(mapping, codes.tolist())
    _validate_similarity_matrix(payload['similarity_matrix'], codes.tolist())
    _validate_payload_constraints(payload, mapping)


def _validate_variable_mapping(mapping, asset_codes):
    """Validate variable mapping."""
    if mapping.empty:
        raise ValueError('Variable mapping must not be empty.')
    expected_indices = list(range(len(mapping)))
    if mapping['variable_index'].tolist() != expected_indices:
        raise ValueError('Variable indices must be consecutive and ordered.')
    if mapping['variable_name'].isna().any():
        raise ValueError('Variable names must not be missing.')
    names = mapping['variable_name'].astype(str).tolist()
    if any(not name for name in names) or len(names) != len(set(names)):
        raise ValueError('Variable names must be unique, non-empty strings.')

    mapped_codes = mapping['code'].astype('string').str.strip().tolist()
    if set(mapped_codes) != set(asset_codes):
        raise ValueError('Variable mapping codes must match asset codes.')
    code_positions = {code: index for index, code in enumerate(asset_codes)}
    for row in mapping.itertuples(index=False):
        code = str(row.code).strip()
        if row.asset_index != code_positions[code]:
            raise ValueError('Variable asset indices are not aligned with assets.')
        if (
            not isinstance(row.level_index, (int, np.integer))
            or row.level_index < 0
        ):
            raise ValueError('Variable level indices must be non-negative integers.')
        if not math.isfinite(float(row.weight_level)) or not 0 <= row.weight_level <= 1:
            raise ValueError('Variable weight levels must be between 0 and 1.')


def _validate_similarity_matrix(matrix, asset_codes):
    """Validate similarity matrix."""
    if not isinstance(matrix, pd.DataFrame):
        raise TypeError("payload['similarity_matrix'] must be a pandas DataFrame.")
    if matrix.shape != (len(asset_codes), len(asset_codes)):
        raise ValueError('Similarity matrix shape must match the assets.')
    if matrix.index.astype('string').tolist() != asset_codes:
        raise ValueError('Similarity matrix index is not aligned with assets.')
    if matrix.columns.astype('string').tolist() != asset_codes:
        raise ValueError('Similarity matrix columns are not aligned with assets.')
    values = matrix.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError('Similarity matrix must contain only finite values.')
    if not np.allclose(values, values.T, rtol=1e-9, atol=1e-9):
        raise ValueError('Similarity matrix must be symmetric.')


def _validate_payload_constraints(payload, mapping):
    """Validate payload constraints."""
    constraints = payload['constraints']
    if not isinstance(constraints, list):
        raise TypeError("payload['constraints'] must be a list.")
    valid_variables = set(mapping['variable_name'].astype(str))
    names = []
    for constraint in constraints:
        if not isinstance(constraint, Mapping):
            raise TypeError('Each payload constraint must be a mapping.')
        name = constraint.get('name')
        family = constraint.get('family')
        terms = constraint.get('terms')
        if not isinstance(name, str) or not name:
            raise ValueError('Constraint names must be non-empty strings.')
        if not isinstance(family, str) or not family:
            raise ValueError('Constraint families must be non-empty strings.')
        if not isinstance(terms, Mapping):
            raise TypeError(f"Constraint '{name}' terms must be a mapping.")
        unknown = set(terms).difference(valid_variables)
        if unknown:
            raise ValueError(f"Constraint '{name}' references unknown variables.")
        if any(not _is_finite_number(value) for value in terms.values()):
            raise ValueError(f"Constraint '{name}' coefficients must be finite.")
        lower = constraint.get('lower_bound')
        upper = constraint.get('upper_bound')
        if lower is None and upper is None:
            raise ValueError(f"Constraint '{name}' must have at least one bound.")
        if lower is not None and not _is_finite_number(lower):
            raise ValueError(f"Constraint '{name}' lower bound must be finite.")
        if upper is not None and not _is_finite_number(upper):
            raise ValueError(f"Constraint '{name}' upper bound must be finite.")
        if lower is not None and upper is not None and lower > upper:
            raise ValueError(f"Constraint '{name}' has inconsistent bounds.")
        names.append(name)
    if len(names) != len(set(names)):
        raise ValueError('Constraint names must be unique.')

    fixed_variables = payload['fixed_variables']
    if not isinstance(fixed_variables, list):
        raise TypeError("payload['fixed_variables'] must be a list.")
    if len(fixed_variables) != len(set(fixed_variables)):
        raise ValueError('Fixed variable names must be unique.')
    if set(fixed_variables).difference(valid_variables):
        raise ValueError('Fixed variables must exist in the variable mapping.')


def _resolve_objective_config(objective_config):
    """Resolve objective config."""
    if not isinstance(objective_config, Mapping):
        raise TypeError('objective_config must be an explicit mapping.')
    allowed = {
        'strategy',
        'sense',
        'offset',
        'score_coefficients',
        'similarity_coefficient',
        'similarity_transform',
    }
    unknown = sorted(set(objective_config).difference(allowed))
    if unknown:
        raise ValueError(f'Unknown objective config fields: {unknown}')

    required = {'score_coefficients', 'similarity_coefficient'}
    missing = sorted(required.difference(objective_config))
    if missing:
        raise ValueError(f'Missing objective config fields: {missing}')

    strategy = objective_config.get('strategy', 'score_similarity')
    sense = objective_config.get('sense', 'minimize')
    offset = objective_config.get('offset', 0.0)
    similarity_transform = objective_config.get('similarity_transform', 'raw')
    if not isinstance(strategy, str) or not strategy:
        raise ValueError('Objective strategy must be a non-empty string.')
    if sense not in {'minimize', 'maximize'}:
        raise ValueError("Objective sense must be 'minimize' or 'maximize'.")
    if similarity_transform not in {'raw', 'positive_only', 'absolute'}:
        raise ValueError(
            "similarity_transform must be 'raw', 'positive_only', or 'absolute'."
        )
    if not _is_finite_number(offset):
        raise ValueError('Objective offset must be finite.')
    similarity_coefficient = objective_config['similarity_coefficient']
    if not _is_finite_number(similarity_coefficient):
        raise ValueError('similarity_coefficient must be finite.')

    score_coefficients = objective_config['score_coefficients']
    if not isinstance(score_coefficients, Mapping) or not score_coefficients:
        raise ValueError('score_coefficients must be a non-empty mapping.')
    unknown_scores = sorted(set(score_coefficients).difference(SCORE_COLUMNS))
    if unknown_scores:
        raise ValueError(f'Unknown score coefficients: {unknown_scores}')
    if any(not _is_finite_number(value) for value in score_coefficients.values()):
        raise ValueError('Score coefficients must be finite.')

    return {
        'strategy': strategy,
        'sense': sense,
        'offset': float(offset),
        'score_coefficients': {
            column: float(score_coefficients.get(column, 0.0))
            for column in SCORE_COLUMNS
        },
        'similarity_coefficient': float(similarity_coefficient),
        'similarity_transform': similarity_transform,
    }


def _resolve_problem_id(payload, problem_id):
    """Resolve problem id."""
    if problem_id is None:
        profile = payload.get('metadata', {}).get('profile', 'portfolio')
        return f'portfolio-{profile}'
    if not isinstance(problem_id, str) or not problem_id.strip():
        raise ValueError('problem_id must be a non-empty string or None.')
    return problem_id.strip()


def _build_cbqm_variables(payload):
    """Build cbqm variables."""
    assets = payload['assets'].set_index('code', drop=False)
    variables = []
    for row in payload['variable_mapping'].itertuples(index=False):
        asset = assets.loc[str(row.code)]
        metadata = {
            'code': str(row.code),
            'asset_index': int(row.asset_index),
            'level_index': int(row.level_index),
            'weight_level': float(row.weight_level),
        }
        for column in (
            'security_name',
            'fund_manager',
            'investment_type_secondary',
            'asset_class',
            'risk_level',
            'style_cluster',
        ):
            if column in asset and not pd.isna(asset[column]):
                metadata[column] = _to_builtin(asset[column])
        variables.append({
            'index': int(row.variable_index),
            'name': str(row.variable_name),
            'vartype': 'BINARY',
            'kind': 'allocation_level',
            'metadata': metadata,
        })
    return variables


def _build_score_similarity_objective(payload, config):
    """Build score similarity objective."""
    return {
        'sense': config['sense'],
        'offset': config['offset'],
        'linear': _build_score_terms(payload, config['score_coefficients']),
        'quadratic': _build_similarity_terms(payload, config),
    }


def _build_score_terms(payload, score_coefficients):
    """Build score terms."""
    assets = payload['assets'].set_index('code')
    terms = []
    for row in payload['variable_mapping'].itertuples(index=False):
        asset = assets.loc[str(row.code)]
        asset_score = sum(
            coefficient * float(asset[column])
            for column, coefficient in score_coefficients.items()
        )
        coefficient = float(row.weight_level) * asset_score
        if coefficient != 0:
            terms.append([int(row.variable_index), float(coefficient)])
    return terms


def _build_similarity_terms(payload, config):
    """Build similarity terms."""
    if config['similarity_coefficient'] == 0:
        return []

    mapping = payload['variable_mapping']
    groups = {
        str(code): group
        for code, group in mapping.groupby('code', sort=False, observed=True)
    }
    codes = payload['assets']['code'].astype('string').tolist()
    matrix = payload['similarity_matrix']
    terms = []
    for left_position, left_code in enumerate(codes):
        for right_code in codes[left_position + 1:]:
            similarity = _transform_similarity(
                float(matrix.loc[left_code, right_code]),
                config['similarity_transform'],
            )
            pair_coefficient = config['similarity_coefficient'] * similarity
            if pair_coefficient == 0:
                continue
            for left in groups[left_code].itertuples(index=False):
                if left.weight_level == 0:
                    continue
                for right in groups[right_code].itertuples(index=False):
                    if right.weight_level == 0:
                        continue
                    coefficient = (
                        pair_coefficient
                        * float(left.weight_level)
                        * float(right.weight_level)
                    )
                    if coefficient != 0:
                        terms.append([
                            int(left.variable_index),
                            int(right.variable_index),
                            float(coefficient),
                        ])
    return terms


def _transform_similarity(value, strategy):
    """Transform similarity."""
    if strategy == 'raw':
        return value
    if strategy == 'positive_only':
        return max(value, 0.0)
    if strategy == 'absolute':
        return abs(value)
    raise NotImplementedError(f"Similarity transform '{strategy}' is not implemented.")


def _build_cbqm_constraints(payload):
    """Build cbqm constraints."""
    mapping = payload['variable_mapping']
    index_by_name = dict(
        zip(mapping['variable_name'], mapping['variable_index'])
    )
    constraints = []
    for source in payload['constraints']:
        linear = [
            [int(index_by_name[name]), float(coefficient)]
            for name, coefficient in source['terms'].items()
            if coefficient != 0
        ]
        linear.sort(key=lambda term: term[0])
        constraint = {
            'name': source['name'],
            'family': source['family'],
            'linear': linear,
        }
        if source.get('lower_bound') is not None:
            constraint['lower_bound'] = float(source['lower_bound'])
        if source.get('upper_bound') is not None:
            constraint['upper_bound'] = float(source['upper_bound'])
        source_type = source.get('type')
        if source_type is not None:
            constraint['metadata'] = {'source_type': source_type}
        constraints.append(constraint)
    return constraints


def _build_fixed_values(payload):
    """Build fixed values."""
    mapping = payload['variable_mapping']
    index_by_name = dict(
        zip(mapping['variable_name'], mapping['variable_index'])
    )
    return [
        {'index': int(index_by_name[name]), 'value': 0}
        for name in sorted(payload['fixed_variables'], key=index_by_name.get)
    ]


def _build_cbqm_metadata(payload, config):
    """Build cbqm metadata."""
    source_metadata = payload.get('metadata', {})
    return {
        'source': 'portfolio_qubo_payload',
        'profile': source_metadata.get('profile'),
        'asset_count': int(len(payload['assets'])),
        'variable_count': int(len(payload['variable_mapping'])),
        'objective_config': copy.deepcopy(config),
    }


def _is_finite_number(value):
    """Return whether a value is a finite, non-boolean real number."""
    return (
        isinstance(value, (int, float, np.integer, np.floating))
        and not isinstance(value, (bool, np.bool_))
        and math.isfinite(float(value))
    )


def _to_builtin(value):
    """Convert a NumPy scalar to its built-in Python equivalent."""
    if isinstance(value, np.generic):
        return value.item()
    return value
