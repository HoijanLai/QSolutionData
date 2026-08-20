"""Fixed-cardinality portfolio selection for quantum-sized candidate pools."""

from __future__ import annotations

import math
from itertools import product

import numpy as np
import pandas as pd

from ..contracts.validation import _validate_cbqm
from .client_profile import client_profile

BASIS_POINTS = 10_000
ASSET_CLASS_ORDER = (
    'cash_money',
    'fixed_income',
    'mixed',
    'equity',
    'alternative',
)

DEFAULT_OBJECTIVE_CONFIG = {
    'score_coefficients': {
        'return_score': -1.0,
        'risk_score': 1.0,
        'stability_score': -1.0,
    },
    'similarity_coefficient': 1.0,
    'similarity_transform': 'positive_only',
    'top_k_similarity': 3,
}


def build_equal_weight_selection_cbqm(
    candidate_payload,
    *,
    profile='steady',
    holding_count=10,
    objective_config=None,
    problem_id='real-asset-selection',
):
    """Build a compact ``cbqm.v1`` that selects an equal-weight portfolio.

    Each candidate contributes exactly one binary variable. Profile allocation
    ranges are deterministically tightened to feasible exact asset-class
    quotas. Exact quotas add only equality penalties during CBQM-to-QUBO
    compilation, avoiding the slack-variable explosion of general ranges.

    Manager, investment-type, and similarity-group concentration policies are
    retained as post-solve audit rules. They are not hard quantum constraints
    in this first compact surrogate and are therefore reported explicitly in
    model metadata and by :func:`audit_equal_weight_selection`.
    """
    assets, similarity = _validate_candidate_payload(candidate_payload)
    selected_profile = client_profile(profile)
    config = _resolve_objective_config(objective_config)
    _validate_holding_count(holding_count, len(assets), selected_profile)
    if BASIS_POINTS % holding_count:
        raise ValueError(
            'holding_count must divide 10,000 for exact equal-weight basis points.'
        )

    weight_bps = BASIS_POINTS // holding_count
    if weight_bps > round(selected_profile['single_asset_cap'] * BASIS_POINTS):
        raise ValueError('Equal portfolio weight exceeds the profile single-asset cap.')

    quotas = _resolve_exact_class_quotas(
        assets,
        selected_profile,
        holding_count,
    )
    partition_quotas = _resolve_exact_partition_quotas(
        assets,
        quotas,
        holding_count,
    )
    variables = _build_variables(assets, weight_bps)
    objective = _build_objective(assets, similarity, weight_bps, config)
    constraints = _build_exact_quota_constraints(assets, partition_quotas)

    problem = {
        'schema': 'cbqm.v1',
        'problem_id': problem_id,
        'variables': variables,
        'objective': objective,
        'constraints': constraints,
        'fixed_values': [],
        'metadata': {
            'source': 'real_asset_equal_weight_selection',
            'candidate_count': len(assets),
            'holding_count': int(holding_count),
            'equal_weight_basis_points': int(weight_bps),
            'budget_basis_points': BASIS_POINTS,
            'profile': selected_profile['name'],
            'profile_asset_class_ranges': {
                name: list(bounds)
                for name, bounds in selected_profile['asset_class_ranges'].items()
            },
            'exact_asset_class_quotas': dict(quotas),
            'exact_selection_partition_quotas': dict(partition_quotas),
            'objective_config': config,
            'derived_equalities': [
                'sum(asset_class_quota) = holding_count',
                'holding_count * equal_weight_basis_points = budget_basis_points',
            ],
            'post_solve_audit_limits': {
                'manager_cap': 0.25,
                'investment_type_cap': 0.35,
                'r5_cap': selected_profile['r5_cap'],
            },
            'diagnostic_limits': {'similarity_group_cap': 0.40},
            'high_similarity_groups': [
                list(group)
                for group in candidate_payload['similarity'].get(
                    'high_similarity_groups',
                    [],
                )
            ],
        },
    }
    _validate_cbqm(problem)
    return problem


def decode_equal_weight_selection(problem, sample):
    """Decode one binary sample into JSON-ready holdings."""
    if len(sample) != len(problem['variables']):
        raise ValueError('Sample length does not match selection variables.')
    holdings = []
    for bit, variable in zip(sample, problem['variables']):
        if bit not in {0, 1} or isinstance(bit, bool):
            raise ValueError('Selection sample must contain integer zero/one values.')
        if bit:
            metadata = variable.get('metadata', {})
            holdings.append({
                'variable_index': variable['index'],
                'code': metadata['code'],
                'security_name': metadata.get('security_name'),
                'asset_class': metadata['asset_class'],
                'risk_level': metadata['risk_level'],
                'investment_type_secondary': metadata[
                    'investment_type_secondary'
                ],
                'fund_manager': metadata['fund_manager'],
                'weight_basis_points': metadata['weight_basis_points'],
                'weight': metadata['weight_basis_points'] / BASIS_POINTS,
            })
    return holdings


def audit_equal_weight_selection(problem, sample):
    """Audit decoded holdings against profile ranges and concentration limits."""
    holdings = decode_equal_weight_selection(problem, sample)
    metadata = problem['metadata']
    ranges = metadata['profile_asset_class_ranges']
    limits = metadata['post_solve_audit_limits']

    class_weights = {
        name: sum(
            item['weight']
            for item in holdings
            if item['asset_class'] == name
        )
        for name in ASSET_CLASS_ORDER
    }
    manager_weights = _aggregate_weights(holdings, 'fund_manager')
    type_weights = _aggregate_weights(holdings, 'investment_type_secondary')
    violations = []

    for name, bounds in ranges.items():
        value = class_weights.get(name, 0.0)
        if value < bounds[0] - 1e-12 or value > bounds[1] + 1e-12:
            violations.append({
                'policy': 'asset_class_range',
                'key': name,
                'value': value,
                'lower_bound': bounds[0],
                'upper_bound': bounds[1],
            })

    r5_weight = sum(item['weight'] for item in holdings if item['risk_level'] == 'R5')
    if r5_weight > limits['r5_cap'] + 1e-12:
        violations.append({
            'policy': 'r5_cap',
            'key': 'R5',
            'value': r5_weight,
            'upper_bound': limits['r5_cap'],
        })
    _append_cap_violations(
        violations,
        'manager_cap',
        manager_weights,
        limits['manager_cap'],
    )
    _append_cap_violations(
        violations,
        'investment_type_cap',
        type_weights,
        limits['investment_type_cap'],
    )

    selected_codes = {item['code'] for item in holdings}
    similarity_group_weights = {}
    diagnostic_warnings = []
    similarity_cap = metadata.get('diagnostic_limits', {}).get(
        'similarity_group_cap'
    )
    for index, group in enumerate(metadata.get('high_similarity_groups', [])):
        value = sum(
            item['weight'] for item in holdings if item['code'] in set(group)
        )
        similarity_group_weights[str(index)] = value
        if similarity_cap is not None and value > similarity_cap + 1e-12:
            diagnostic_warnings.append({
                'policy': 'similarity_group_cap',
                'key': index,
                'value': value,
                'upper_bound': similarity_cap,
                'selected_codes': sorted(selected_codes.intersection(group)),
                'enforced': False,
            })

    total_bps = sum(item['weight_basis_points'] for item in holdings)
    return {
        'feasible': not violations and total_bps == BASIS_POINTS,
        'holding_count': len(holdings),
        'total_weight_basis_points': total_bps,
        'asset_class_weights': class_weights,
        'r5_weight': r5_weight,
        'manager_weights': {str(key): value for key, value in manager_weights.items()},
        'investment_type_weights': {
            str(key): value for key, value in type_weights.items()
        },
        'similarity_group_weights': similarity_group_weights,
        'violations': violations,
        'diagnostic_warnings': diagnostic_warnings,
    }


def _validate_candidate_payload(candidate_payload):
    if not isinstance(candidate_payload, dict):
        raise TypeError('candidate_payload must be a dictionary.')
    assets = candidate_payload.get('assets')
    similarity_payload = candidate_payload.get('similarity')
    if not isinstance(assets, pd.DataFrame) or assets.empty:
        raise ValueError("candidate_payload['assets'] must be a non-empty DataFrame.")
    if not isinstance(similarity_payload, dict):
        raise TypeError("candidate_payload['similarity'] must be a dictionary.")
    matrix = similarity_payload.get('similarity_matrix')
    if not isinstance(matrix, pd.DataFrame):
        raise TypeError('candidate payload must contain a similarity matrix.')

    required = {
        'code',
        'fund_manager',
        'investment_type_secondary',
        'asset_class',
        'risk_level',
        'return_score',
        'risk_score',
        'stability_score',
    }
    missing = sorted(required.difference(assets.columns))
    if missing:
        raise ValueError(f'Candidate assets are missing columns: {missing}')
    codes = assets['code'].astype('string').tolist()
    if len(codes) != len(set(codes)):
        raise ValueError('Candidate asset codes must be unique.')
    if matrix.index.astype('string').tolist() != codes:
        raise ValueError('Similarity matrix index is not aligned with candidates.')
    if matrix.columns.astype('string').tolist() != codes:
        raise ValueError('Similarity matrix columns are not aligned with candidates.')
    return assets.reset_index(drop=True), matrix


def _resolve_objective_config(config):
    resolved = {
        'score_coefficients': dict(DEFAULT_OBJECTIVE_CONFIG['score_coefficients']),
        'similarity_coefficient': DEFAULT_OBJECTIVE_CONFIG[
            'similarity_coefficient'
        ],
        'similarity_transform': DEFAULT_OBJECTIVE_CONFIG['similarity_transform'],
        'top_k_similarity': DEFAULT_OBJECTIVE_CONFIG['top_k_similarity'],
    }
    if config is not None:
        if not isinstance(config, dict):
            raise TypeError('objective_config must be a dictionary or None.')
        unknown = sorted(set(config).difference(resolved))
        if unknown:
            raise ValueError(f'Unknown objective config fields: {unknown}')
        resolved.update(config)
        if 'score_coefficients' in config:
            resolved['score_coefficients'] = {
                **DEFAULT_OBJECTIVE_CONFIG['score_coefficients'],
                **config['score_coefficients'],
            }

    if resolved['similarity_transform'] not in {'raw', 'positive_only', 'absolute'}:
        raise ValueError('Unsupported similarity transform.')
    top_k = resolved['top_k_similarity']
    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 0:
        raise ValueError('top_k_similarity must be a non-negative integer.')
    for value in resolved['score_coefficients'].values():
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise ValueError('Score coefficients must be finite numbers.')
    coefficient = resolved['similarity_coefficient']
    if not isinstance(coefficient, (int, float)) or not math.isfinite(float(coefficient)):
        raise ValueError('similarity_coefficient must be finite.')
    return {
        'score_coefficients': {
            name: float(value)
            for name, value in resolved['score_coefficients'].items()
        },
        'similarity_coefficient': float(coefficient),
        'similarity_transform': resolved['similarity_transform'],
        'top_k_similarity': top_k,
    }


def _validate_holding_count(holding_count, candidate_count, profile):
    if not isinstance(holding_count, int) or isinstance(holding_count, bool):
        raise TypeError('holding_count must be an integer.')
    if holding_count <= 0 or holding_count > candidate_count:
        raise ValueError('holding_count must be between one and candidate count.')
    lower, upper = profile['holding_count']
    if not lower <= holding_count <= upper:
        raise ValueError(
            f"holding_count={holding_count} is outside profile range {(lower, upper)}."
        )


def _resolve_exact_class_quotas(assets, profile, holding_count):
    availability = {
        name: int(assets['asset_class'].astype('string').eq(name).sum())
        for name in ASSET_CLASS_ORDER
    }
    ranges = profile['asset_class_ranges']
    count_ranges = {}
    midpoints = {}
    for name in ASSET_CLASS_ORDER:
        lower, upper = ranges[name]
        minimum = math.ceil(lower * holding_count - 1e-12)
        maximum = math.floor(upper * holding_count + 1e-12)
        maximum = min(maximum, availability[name])
        if minimum > maximum:
            raise ValueError(
                f"Candidate pool cannot satisfy profile range for '{name}'."
            )
        count_ranges[name] = range(minimum, maximum + 1)
        midpoints[name] = ((lower + upper) / 2) * holding_count

    feasible = []
    for values in product(*(count_ranges[name] for name in ASSET_CLASS_ORDER)):
        if sum(values) != holding_count:
            continue
        distance = sum(
            (value - midpoints[name]) ** 2
            for name, value in zip(ASSET_CLASS_ORDER, values)
        )
        feasible.append((distance, values))
    if not feasible:
        raise ValueError('No exact asset-class quotas satisfy the selected profile.')
    _, values = min(feasible, key=lambda item: (item[0], item[1]))
    return dict(zip(ASSET_CLASS_ORDER, values))


def _resolve_exact_partition_quotas(assets, class_quotas, holding_count):
    """Refine class quotas into disjoint secondary-type partitions.

    A 35% investment-type cap becomes an integer count cap under equal
    weighting. Refining the quantum feasible subspace here enforces that cap
    without inequality slack variables.
    """
    maximum_per_type = math.floor(0.35 * holding_count + 1e-12)
    output = {}
    for asset_class in ASSET_CLASS_ORDER:
        group = assets.loc[
            assets['asset_class'].astype('string').eq(asset_class)
        ]
        type_names = sorted(
            group['investment_type_secondary'].astype('string').unique().tolist()
        )
        ranges = []
        costs = {}
        for name in type_names:
            typed = group.loc[
                group['investment_type_secondary'].astype('string').eq(name)
            ].sort_values(['selection_cost', 'code'], kind='mergesort')
            maximum = min(maximum_per_type, len(typed))
            ranges.append(range(maximum + 1))
            values = typed['selection_cost'].astype(float).tolist()
            costs[name] = [sum(values[:count]) for count in range(maximum + 1)]

        feasible = []
        for values in product(*ranges):
            if sum(values) != class_quotas[asset_class]:
                continue
            cost = sum(
                costs[name][value]
                for name, value in zip(type_names, values)
            )
            feasible.append((cost, values))
        if not feasible:
            raise ValueError(
                'Candidate pool cannot satisfy the investment-type cap for '
                f"asset class '{asset_class}'."
            )
        _, values = min(feasible, key=lambda item: (item[0], item[1]))
        for name, value in zip(type_names, values):
            output[f'{asset_class}::{name}'] = int(value)
    return output


def _build_variables(assets, weight_bps):
    variables = []
    for index, row in enumerate(assets.itertuples(index=False)):
        metadata = {
            'code': str(row.code),
            'fund_manager': _builtin(row.fund_manager),
            'investment_type_secondary': str(row.investment_type_secondary),
            'asset_class': str(row.asset_class),
            'risk_level': str(row.risk_level),
            'weight_basis_points': int(weight_bps),
        }
        security_name = getattr(row, 'security_name', None)
        if security_name is not None and not pd.isna(security_name):
            metadata['security_name'] = str(security_name)
        variables.append({
            'index': index,
            'name': f"select::{row.code}",
            'vartype': 'BINARY',
            'kind': 'asset_selection',
            'metadata': metadata,
        })
    return variables


def _build_objective(assets, similarity, weight_bps, config):
    weight = weight_bps / BASIS_POINTS
    linear = []
    for index, row in assets.iterrows():
        coefficient = weight * sum(
            score_coefficient * float(row[column])
            for column, score_coefficient in config['score_coefficients'].items()
        )
        if coefficient:
            linear.append([int(index), float(coefficient)])

    quadratic = []
    for left, right in sorted(
        _top_k_similarity_pairs(
            similarity,
            config['top_k_similarity'],
            config['similarity_transform'],
        )
    ):
        transformed = _transform_similarity(
            float(similarity.iat[left, right]),
            config['similarity_transform'],
        )
        coefficient = (
            config['similarity_coefficient'] * transformed * weight * weight
        )
        if coefficient:
            quadratic.append([left, right, float(coefficient)])
    return {
        'sense': 'minimize',
        'offset': 0.0,
        'linear': linear,
        'quadratic': quadratic,
    }


def _top_k_similarity_pairs(matrix, top_k, transform):
    if top_k == 0:
        return set()
    pairs = set()
    size = len(matrix)
    for left in range(size):
        ranked = []
        for right in range(size):
            if left == right:
                continue
            value = _transform_similarity(float(matrix.iat[left, right]), transform)
            if value > 0:
                ranked.append((-value, right))
        ranked.sort()
        for _, right in ranked[:top_k]:
            pairs.add((min(left, right), max(left, right)))
    return pairs


def _transform_similarity(value, transform):
    if transform == 'raw':
        return value
    if transform == 'positive_only':
        return max(value, 0.0)
    return abs(value)


def _build_exact_quota_constraints(assets, quotas):
    constraints = []
    for partition_name, quota in quotas.items():
        asset_class, investment_type = partition_name.split('::', 1)
        indices = [
            int(index)
            for index, row in assets.iterrows()
            if str(row['asset_class']) == asset_class
            and str(row['investment_type_secondary']) == investment_type
        ]
        constraints.append({
            'name': f'selection_partition_quota::{partition_name}',
            'family': 'selection_partition_quota',
            'linear': [[index, 1] for index in indices],
            'lower_bound': int(quota),
            'upper_bound': int(quota),
        })
    return constraints


def _aggregate_weights(holdings, key):
    totals = {}
    for item in holdings:
        value = item[key]
        totals[value] = totals.get(value, 0.0) + item['weight']
    return totals


def _append_cap_violations(violations, policy, totals, cap):
    for key, value in totals.items():
        if value > cap + 1e-12:
            violations.append({
                'policy': policy,
                'key': _builtin(key),
                'value': value,
                'upper_bound': cap,
            })


def _builtin(value):
    return value.item() if isinstance(value, np.generic) else value


__all__ = [
    'ASSET_CLASS_ORDER',
    'BASIS_POINTS',
    'audit_equal_weight_selection',
    'build_equal_weight_selection_cbqm',
    'decode_equal_weight_selection',
]
