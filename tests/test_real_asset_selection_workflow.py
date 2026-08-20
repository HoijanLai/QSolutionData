from pathlib import Path

import pytest

from lib.compilers import compile_qubo
from lib.contracts import validate_cbqm, validate_qubo, validate_qubo_result
from lib.contracts.cbqm_validation import _is_cbqm_feasible_exact
from lib.pipeline import load_real_asset_pool, select_asset_candidates
from lib.portfolio import (
    audit_equal_weight_selection,
    build_equal_weight_selection_cbqm,
)
from lib.solvers.qubo import AerMpsQaoaSolver, ExactQuboSolver

ROOT = Path(__file__).resolve().parents[1]


def _source_workbook():
    candidates = sorted(
        path
        for path in ROOT.glob('*.xlsx')
        if 'copy' not in path.name.lower()
    )
    if not candidates:
        pytest.skip('Real asset workbook is not available.')
    return candidates[0]


def test_real_asset_pool_builds_a_compact_contract():
    assets, provenance = load_real_asset_pool(_source_workbook())
    payload = select_asset_candidates(assets, candidate_count=40)
    cbqm = build_equal_weight_selection_cbqm(
        payload,
        profile='steady',
        holding_count=10,
    )
    validate_cbqm(cbqm)

    assert provenance['source_row_count'] == 610
    assert payload['candidate_quotas'] == {
        'cash_money': 8,
        'fixed_income': 8,
        'mixed': 8,
        'equity': 8,
        'alternative': 8,
    }
    assert len(cbqm['variables']) == 40
    assert sum(cbqm['metadata']['exact_asset_class_quotas'].values()) == 10

    sample = [0] * 40
    for constraint in cbqm['constraints']:
        quota = constraint['lower_bound']
        for index, _ in constraint['linear'][:quota]:
            sample[index] = 1
    assert _is_cbqm_feasible_exact(cbqm, sample)
    assert audit_equal_weight_selection(cbqm, sample)['feasible']

    qubo, context = compile_qubo(
        cbqm,
        {
            'default_penalty': 20.0,
            'penalty_by_family': {'selection_partition_quota': 50.0},
        },
    )
    validate_qubo(qubo)
    assert qubo['num_variables'] == 40
    assert context['slack_variables'] == []


def test_aer_mps_solver_preserves_partition_cardinality():
    pytest.importorskip('qiskit')
    pytest.importorskip('qiskit_aer')
    problem = {
        'schema': 'qubo.v1',
        'problem_id': 'mps-cardinality-smoke',
        'sense': 'minimize',
        'num_variables': 4,
        'variable_names': ['a', 'b', 'c', 'd'],
        'offset': 20.0,
        'terms': [
            [0, 0, -12.0],
            [0, 1, 20.0],
            [1, 1, -11.0],
            [2, 2, -13.0],
            [2, 3, 20.0],
            [3, 3, -10.5],
            [0, 2, 0.2],
        ],
        'metadata': {},
    }
    result = AerMpsQaoaSolver().solve(
        problem,
        {
            'layers': 1,
            'optimizer_iterations': 0,
            'shots': 128,
            'seed': 3,
            'max_bond_dimension': 32,
            'truncation_threshold': 1e-10,
            'cardinality_partitions': [
                {'indices': [0, 1], 'count': 1},
                {'indices': [2, 3], 'count': 1},
            ],
        },
    )
    validate_qubo_result(problem, result)
    exact = ExactQuboSolver().solve(problem, {'max_variables': 4})
    assert sum(result['best_sample'][:2]) == 1
    assert sum(result['best_sample'][2:]) == 1
    assert result['best_energy'] == exact['best_energy']
    assert result['metadata']['mixer'] == 'partitioned_xy'
    assert result['metrics']['max_bond_dimension_observed'] is not None
