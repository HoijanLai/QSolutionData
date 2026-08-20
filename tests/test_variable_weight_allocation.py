from pathlib import Path

import numpy as np
import pytest

from lib.pipeline import load_real_asset_pool, select_asset_candidates
from lib.portfolio import (
    build_variable_weight_problem,
    relax_round_reoptimize,
    reoptimize_selected_support,
    solve_joint_allocation_milp,
    validate_variable_weight_problem,
)

ROOT = Path(__file__).resolve().parents[1]


def _source_workbook():
    candidates = sorted(
        path for path in ROOT.glob("*.xlsx") if "copy" not in path.name.lower()
    )
    if not candidates:
        pytest.skip("Real asset workbook is not available.")
    return candidates[0]


@pytest.fixture(scope="module")
def candidate_problem():
    pytest.importorskip("scipy")
    assets, _ = load_real_asset_pool(_source_workbook())
    candidates = select_asset_candidates(assets, candidate_count=40)
    return build_variable_weight_problem(
        candidates["assets"],
        candidates["similarity"],
        profile="steady",
        minimum_active_weight=0.01,
        problem_id="variable-weight-test",
    )


def test_builds_pdf_policy_backed_continuous_contract(candidate_problem):
    validate_variable_weight_problem(candidate_problem)
    constraints = candidate_problem["constraints"]

    assert candidate_problem["schema"] == "portfolio-allocation.v1"
    assert len(candidate_problem["assets"]) == 40
    assert constraints["holding_count"] == {"min": 8, "max": 15}
    assert constraints["single_asset_weight_upper_bound"] == 0.15
    assert constraints["manager_weight_upper_bound"] == 0.25
    assert constraints["secondary_type_weight_upper_bound"] == 0.35
    assert constraints["high_similarity_group_weight_upper_bound"] == 0.40
    assert constraints["performance"]["enabled"]


def test_joint_milp_returns_feasible_non_equal_weights(candidate_problem):
    result = solve_joint_allocation_milp(
        candidate_problem,
        target_holding_count=10,
        time_limit=30,
    )

    assert result["status"] == "optimal"
    assert result["audit"]["feasible"]
    assert result["audit"]["holding_count"] == 10
    positive = [weight for weight in result["weights"] if weight > 1e-8]
    assert len(positive) == 10
    assert not np.allclose(positive, [0.1] * 10)
    assert sum(positive) == pytest.approx(1.0)


def test_relax_round_reoptimize_and_selected_support_share_audit(candidate_problem):
    baseline = relax_round_reoptimize(
        candidate_problem,
        target_holding_count=10,
        time_limit=30,
    )
    final = baseline["repair"] or baseline["reoptimized"]

    assert baseline["status"] == "feasible"
    assert final["audit"]["feasible"]
    selected_codes = [item["code"] for item in final["allocations"]]

    repeated = reoptimize_selected_support(
        candidate_problem,
        selected_codes,
        time_limit=30,
    )
    assert repeated["repair"] is None
    assert repeated["direct"]["status"] == "optimal"
    assert repeated["direct"]["audit"]["feasible"]
    assert repeated["direct"]["objective"] == pytest.approx(final["objective"])


def test_records_complete_linkage_similarity_group_definition(candidate_problem):
    # The problem records the construction rule because transitive connected
    # components can merge assets that are not mutually similar.
    definition = candidate_problem["metadata"]["similarity_group_definition"]
    assert definition.startswith("complete-linkage clusters")
    assert candidate_problem["constraints"]["high_similarity_groups"]
