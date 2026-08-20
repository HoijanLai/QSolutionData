"""End-to-end variable-weight allocation and classical/quantum comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from lib.pipeline import load_real_asset_pool, select_asset_candidates
from lib.portfolio import (
    build_variable_weight_problem,
    relax_round_reoptimize,
    reoptimize_selected_support,
    solve_joint_allocation_milp,
)
from lib.preprocessing import asset_classification, asset_scoring, asset_similarity

from .portfolio_workflow import run_real_asset_selection


def run_variable_weight_workflow(
    source_path,
    output_directory,
    *,
    profile="steady",
    candidate_count=40,
    target_holding_count=10,
    minimum_active_weight=0.01,
    policy_path=None,
    seed=7,
    run_mps=True,
    solver_time_limit=60.0,
    mps_config=None,
):
    """Run full-universe, candidate MILP, and MPS-to-LP allocation paths."""
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)

    raw_assets, provenance = load_real_asset_pool(source_path)
    scored_assets, score_weights = asset_scoring(raw_assets)
    full_assets = asset_classification(scored_assets)
    full_assets = full_assets.copy()
    full_assets["selection_cost"] = (
        -full_assets["return_score"]
        + full_assets["risk_score"]
        - full_assets["stability_score"]
    )
    full_similarity = asset_similarity(full_assets)
    full_problem = build_variable_weight_problem(
        full_assets,
        full_similarity,
        profile=profile,
        policy_path=policy_path,
        minimum_active_weight=minimum_active_weight,
        problem_id="real-asset-full-universe-variable-weight",
    )
    full_relax_round = relax_round_reoptimize(
        full_problem,
        target_holding_count=target_holding_count,
        time_limit=solver_time_limit,
    )

    candidates = select_asset_candidates(
        raw_assets,
        candidate_count=candidate_count,
    )
    candidate_problem = build_variable_weight_problem(
        candidates["assets"],
        candidates["similarity"],
        profile=profile,
        policy_path=policy_path,
        minimum_active_weight=minimum_active_weight,
        problem_id=f"real-asset-candidate-{candidate_count}-variable-weight",
    )
    candidate_relax_round = relax_round_reoptimize(
        candidate_problem,
        target_holding_count=target_holding_count,
        time_limit=solver_time_limit,
    )
    candidate_joint = solve_joint_allocation_milp(
        candidate_problem,
        target_holding_count=target_holding_count,
        time_limit=solver_time_limit,
    )

    selection_directory = output / "selection"
    selection_report = run_real_asset_selection(
        source_path,
        selection_directory,
        profile=profile,
        candidate_count=candidate_count,
        holding_count=target_holding_count,
        seed=seed,
        run_mps=run_mps,
        mps_config=mps_config,
    )
    selected_support_results = {}
    for solver_name, solver_result in selection_report["solvers"].items():
        holdings = solver_result.get("holdings") or []
        if not holdings:
            selected_support_results[solver_name] = None
            continue
        selected_support_results[solver_name] = reoptimize_selected_support(
            candidate_problem,
            [holding["code"] for holding in holdings],
            time_limit=solver_time_limit,
        )

    _write_json(output / "full-universe-problem.json", full_problem)
    _write_json(output / "candidate-problem.json", candidate_problem)
    _write_json(output / "full-relax-round-reoptimize.json", full_relax_round)
    _write_json(output / "candidate-relax-round-reoptimize.json", candidate_relax_round)
    _write_json(output / "candidate-joint-milp.json", candidate_joint)
    _write_json(
        output / "selected-support-reoptimization.json", selected_support_results
    )
    candidates["assets"].to_csv(
        output / "candidates.csv",
        index=False,
        encoding="utf-8-sig",
    )

    final_results = {
        "full_relax_round_reoptimize": _final_relax_round_result(full_relax_round),
        "candidate_relax_round_reoptimize": _final_relax_round_result(
            candidate_relax_round
        ),
        "candidate_joint_milp": candidate_joint,
    }
    for solver_name, result in selected_support_results.items():
        final_results[f"{solver_name}_selection_then_reoptimize"] = (
            None if result is None else (result["repair"] or result["direct"])
        )
    for name, result in final_results.items():
        if result is not None and result.get("allocations"):
            pd.DataFrame(result["allocations"]).to_csv(
                output / f"{name}-allocation.csv",
                index=False,
                encoding="utf-8-sig",
            )

    report = {
        "schema": "real-asset-variable-weight-workflow.v1",
        "source": {
            "path": provenance["source_path"],
            "sha256": provenance["source_sha256"],
            "asset_count": provenance["source_row_count"],
        },
        "profile": profile,
        "candidate_count": candidate_count,
        "target_holding_count": target_holding_count,
        "minimum_active_weight": minimum_active_weight,
        "score_weights": score_weights,
        "selection_report_path": str((selection_directory / "report.json").resolve()),
        "methods": {
            name: _method_summary(result) for name, result in final_results.items()
        },
    }
    _write_json(output / "report.json", report)
    return report


def _final_relax_round_result(result):
    return result.get("repair") or result.get("reoptimized")


def _method_summary(result):
    if result is None:
        return {
            "status": "not_run",
            "objective": None,
            "feasible": None,
            "holding_count": None,
            "runtime_seconds": None,
        }
    audit = result.get("audit")
    return {
        "status": result.get("status"),
        "objective": result.get("objective"),
        "feasible": None if audit is None else audit["feasible"],
        "holding_count": None if audit is None else audit["holding_count"],
        "runtime_seconds": result.get("runtime_seconds"),
        "optimality_scope": result.get("metadata", {}).get("optimality_scope"),
        "performance_proxies": (
            None if audit is None else audit["performance_proxies"]
        ),
    }


def _write_json(path, payload):
    with Path(path).open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def _default_source_path():
    sources = sorted(
        path for path in Path.cwd().glob("*.xlsx") if "copy" not in path.name.lower()
    )
    if not sources:
        raise FileNotFoundError("No source .xlsx file found in the working directory.")
    return sources[0]


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Run full-universe and 40-candidate variable-weight portfolio "
            "allocation, including MPS selection reoptimization."
        ),
    )
    parser.add_argument("--input", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("tmp/real-asset-variable-weight-run"),
    )
    parser.add_argument(
        "--profile",
        choices=("conservative", "steady", "balanced", "aggressive"),
        default="steady",
    )
    parser.add_argument("--candidate-count", type=int, default=40)
    parser.add_argument("--holding-count", type=int, default=10)
    parser.add_argument("--minimum-active-weight", type=float, default=0.01)
    parser.add_argument("--policy", type=Path)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--solver-timeout", type=float, default=60.0)
    parser.add_argument("--skip-mps", action="store_true")
    parser.add_argument("--mps-iterations", type=int, default=8)
    parser.add_argument("--mps-shots", type=int, default=4096)
    parser.add_argument("--mps-timeout", type=float, default=180.0)
    arguments = parser.parse_args(argv)
    source = _default_source_path() if arguments.input is None else arguments.input
    report = run_variable_weight_workflow(
        source,
        arguments.output,
        profile=arguments.profile,
        candidate_count=arguments.candidate_count,
        target_holding_count=arguments.holding_count,
        minimum_active_weight=arguments.minimum_active_weight,
        policy_path=arguments.policy,
        seed=arguments.seed,
        run_mps=not arguments.skip_mps,
        solver_time_limit=arguments.solver_timeout,
        mps_config={
            "optimizer_iterations": arguments.mps_iterations,
            "shots": arguments.mps_shots,
            "timeout_seconds": arguments.mps_timeout,
        },
    )
    print(json.dumps(report["methods"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["run_variable_weight_workflow"]
