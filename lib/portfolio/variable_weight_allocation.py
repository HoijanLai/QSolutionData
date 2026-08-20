"""Continuous-weight portfolio contracts and reproducible classical baselines.

The module deliberately separates binary support selection from continuous
allocation.  It uses the visually transcribed policy JSON as its source of
business limits and uses SciPy/HiGHS for the ordinary LP and MILP algorithms.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_POLICY_PATH = Path(__file__).resolve().parents[2] / "资产配置约束.v1.json"
RESULT_TOLERANCE = 1e-8


def load_allocation_policy(path=None):
    """Load and minimally validate the persisted PDF-derived policy."""
    source = DEFAULT_POLICY_PATH if path is None else Path(path)
    with source.open(encoding="utf-8") as stream:
        policy = json.load(stream)
    if policy.get("schema") != "asset-allocation-policy.v1":
        raise ValueError("Policy schema must be 'asset-allocation-policy.v1'.")
    required = {"client_profiles", "default_profile", "constraints"}
    missing = sorted(required.difference(policy))
    if missing:
        raise ValueError(f"Missing allocation-policy fields: {missing}")
    return policy


def build_variable_weight_problem(
    assets,
    similarity,
    *,
    profile="steady",
    policy_path=None,
    minimum_active_weight=0.01,
    performance_constraints=True,
    problem_id="real-asset-variable-weight-allocation",
):
    """Build one JSON-safe continuous-weight plus selection problem contract."""
    _validate_assets(assets)
    policy = load_allocation_policy(policy_path)
    if profile not in policy["client_profiles"]:
        raise ValueError(f"Unknown allocation profile '{profile}'.")
    if (
        not isinstance(minimum_active_weight, (int, float))
        or isinstance(minimum_active_weight, bool)
        or not math.isfinite(minimum_active_weight)
        or minimum_active_weight <= 0
    ):
        raise ValueError("minimum_active_weight must be a positive finite number.")

    template = policy["client_profiles"][profile]
    single_cap = float(template["single_fund_weight_upper_bound"])
    if minimum_active_weight > single_cap:
        raise ValueError("minimum_active_weight exceeds the profile single-fund cap.")

    frame = assets.reset_index(drop=True).copy()
    return_proxy = _numeric_proxy(frame, ("3y_annualized_return",), absolute=False)
    volatility_proxy = _numeric_proxy(
        frame,
        ("1y_return_std", "3y_return_std"),
        absolute=True,
    )
    drawdown_proxy = _numeric_proxy(
        frame,
        ("1y_max_drawdown", "3y_max_drawdown"),
        absolute=True,
    )
    selection_cost = _selection_cost(frame)

    asset_records = []
    for index, row in frame.iterrows():
        record = {
            "index": int(index),
            "code": str(row["code"]),
            "fund_manager": _builtin(row["fund_manager"]),
            "investment_type_secondary": str(row["investment_type_secondary"]),
            "asset_class": str(row["asset_class"]),
            "risk_level": str(row["risk_level"]),
            "scores": {
                "return": float(row["return_score"]),
                "risk": float(row["risk_score"]),
                "stability": float(row["stability_score"]),
                "selection_cost": float(selection_cost.iloc[index]),
            },
            "performance_proxies": {
                "annualized_return": float(return_proxy.iloc[index]),
                "volatility": float(volatility_proxy.iloc[index]),
                "max_drawdown": float(drawdown_proxy.iloc[index]),
            },
        }
        security_name = row.get("security_name")
        if security_name is not None and not pd.isna(security_name):
            record["security_name"] = str(security_name)
        asset_records.append(record)

    high_threshold = float(policy["similarity"]["high_similarity_threshold"]["value"])
    high_groups = _similarity_groups_as_indices(
        similarity,
        frame["code"],
        threshold=high_threshold,
    )
    caps = {
        item["id"]: float(item["value"])
        for item in policy["constraints"]
        if "value" in item
    }
    problem = {
        "schema": "portfolio-allocation.v1",
        "problem_id": str(problem_id),
        "assets": asset_records,
        "objective": {
            "sense": "minimize",
            "weight_cost_field": "scores.selection_cost",
            "description": "-return_score + risk_score - stability_score",
        },
        "constraints": {
            "total_weight": 1.0,
            "holding_count": {
                "min": int(template["holding_count"]["min"]),
                "max": int(template["holding_count"]["max"]),
            },
            "minimum_active_weight": float(minimum_active_weight),
            "single_asset_weight_upper_bound": single_cap,
            "asset_class_weight_bounds": {
                name: {
                    "min": float(bounds["min"]),
                    "max": float(bounds["max"]),
                }
                for name, bounds in template["asset_class_weight_bounds"].items()
            },
            "r5_weight_upper_bound": float(template["r5_weight_upper_bound"]),
            "manager_weight_upper_bound": caps["manager_concentration_cap"],
            "secondary_type_weight_upper_bound": caps[
                "secondary_type_concentration_cap"
            ],
            "high_similarity_group_weight_upper_bound": caps[
                "high_similarity_group_cap"
            ],
            "high_similarity_groups": high_groups,
            "performance": {
                "enabled": bool(performance_constraints),
                "return_proxy_lower_bound": float(template["target_return"]),
                "volatility_proxy_upper_bound": float(
                    template["volatility_upper_bound"]
                ),
                "drawdown_proxy_upper_bound": float(
                    template["max_drawdown_upper_bound"]
                ),
            },
        },
        "metadata": {
            "profile": profile,
            "policy_schema": policy["schema"],
            "policy_source": str(
                (
                    DEFAULT_POLICY_PATH if policy_path is None else Path(policy_path)
                ).resolve()
            ),
            "performance_proxy_definition": {
                "return": "3y_annualized_return",
                "volatility": "mean(abs(1y_return_std), abs(3y_return_std))",
                "max_drawdown": ("mean(abs(1y_max_drawdown), abs(3y_max_drawdown))"),
            },
            "performance_proxy_warning": (
                "These are document-supported multi-period linear proxies, not "
                "portfolio covariance volatility or NAV-path drawdown."
            ),
            "similarity_group_definition": (
                "complete-linkage clusters whose maximum cosine distance is "
                f"<= {1.0 - high_threshold:.12g}"
            ),
        },
    }
    validate_variable_weight_problem(problem)
    return problem


def validate_variable_weight_problem(problem):
    """Validate the closed subset of ``portfolio-allocation.v1`` used here."""
    if not isinstance(problem, dict):
        raise TypeError("problem must be a dictionary.")
    if problem.get("schema") != "portfolio-allocation.v1":
        raise ValueError("Problem schema must be 'portfolio-allocation.v1'.")
    for field in ("problem_id", "assets", "objective", "constraints", "metadata"):
        if field not in problem:
            raise ValueError(f"Problem is missing field '{field}'.")
    if not isinstance(problem["problem_id"], str) or not problem["problem_id"]:
        raise ValueError("problem_id must be a non-empty string.")

    assets = problem["assets"]
    if not isinstance(assets, list) or not assets:
        raise ValueError("assets must be a non-empty list.")
    codes = []
    for expected_index, asset in enumerate(assets):
        required = {
            "index",
            "code",
            "fund_manager",
            "investment_type_secondary",
            "asset_class",
            "risk_level",
            "scores",
            "performance_proxies",
        }
        missing = sorted(required.difference(asset))
        if missing:
            raise ValueError(f"Asset {expected_index} is missing fields: {missing}")
        if asset["index"] != expected_index:
            raise ValueError("Asset indices must be contiguous and ordered.")
        code = asset["code"]
        if not isinstance(code, str) or not code:
            raise ValueError("Asset codes must be non-empty strings.")
        codes.append(code)
        for value in (
            asset["scores"]["selection_cost"],
            *asset["performance_proxies"].values(),
        ):
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"Asset '{code}' contains a non-finite coefficient.")
    if len(codes) != len(set(codes)):
        raise ValueError("Asset codes must be unique.")

    constraints = problem["constraints"]
    required_constraints = {
        "total_weight",
        "holding_count",
        "minimum_active_weight",
        "single_asset_weight_upper_bound",
        "asset_class_weight_bounds",
        "r5_weight_upper_bound",
        "manager_weight_upper_bound",
        "secondary_type_weight_upper_bound",
        "high_similarity_group_weight_upper_bound",
        "high_similarity_groups",
        "performance",
    }
    missing = sorted(required_constraints.difference(constraints))
    if missing:
        raise ValueError(f"Missing allocation constraints: {missing}")
    holding = constraints["holding_count"]
    if not 0 < holding["min"] <= holding["max"] <= len(assets):
        raise ValueError("holding_count must satisfy 0 < min <= max <= asset count.")
    minimum = constraints["minimum_active_weight"]
    cap = constraints["single_asset_weight_upper_bound"]
    if not 0 < minimum <= cap <= 1:
        raise ValueError("Active-weight bounds must satisfy 0 < min <= cap <= 1.")
    if holding["max"] * cap + RESULT_TOLERANCE < constraints["total_weight"]:
        raise ValueError("Holding-count range cannot supply the budget at the cap.")
    if holding["min"] * minimum - RESULT_TOLERANCE > constraints["total_weight"]:
        raise ValueError("Minimum holdings exceed the budget at minimum weights.")
    for bounds in constraints["asset_class_weight_bounds"].values():
        if not 0 <= bounds["min"] <= bounds["max"] <= 1:
            raise ValueError("Asset-class bounds must satisfy 0 <= min <= max <= 1.")
    valid_indices = set(range(len(assets)))
    for group in constraints["high_similarity_groups"]:
        if len(group) != len(set(group)) or not set(group).issubset(valid_indices):
            raise ValueError("High-similarity groups contain invalid indices.")
    json.dumps(problem, ensure_ascii=False, allow_nan=False)


def solve_continuous_allocation(
    problem,
    *,
    support=None,
    time_limit=30.0,
):
    """Solve an LP relaxation or reoptimize weights on one fixed support."""
    validate_variable_weight_problem(problem)
    from scipy.optimize import linprog

    started = time.perf_counter()
    count = len(problem["assets"])
    support_indices = None if support is None else _resolve_support(problem, support)
    constraints = problem["constraints"]
    cap = constraints["single_asset_weight_upper_bound"]
    minimum = constraints["minimum_active_weight"]

    if support_indices is None:
        bounds = [(0.0, cap)] * count
    else:
        holding = constraints["holding_count"]
        if not holding["min"] <= len(support_indices) <= holding["max"]:
            raise ValueError("Fixed support size is outside the profile holding range.")
        selected = set(support_indices)
        bounds = [
            (minimum, cap) if index in selected else (0.0, 0.0)
            for index in range(count)
        ]

    rows = _weight_constraint_rows(problem)
    a_ub, b_ub, a_eq, b_eq = _linprog_matrices(rows, count)
    result = linprog(
        _objective_coefficients(problem),
        A_ub=a_ub,
        b_ub=b_ub,
        A_eq=a_eq,
        b_eq=b_eq,
        bounds=bounds,
        method="highs",
        options={"time_limit": float(time_limit)},
    )
    runtime = time.perf_counter() - started
    if not result.success:
        return _empty_result(
            problem,
            solver_name="scipy-highs-lp",
            status=_scipy_status(result.status),
            message=result.message,
            runtime=runtime,
            metadata={"fixed_support": support_indices},
        )

    weights = np.asarray(result.x, dtype=float)
    weights[np.abs(weights) < RESULT_TOLERANCE] = 0.0
    audit = audit_variable_weight_allocation(
        problem,
        weights,
        relaxation=support_indices is None,
    )
    return _allocation_result(
        problem,
        weights,
        solver_name="scipy-highs-lp",
        status="optimal" if audit["feasible"] else "feasible",
        objective=float(result.fun),
        runtime=runtime,
        audit=audit,
        metadata={
            "fixed_support": support_indices,
            "iterations": int(getattr(result, "nit", 0)),
            "message": str(result.message),
            "is_continuous_relaxation": support_indices is None,
            "optimality_scope": (
                "continuous_relaxation" if support_indices is None else "fixed_support"
            ),
        },
    )


def solve_joint_allocation_milp(
    problem,
    *,
    target_holding_count=None,
    preferred_support=None,
    time_limit=60.0,
):
    """Jointly choose support and continuous weights with a HiGHS MILP."""
    validate_variable_weight_problem(problem)
    from scipy.optimize import Bounds, LinearConstraint, milp

    started = time.perf_counter()
    count = len(problem["assets"])
    constraints = problem["constraints"]
    holding = constraints["holding_count"]
    if target_holding_count is not None:
        if not isinstance(target_holding_count, int) or isinstance(
            target_holding_count, bool
        ):
            raise TypeError("target_holding_count must be an integer or None.")
        if not holding["min"] <= target_holding_count <= holding["max"]:
            raise ValueError("target_holding_count is outside the profile range.")

    objective = np.zeros(2 * count, dtype=float)
    base_cost = _objective_coefficients(problem)
    preferred = None
    if preferred_support is None:
        objective[:count] = base_cost
        objective[count:] = np.arange(count, dtype=float) * 1e-10
    else:
        preferred = set(_resolve_support(problem, preferred_support))
        scale = max(float(np.max(np.abs(base_cost))), 1.0)
        objective[:count] = base_cost / scale * 1e-6
        objective[count:] = np.array(
            [0.0 if index in preferred else 1.0 for index in range(count)]
        )

    rows = []
    lower = []
    upper = []
    for item in _weight_constraint_rows(problem):
        row = np.zeros(2 * count, dtype=float)
        row[:count] = item["coefficients"]
        rows.append(row)
        lower.append(item["lower_bound"])
        upper.append(item["upper_bound"])

    cap = constraints["single_asset_weight_upper_bound"]
    minimum = constraints["minimum_active_weight"]
    for index in range(count):
        upper_link = np.zeros(2 * count, dtype=float)
        upper_link[index] = 1.0
        upper_link[count + index] = -cap
        rows.append(upper_link)
        lower.append(-np.inf)
        upper.append(0.0)

        lower_link = np.zeros(2 * count, dtype=float)
        lower_link[index] = 1.0
        lower_link[count + index] = -minimum
        rows.append(lower_link)
        lower.append(0.0)
        upper.append(np.inf)

    count_row = np.zeros(2 * count, dtype=float)
    count_row[count:] = 1.0
    rows.append(count_row)
    if target_holding_count is None:
        lower.append(float(holding["min"]))
        upper.append(float(holding["max"]))
    else:
        lower.append(float(target_holding_count))
        upper.append(float(target_holding_count))

    result = milp(
        c=objective,
        integrality=np.r_[np.zeros(count, dtype=int), np.ones(count, dtype=int)],
        bounds=Bounds(
            np.zeros(2 * count, dtype=float),
            np.r_[
                np.full(count, cap, dtype=float),
                np.ones(count, dtype=float),
            ],
        ),
        constraints=LinearConstraint(
            np.vstack(rows),
            np.asarray(lower, dtype=float),
            np.asarray(upper, dtype=float),
        ),
        options={
            "time_limit": float(time_limit),
            "mip_rel_gap": 1e-6,
            "presolve": True,
        },
    )
    runtime = time.perf_counter() - started
    if result.x is None:
        return _empty_result(
            problem,
            solver_name="scipy-highs-milp",
            status=_scipy_status(result.status),
            message=result.message,
            runtime=runtime,
            metadata={
                "target_holding_count": target_holding_count,
                "preferred_support": sorted(preferred)
                if preferred is not None
                else None,
            },
        )

    raw_weights = np.asarray(result.x[:count], dtype=float)
    selected_indices = [
        index for index, value in enumerate(result.x[count:]) if value >= 0.5
    ]
    reoptimized = solve_continuous_allocation(
        problem,
        support=selected_indices,
        time_limit=time_limit,
    )
    if reoptimized["weights"] is not None:
        weights = np.asarray(reoptimized["weights"], dtype=float)
        objective_value = reoptimized["objective"]
        audit = reoptimized["audit"]
    else:
        weights = raw_weights
        objective_value = float(np.dot(base_cost, weights))
        audit = audit_variable_weight_allocation(problem, weights)

    metadata = {
        "target_holding_count": target_holding_count,
        "selected_indices": selected_indices,
        "preferred_support": sorted(preferred) if preferred is not None else None,
        "replacement_count": (
            None
            if preferred is None
            else len(set(selected_indices).difference(preferred))
        ),
        "message": str(result.message),
        "mip_gap": _optional_float(getattr(result, "mip_gap", None)),
        "mip_node_count": _optional_int(getattr(result, "mip_node_count", None)),
        "mip_dual_bound": _optional_float(getattr(result, "mip_dual_bound", None)),
        "weight_reoptimization_status": reoptimized["status"],
        "optimality_scope": "joint_milp",
    }
    return _allocation_result(
        problem,
        weights,
        solver_name="scipy-highs-milp",
        status="optimal" if result.status == 0 and audit["feasible"] else "feasible",
        objective=objective_value,
        runtime=runtime + reoptimized["runtime_seconds"],
        audit=audit,
        metadata=metadata,
    )


def relax_round_reoptimize(
    problem,
    *,
    target_holding_count=10,
    time_limit=60.0,
):
    """Continuous relaxation, deterministic top-k rounding, and reoptimization."""
    relaxation = solve_continuous_allocation(problem, time_limit=time_limit)
    if relaxation["weights"] is None:
        return {
            "schema": "portfolio-relax-round-reoptimize.v1",
            "problem_id": problem["problem_id"],
            "status": relaxation["status"],
            "relaxation": relaxation,
            "rounded_support": None,
            "reoptimized": None,
            "repair": None,
        }
    weights = relaxation["weights"]
    costs = _objective_coefficients(problem)
    ranking = sorted(
        range(len(weights)),
        key=lambda index: (
            -weights[index],
            costs[index],
            problem["assets"][index]["code"],
        ),
    )
    rounded = ranking[:target_holding_count]
    reoptimized = solve_continuous_allocation(
        problem,
        support=rounded,
        time_limit=time_limit,
    )
    repair = None
    final_status = (
        "feasible" if reoptimized["weights"] is not None else reoptimized["status"]
    )
    if reoptimized["weights"] is None or not reoptimized["audit"]["feasible"]:
        repair = solve_joint_allocation_milp(
            problem,
            target_holding_count=target_holding_count,
            preferred_support=rounded,
            time_limit=time_limit,
        )
        final_status = "feasible" if repair["weights"] is not None else repair["status"]
    return {
        "schema": "portfolio-relax-round-reoptimize.v1",
        "problem_id": problem["problem_id"],
        "status": final_status,
        "relaxation": relaxation,
        "rounded_support": [problem["assets"][index]["code"] for index in rounded],
        "reoptimized": reoptimized,
        "repair": repair,
    }


def reoptimize_selected_support(
    problem,
    selected_codes,
    *,
    repair=True,
    time_limit=60.0,
):
    """Allocate continuous weights on a selection and optionally repair it."""
    support = _resolve_support(problem, selected_codes)
    direct = solve_continuous_allocation(
        problem,
        support=support,
        time_limit=time_limit,
    )
    repaired = None
    if repair and (direct["weights"] is None or not direct["audit"]["feasible"]):
        repaired = solve_joint_allocation_milp(
            problem,
            target_holding_count=len(support),
            preferred_support=support,
            time_limit=time_limit,
        )
    return {
        "schema": "portfolio-selected-support-reoptimization.v1",
        "problem_id": problem["problem_id"],
        "selected_codes": [problem["assets"][index]["code"] for index in support],
        "direct": direct,
        "repair": repaired,
    }


def audit_variable_weight_allocation(
    problem,
    weights,
    *,
    relaxation=False,
    tolerance=1e-7,
):
    """Recompute every implemented policy constraint from returned weights."""
    validate_variable_weight_problem(problem)
    values = np.asarray(weights, dtype=float)
    if values.shape != (len(problem["assets"]),):
        raise ValueError("weights length must equal the number of assets.")
    if not np.isfinite(values).all():
        raise ValueError("weights must be finite.")

    violations = []
    constraints = problem["constraints"]
    selected = np.flatnonzero(values > tolerance).tolist()
    if np.min(values, initial=0.0) < -tolerance:
        violations.append(
            {
                "constraint": "nonnegative_weights",
                "magnitude": float(max(0.0, -float(np.min(values)))),
            }
        )
    if (
        np.max(values, initial=0.0)
        > constraints["single_asset_weight_upper_bound"] + tolerance
    ):
        violations.append(
            {
                "constraint": "single_asset_weight_upper_bound",
                "activity": float(np.max(values)),
                "upper_bound": constraints["single_asset_weight_upper_bound"],
                "magnitude": float(
                    np.max(values) - constraints["single_asset_weight_upper_bound"]
                ),
            }
        )

    if not relaxation:
        holding = constraints["holding_count"]
        if not holding["min"] <= len(selected) <= holding["max"]:
            violations.append(
                {
                    "constraint": "holding_count",
                    "activity": len(selected),
                    "lower_bound": holding["min"],
                    "upper_bound": holding["max"],
                    "magnitude": float(
                        max(
                            holding["min"] - len(selected),
                            len(selected) - holding["max"],
                        )
                    ),
                }
            )
        minimum = constraints["minimum_active_weight"]
        for index in selected:
            if values[index] < minimum - tolerance:
                violations.append(
                    {
                        "constraint": f"minimum_active_weight::{index}",
                        "activity": float(values[index]),
                        "lower_bound": minimum,
                        "magnitude": float(minimum - values[index]),
                    }
                )

    activities = {}
    for row in _weight_constraint_rows(problem):
        activity = float(np.dot(row["coefficients"], values))
        activities[row["name"]] = activity
        lower = row["lower_bound"]
        upper = row["upper_bound"]
        magnitude = 0.0
        if math.isfinite(lower) and activity < lower - tolerance:
            magnitude = lower - activity
        elif math.isfinite(upper) and activity > upper + tolerance:
            magnitude = activity - upper
        if magnitude:
            item = {
                "constraint": row["name"],
                "family": row["family"],
                "activity": activity,
                "magnitude": float(magnitude),
            }
            if math.isfinite(lower):
                item["lower_bound"] = lower
            if math.isfinite(upper):
                item["upper_bound"] = upper
            violations.append(item)

    proxies = {
        name: float(
            sum(
                values[index] * asset["performance_proxies"][name]
                for index, asset in enumerate(problem["assets"])
            )
        )
        for name in ("annualized_return", "volatility", "max_drawdown")
    }
    return {
        "feasible": not violations,
        "relaxation": bool(relaxation),
        "holding_count": len(selected),
        "total_weight": float(np.sum(values)),
        "performance_proxies": proxies,
        "violated_count": len(violations),
        "max_violation": max(
            (item["magnitude"] for item in violations),
            default=0.0,
        ),
        "violations": violations,
        "activities": activities,
    }


def _validate_assets(assets):
    if not isinstance(assets, pd.DataFrame) or assets.empty:
        raise ValueError("assets must be a non-empty pandas DataFrame.")
    required = {
        "code",
        "fund_manager",
        "investment_type_secondary",
        "asset_class",
        "risk_level",
        "return_score",
        "risk_score",
        "stability_score",
        "3y_annualized_return",
        "1y_return_std",
        "3y_return_std",
        "1y_max_drawdown",
        "3y_max_drawdown",
    }
    missing = sorted(required.difference(assets.columns))
    if missing:
        raise ValueError(f"Missing variable-weight asset columns: {missing}")
    codes = assets["code"].astype("string").str.strip()
    if codes.isna().any() or codes.eq("").any() or codes.duplicated().any():
        raise ValueError("Asset codes must be unique non-empty strings.")


def _numeric_proxy(frame, columns, *, absolute):
    values = pd.DataFrame(index=frame.index)
    for column in columns:
        series = pd.to_numeric(frame[column], errors="coerce").replace(
            [np.inf, -np.inf], np.nan
        )
        if series.notna().sum() == 0:
            raise ValueError(f"Proxy column '{column}' contains no numeric values.")
        series = series.fillna(series.median())
        values[column] = series.abs() if absolute else series
    return values.mean(axis=1)


def _selection_cost(frame):
    if "selection_cost" in frame:
        values = pd.to_numeric(frame["selection_cost"], errors="coerce")
    else:
        values = (
            -pd.to_numeric(frame["return_score"], errors="coerce")
            + pd.to_numeric(frame["risk_score"], errors="coerce")
            - pd.to_numeric(frame["stability_score"], errors="coerce")
        )
    if not np.isfinite(values.to_numpy(dtype=float)).all():
        raise ValueError("selection_cost must contain only finite values.")
    return values.astype(float)


def _similarity_groups_as_indices(similarity, codes, *, threshold):
    if not isinstance(similarity, dict):
        raise TypeError(
            "similarity must be the dictionary returned by asset_similarity."
        )
    matrix = similarity.get("similarity_matrix")
    if not isinstance(matrix, pd.DataFrame):
        raise TypeError("similarity_matrix must be a pandas DataFrame.")
    labels = [str(code) for code in codes]
    if matrix.index.astype(str).tolist() != labels:
        matrix = matrix.reindex(index=labels, columns=labels)
    if matrix.isna().any().any():
        raise ValueError("similarity_matrix cannot be aligned to asset codes.")
    if len(labels) < 2:
        return []

    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform

    values = matrix.to_numpy(dtype=float)
    distances = np.clip(1.0 - values, 0.0, 2.0)
    distances = (distances + distances.T) / 2.0
    np.fill_diagonal(distances, 0.0)
    clusters = fcluster(
        linkage(squareform(distances, checks=False), method="complete"),
        t=1.0 - float(threshold),
        criterion="distance",
    )
    output = []
    for cluster in sorted(set(clusters.tolist())):
        indices = np.flatnonzero(clusters == cluster).astype(int).tolist()
        if len(indices) >= 2:
            output.append(indices)
    return output


def _objective_coefficients(problem):
    return np.asarray(
        [asset["scores"]["selection_cost"] for asset in problem["assets"]],
        dtype=float,
    )


def _weight_constraint_rows(problem):
    assets = problem["assets"]
    count = len(assets)
    constraints = problem["constraints"]
    rows = []

    def append(name, family, indices, lower=-np.inf, upper=np.inf, values=None):
        coefficients = np.zeros(count, dtype=float)
        if values is None:
            coefficients[list(indices)] = 1.0
        else:
            coefficients[:] = np.asarray(values, dtype=float)
        rows.append(
            {
                "name": name,
                "family": family,
                "coefficients": coefficients,
                "lower_bound": float(lower),
                "upper_bound": float(upper),
            }
        )

    budget = constraints["total_weight"]
    append("budget", "budget", range(count), budget, budget)

    for name, bounds in constraints["asset_class_weight_bounds"].items():
        indices = [
            index for index, asset in enumerate(assets) if asset["asset_class"] == name
        ]
        append(
            f"asset_class::{name}",
            "asset_class",
            indices,
            bounds["min"],
            bounds["max"],
        )

    r5_indices = [
        index for index, asset in enumerate(assets) if asset["risk_level"] == "R5"
    ]
    append(
        "risk_level::R5",
        "risk_level",
        r5_indices,
        upper=constraints["r5_weight_upper_bound"],
    )

    for field, family, cap_name in (
        ("fund_manager", "manager", "manager_weight_upper_bound"),
        (
            "investment_type_secondary",
            "secondary_type",
            "secondary_type_weight_upper_bound",
        ),
    ):
        groups = {}
        for index, asset in enumerate(assets):
            groups.setdefault(str(asset[field]), []).append(index)
        for key, indices in sorted(groups.items()):
            if len(indices) < 2:
                continue
            append(
                f"{family}::{key}",
                family,
                indices,
                upper=constraints[cap_name],
            )

    similarity_cap = constraints["high_similarity_group_weight_upper_bound"]
    for group_index, indices in enumerate(constraints["high_similarity_groups"]):
        append(
            f"high_similarity_group::{group_index}",
            "high_similarity_group",
            indices,
            upper=similarity_cap,
        )

    performance = constraints["performance"]
    if performance["enabled"]:
        append(
            "performance::return_proxy",
            "performance",
            range(count),
            lower=performance["return_proxy_lower_bound"],
            values=[
                asset["performance_proxies"]["annualized_return"] for asset in assets
            ],
        )
        append(
            "performance::volatility_proxy",
            "performance",
            range(count),
            upper=performance["volatility_proxy_upper_bound"],
            values=[asset["performance_proxies"]["volatility"] for asset in assets],
        )
        append(
            "performance::drawdown_proxy",
            "performance",
            range(count),
            upper=performance["drawdown_proxy_upper_bound"],
            values=[asset["performance_proxies"]["max_drawdown"] for asset in assets],
        )
    return rows


def _linprog_matrices(rows, count):
    a_ub = []
    b_ub = []
    a_eq = []
    b_eq = []
    for row in rows:
        lower = row["lower_bound"]
        upper = row["upper_bound"]
        coefficients = row["coefficients"]
        if math.isfinite(lower) and math.isfinite(upper) and lower == upper:
            a_eq.append(coefficients)
            b_eq.append(lower)
        else:
            if math.isfinite(upper):
                a_ub.append(coefficients)
                b_ub.append(upper)
            if math.isfinite(lower):
                a_ub.append(-coefficients)
                b_ub.append(-lower)
    return (
        np.asarray(a_ub, dtype=float).reshape((-1, count)) if a_ub else None,
        np.asarray(b_ub, dtype=float) if b_ub else None,
        np.asarray(a_eq, dtype=float).reshape((-1, count)) if a_eq else None,
        np.asarray(b_eq, dtype=float) if b_eq else None,
    )


def _resolve_support(problem, support):
    if isinstance(support, (str, bytes)) or not hasattr(support, "__iter__"):
        raise TypeError("support must be an iterable of asset codes or indices.")
    code_to_index = {asset["code"]: asset["index"] for asset in problem["assets"]}
    output = []
    for value in support:
        if isinstance(value, int) and not isinstance(value, bool):
            index = value
        else:
            key = str(value)
            if key not in code_to_index:
                raise ValueError(f"Unknown support asset code '{key}'.")
            index = code_to_index[key]
        if index < 0 or index >= len(problem["assets"]):
            raise ValueError(f"Invalid support index {index}.")
        output.append(index)
    if len(output) != len(set(output)):
        raise ValueError("support must not contain duplicates.")
    return sorted(output)


def _allocation_result(
    problem,
    weights,
    *,
    solver_name,
    status,
    objective,
    runtime,
    audit,
    metadata,
):
    public_weights = [float(value) for value in weights]
    allocations = []
    for asset, weight in zip(problem["assets"], public_weights):
        if weight <= RESULT_TOLERANCE:
            continue
        allocations.append(
            {
                "index": asset["index"],
                "code": asset["code"],
                "security_name": asset.get("security_name"),
                "weight": weight,
                "asset_class": asset["asset_class"],
                "investment_type_secondary": asset["investment_type_secondary"],
                "risk_level": asset["risk_level"],
                "fund_manager": asset["fund_manager"],
            }
        )
    return {
        "schema": "portfolio-allocation-result.v1",
        "problem_id": problem["problem_id"],
        "solver": {"name": solver_name, "version": "scipy-highs"},
        "status": status,
        "objective": float(objective),
        "weights": public_weights,
        "allocations": allocations,
        "audit": audit,
        "runtime_seconds": float(runtime),
        "metadata": metadata,
    }


def _empty_result(problem, *, solver_name, status, message, runtime, metadata):
    output_metadata = dict(metadata)
    output_metadata["message"] = str(message)
    return {
        "schema": "portfolio-allocation-result.v1",
        "problem_id": problem["problem_id"],
        "solver": {"name": solver_name, "version": "scipy-highs"},
        "status": status,
        "objective": None,
        "weights": None,
        "allocations": None,
        "audit": None,
        "runtime_seconds": float(runtime),
        "metadata": output_metadata,
    }


def _scipy_status(status):
    return {
        0: "optimal",
        1: "timeout",
        2: "infeasible",
        3: "unbounded",
        4: "error",
    }.get(int(status), "unknown")


def _optional_float(value):
    if value is None or not math.isfinite(float(value)):
        return None
    return float(value)


def _optional_int(value):
    if value is None:
        return None
    return int(value)


def _builtin(value):
    return value.item() if isinstance(value, np.generic) else value


__all__ = [
    "DEFAULT_POLICY_PATH",
    "audit_variable_weight_allocation",
    "build_variable_weight_problem",
    "load_allocation_policy",
    "relax_round_reoptimize",
    "reoptimize_selected_support",
    "solve_continuous_allocation",
    "solve_joint_allocation_milp",
    "validate_variable_weight_problem",
]
