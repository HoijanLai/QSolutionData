"""End-to-end real-asset selection through canonical problem contracts."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from lib.compilers import compile_qubo
from lib.pipeline import load_real_asset_pool, select_asset_candidates
from lib.portfolio import (
    audit_equal_weight_selection,
    build_equal_weight_selection_cbqm,
    decode_equal_weight_selection,
)
from lib.solvers.qubo import (
    AerMpsQaoaSolver,
    SimulatedAnnealingQuboSolver,
)

from .case_operations import compile_case_qubo
from .problem_def import ProblemArtifact, ProblemCase, TaskDefinition
from .reader import save_problem_case
from .solving import solve_problem_task

DEFAULT_COMPILER_CONFIG = {
    'default_penalty': 20.0,
    'penalty_by_family': {'selection_partition_quota': 50.0},
}


def build_real_asset_selection_case(
    source_path,
    *,
    profile='steady',
    candidate_count=40,
    holding_count=10,
    top_k_similarity=3,
    problem_id='real-asset-selection-40',
    compiler_config=None,
):
    """Read the full pool and produce canonical CBQM and QUBO artifacts."""
    assets, source_provenance = load_real_asset_pool(source_path)
    candidates = select_asset_candidates(
        assets,
        candidate_count=candidate_count,
    )
    cbqm = build_equal_weight_selection_cbqm(
        candidates,
        profile=profile,
        holding_count=holding_count,
        objective_config={'top_k_similarity': top_k_similarity},
        problem_id=problem_id,
    )
    case = ProblemCase(
        problem_id=problem_id,
        artifacts=(
            ProblemArtifact(
                artifact_id='cbqm',
                representation='cbqm.v1',
                payload=cbqm,
                metadata={'role': 'canonical-selection-model'},
            ),
        ),
        tasks=(
            TaskDefinition(
                task_id='portfolio-selection',
                canonical_artifact_id='cbqm',
                task_type='equal-weight-portfolio-selection',
                sense='minimize',
                metadata={
                    'profile': profile,
                    'candidate_count': candidate_count,
                    'holding_count': holding_count,
                },
            ),
        ),
        primary_artifact_id='cbqm',
        metadata={
            'source': 'real_asset_pool',
            'source_path': source_provenance['source_path'],
            'source_sha256': source_provenance['source_sha256'],
            'source_size_bytes': source_provenance['source_size_bytes'],
            'source_asset_count': source_provenance['source_row_count'],
            'candidate_selection': candidates['metadata'],
            'candidate_quotas': candidates['candidate_quotas'],
        },
    )
    resolved_compiler_config = (
        DEFAULT_COMPILER_CONFIG
        if compiler_config is None
        else compiler_config
    )
    case = compile_case_qubo(
        case,
        'cbqm',
        resolved_compiler_config,
        target_artifact_id='qubo',
        compiler=compile_qubo,
    )
    return case, candidates


def run_real_asset_selection(
    source_path,
    output_directory,
    *,
    profile='steady',
    candidate_count=40,
    holding_count=10,
    top_k_similarity=3,
    seed=7,
    run_mps=True,
    mps_config=None,
    sa_config=None,
):
    """Build, solve, decode, audit, and persist one reproducible experiment."""
    case, candidates = build_real_asset_selection_case(
        source_path,
        profile=profile,
        candidate_count=candidate_count,
        holding_count=holding_count,
        top_k_similarity=top_k_similarity,
        problem_id=f'real-asset-selection-{candidate_count}',
    )
    cbqm = case.get_artifact('cbqm').payload
    feasible_starts = _feasible_initial_samples(cbqm, count=16, seed=seed)
    resolved_sa_config = {
        'num_reads': 16,
        'sweeps': 1000,
        'seed': seed,
        'initial_samples': feasible_starts,
        'timeout_seconds': 60.0,
    }
    if sa_config is not None:
        resolved_sa_config.update(sa_config)
    sa_record = solve_problem_task(
        case,
        'portfolio-selection',
        'qubo',
        SimulatedAnnealingQuboSolver(),
        resolved_sa_config,
        exact_verification_max_variables=24,
    )

    mps_record = None
    if run_mps:
        resolved_mps_config = {
            'layers': 1,
            'optimizer_iterations': 8,
            'restarts': 1,
            'shots': 4096,
            'seed': seed,
            'max_bond_dimension': 128,
            'truncation_threshold': 1e-8,
            'timeout_seconds': 180.0,
            'cardinality_partitions': _cardinality_partitions(cbqm),
        }
        if mps_config is not None:
            resolved_mps_config.update(mps_config)
        mps_record = solve_problem_task(
            case,
            'portfolio-selection',
            'qubo',
            AerMpsQaoaSolver(),
            resolved_mps_config,
            exact_verification_max_variables=24,
        )

    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    case_path = save_problem_case(case, output)
    candidates['assets'].to_csv(
        output / 'candidates.csv',
        index=False,
        encoding='utf-8-sig',
    )
    report = {
        'schema': 'real-asset-selection-run.v1',
        'problem_id': case.problem_id,
        'case_path': str(case_path),
        'profile': profile,
        'candidate_count': candidate_count,
        'holding_count': holding_count,
        'solvers': {
            'simulated_annealing': _record_report(cbqm, sa_record),
        },
    }
    _write_solver_outputs(output, 'simulated-annealing', cbqm, sa_record)
    if mps_record is not None:
        report['solvers']['aer_mps_qaoa'] = _record_report(cbqm, mps_record)
        _write_solver_outputs(output, 'aer-mps-qaoa', cbqm, mps_record)
    _write_json(output / 'report.json', report)
    return report


def _cardinality_partitions(cbqm):
    partitions = []
    for constraint in cbqm['constraints']:
        lower = constraint.get('lower_bound')
        upper = constraint.get('upper_bound')
        if lower != upper:
            raise ValueError('MPS cardinality partitions require equality constraints.')
        indices = [index for index, coefficient in constraint['linear'] if coefficient]
        if any(coefficient != 1 for _, coefficient in constraint['linear']):
            raise ValueError('MPS cardinality partitions require unit coefficients.')
        partitions.append({'indices': indices, 'count': int(lower)})
    return partitions


def _feasible_initial_samples(cbqm, *, count, seed):
    generator = random.Random(seed)
    samples = []
    partitions = _cardinality_partitions(cbqm)
    for _ in range(count):
        sample = [0] * len(cbqm['variables'])
        for partition in partitions:
            for index in generator.sample(
                partition['indices'],
                partition['count'],
            ):
                sample[index] = 1
        samples.append(sample)
    return samples


def _record_report(cbqm, record):
    sample = record.canonical_solution
    if sample is None:
        return {
            'status': record.raw_result['status'],
            'canonical_solution': None,
            'canonical_objective': None,
            'audit': None,
            'raw_result': record.raw_result,
        }
    return {
        'status': record.raw_result['status'],
        'canonical_solution': sample,
        'canonical_objective': record.canonical_objective_value,
        'holdings': decode_equal_weight_selection(cbqm, sample),
        'audit': audit_equal_weight_selection(cbqm, sample),
        'raw_result': record.raw_result,
    }


def _write_solver_outputs(output, name, cbqm, record):
    payload = _record_report(cbqm, record)
    _write_json(output / f'{name}-result.json', payload)
    if payload.get('holdings'):
        import pandas as pd

        pd.DataFrame(payload['holdings']).to_csv(
            output / f'{name}-allocation.csv',
            index=False,
            encoding='utf-8-sig',
        )


def _write_json(path, payload):
    with Path(path).open('w', encoding='utf-8') as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def _default_source_path():
    sources = sorted(
        path
        for path in Path.cwd().glob('*.xlsx')
        if 'copy' not in path.name.lower()
    )
    if not sources:
        raise FileNotFoundError('No source .xlsx file found in the working directory.')
    return sources[0]


def main(argv=None):
    parser = argparse.ArgumentParser(
        description='Solve a real-asset selection problem through CBQM/QUBO contracts.',
    )
    parser.add_argument('--input', type=Path)
    parser.add_argument('--output', type=Path, default=Path('tmp/real-asset-mps-run'))
    parser.add_argument(
        '--profile',
        choices=('conservative', 'steady', 'balanced', 'aggressive'),
        default='steady',
    )
    parser.add_argument('--candidate-count', type=int, default=40)
    parser.add_argument('--holding-count', type=int, default=10)
    parser.add_argument('--top-k-similarity', type=int, default=3)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--skip-mps', action='store_true')
    parser.add_argument('--mps-iterations', type=int, default=8)
    parser.add_argument('--mps-shots', type=int, default=4096)
    parser.add_argument('--mps-timeout', type=float, default=180.0)
    arguments = parser.parse_args(argv)
    source = _default_source_path() if arguments.input is None else arguments.input
    report = run_real_asset_selection(
        source,
        arguments.output,
        profile=arguments.profile,
        candidate_count=arguments.candidate_count,
        holding_count=arguments.holding_count,
        top_k_similarity=arguments.top_k_similarity,
        seed=arguments.seed,
        run_mps=not arguments.skip_mps,
        mps_config={
            'optimizer_iterations': arguments.mps_iterations,
            'shots': arguments.mps_shots,
            'timeout_seconds': arguments.mps_timeout,
        },
    )
    summary = {
        name: {
            'status': result['status'],
            'canonical_objective': result['canonical_objective'],
            'audit_feasible': (
                None if result['audit'] is None else result['audit']['feasible']
            ),
        }
        for name, result in report['solvers'].items()
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())


__all__ = [
    'build_real_asset_selection_case',
    'run_real_asset_selection',
]
