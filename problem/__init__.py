"""Unified problem cases, model artifacts, graph views and best-known results."""

from .case_operations import (
    compile_case_qubo,
    create_graph_case,
    derive_cbqm_factor_graph,
    derive_qubo_interaction_graph,
    derive_qubo_maxcut_graph,
    project_qubo_sample_to_cbqm,
)
from .graph_codec import (
    artifact_from_networkx,
    artifact_to_networkx,
    graph_from_node_link,
    graph_to_node_link,
)
from .problem_def import (
    PROBLEM_CASE_SCHEMA,
    BestKnownSolution,
    ProblemArtifact,
    ProblemCase,
    ProblemSet,
    TaskDefinition,
    TransformationRecord,
)
from .reader import (
    case_from_dict,
    case_to_dict,
    load_problem,
    load_problem_case,
    load_problem_set,
    problem_to_dict,
    save_problem,
    save_problem_case,
)
from .registries import (
    NativeSolverRunner,
    register_native_solver_runner,
    register_representation_validator,
    register_task_evaluator,
    registered_problem_components,
)
from .solving import (
    CaseSolveRecord,
    solve_native_problem_task,
    solve_problem_task,
)
from .transforms import (
    cbqm_to_factor_graph,
    factor_graph_to_cbqm,
    interaction_graph_to_qubo,
    qubo_to_interaction_graph,
    qubo_to_maxcut_graph,
)
from .updater import (
    BestKnownUpdate,
    evaluate_task_solution,
    update_best_known,
    update_best_known_file,
)
from .validation import (
    ProblemValidationError,
    ProblemValidationReport,
    validate_problem_case,
    validate_problem_path,
    validate_problem_set,
)

__all__ = [
    'PROBLEM_CASE_SCHEMA',
    'BestKnownSolution',
    'BestKnownUpdate',
    'CaseSolveRecord',
    'NativeSolverRunner',
    'ProblemArtifact',
    'ProblemCase',
    'ProblemSet',
    'ProblemValidationError',
    'ProblemValidationReport',
    'TaskDefinition',
    'TransformationRecord',
    'artifact_from_networkx',
    'artifact_to_networkx',
    'case_from_dict',
    'case_to_dict',
    'cbqm_to_factor_graph',
    'compile_case_qubo',
    'create_graph_case',
    'derive_cbqm_factor_graph',
    'derive_qubo_interaction_graph',
    'derive_qubo_maxcut_graph',
    'evaluate_task_solution',
    'factor_graph_to_cbqm',
    'graph_from_node_link',
    'graph_to_node_link',
    'interaction_graph_to_qubo',
    'load_problem',
    'load_problem_case',
    'load_problem_set',
    'problem_to_dict',
    'project_qubo_sample_to_cbqm',
    'qubo_to_interaction_graph',
    'qubo_to_maxcut_graph',
    'register_native_solver_runner',
    'register_representation_validator',
    'register_task_evaluator',
    'registered_problem_components',
    'save_problem',
    'save_problem_case',
    'solve_native_problem_task',
    'solve_problem_task',
    'update_best_known',
    'update_best_known_file',
    'validate_problem_case',
    'validate_problem_path',
    'validate_problem_set',
]
