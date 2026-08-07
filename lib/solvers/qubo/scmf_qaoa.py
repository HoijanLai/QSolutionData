"""Self-consistent mean-field QAOA for canonical QUBO problems.

This is an independent, ideal-statevector reproduction of Dupont, Sundar and
Gowrishankar, *Self-consistent mean-field quantum approximate optimization*,
arXiv:2603.09838v1 (2026).

The public solver deliberately exposes only the usual ``solve`` entry point.
The paper-specific mechanics--QUBO/Ising conversion, symmetry breaking,
balanced partitioning, asynchronous mean-field updates, shared-angle QAOA,
Nelder--Mead search and product-state sampling--live behind protected methods
so the main workflow reads like the algorithm's pseudocode.

The implementation is a reference reproduction, not a hardware claim.  It
computes every subproblem state exactly with NumPy and therefore scales with
the largest subproblem, not with the full problem, but remains exponential in
that subproblem size.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ...contracts.validation import _evaluate_qubo
from .base import BaseQuboSolver, QuboSolveOutcome
from .qaoa import QaoaQuboSolver


_PAPER_TITLE = 'Self-consistent mean-field quantum approximate optimization'
_PAPER_REVISION = 'arXiv:2603.09838v1'


@dataclass(frozen=True)
class _IsingModel:
    """Binary64 Ising form plus the map back to canonical QUBO variables."""

    constant: float
    fields: np.ndarray
    couplings: np.ndarray
    original_indices: tuple[int, ...]
    fixed_spins: tuple[tuple[int, int], ...]
    source_variable_count: int


@dataclass(frozen=True)
class _SubproblemState:
    """One QAOA product factor and the observables needed by SCMF."""

    variables: tuple[int, ...]
    probabilities: np.ndarray
    energies: np.ndarray
    one_body: np.ndarray
    correlations: np.ndarray


@dataclass(frozen=True)
class _ScmfEvaluation:
    """Result of converging the environment for one shared angle vector."""

    expectation: float
    environment: np.ndarray
    one_body: np.ndarray
    subproblem_states: tuple[_SubproblemState, ...]
    sweeps: int
    converged: bool
    max_environment_delta: float
    energy_delta: float | None


@dataclass(frozen=True)
class _ParameterOptimisation:
    """Outer variational search result including its nested SCMF work."""

    parameters: np.ndarray
    evaluation: _ScmfEvaluation
    evaluations: int
    iterations: int
    converged: bool
    total_environment_sweeps: int


@dataclass(frozen=True)
class _SamplingResult:
    """Best stitched product-state observation and sampling diagnostics."""

    sample: list[int]
    selected_probability: float
    unique_samples: int | None
    selected_occurrences: int | None


class ScmfQaoaSolver(QaoaQuboSolver):
    """Solve ``qubo.v1`` through self-consistent mean-field QAOA.

    ``subproblem_count`` controls the qubit/resource reduction.  All
    subproblems share the same ``2 * layers`` QAOA angles.  For each trial
    angle vector, the solver repeatedly updates subproblem spin expectations
    until both the environment and global product-state energy stabilize.

    Configuration fields:

    ``layers``
        QAOA depth shared by every subproblem.  Defaults to ``1``.
    ``subproblem_count``
        Number of balanced random partitions.  The default is two, reduced to
        one when fewer than two active variables remain.
    ``optimizer_iterations``
        Maximum protected Nelder--Mead iterations.  Defaults to ``20``.
    ``optimizer_tolerance``
        Joint energy/simplex convergence tolerance.  Defaults to ``1e-6``.
    ``simplex_step``
        Initial angle displacement for Nelder--Mead.  Defaults to ``0.2``.
    ``max_environment_sweeps``
        Maximum random-order asynchronous sweeps per objective evaluation.
        Defaults to ``60``.  Zero intentionally creates the environmentless
        independent-subproblem baseline used by the paper.
    ``environment_tolerance`` / ``energy_tolerance``
        Self-consistency thresholds.  Defaults to ``1e-6``.
    ``environment_scale``
        Paper's optional environment multiplier ``eta``.  Defaults to ``1``.
    ``shots``
        Number of stitched product-state samples.  Defaults to ``1024``;
        ``None`` chooses each factor's most likely basis state.
    ``seed`` / ``partition_seed``
        Independent reproducibility controls for algorithm sampling and graph
        partitioning.  ``partition_seed`` defaults to ``seed``.
    ``max_subproblem_variables``
        Statevector safety guard applied per partition.  Defaults to ``16``.
    ``symmetry_breaking``
        ``'auto'`` fixes the final spin to ``+1`` only for a zero-field Ising
        model, as prescribed by the paper.  Explicit alternatives are
        ``'none'``, ``'fix-last-positive'`` and ``'fix-last-negative'``.
    ``initial_parameters``
        Optional interleaved ``[gamma_1, beta_1, ...]`` angle vector.
    """

    SOLVER_NAME = 'scmf-qaoa-statevector'
    SOLVER_VERSION = '0.1.0'
    BACKEND = 'numpy-subproblem-statevector'

    DEFAULT_LAYERS = 1
    DEFAULT_SUBPROBLEM_COUNT = None
    DEFAULT_OPTIMIZER_ITERATIONS = 20
    DEFAULT_OPTIMIZER_TOLERANCE = 1e-6
    DEFAULT_SIMPLEX_STEP = 0.2
    DEFAULT_MAX_ENVIRONMENT_SWEEPS = 60
    DEFAULT_ENVIRONMENT_TOLERANCE = 1e-6
    DEFAULT_ENERGY_TOLERANCE = 1e-6
    DEFAULT_ENVIRONMENT_SCALE = 1.0
    DEFAULT_SHOTS = 1024
    DEFAULT_SEED = 0
    DEFAULT_MAX_SUBPROBLEM_VARIABLES = 16
    DEFAULT_SYMMETRY_BREAKING = 'auto'

    _SYMMETRY_BREAKING_MODES = frozenset(
        {
            'auto',
            'none',
            'fix-last-positive',
            'fix-last-negative',
        }
    )
    _ALLOWED_CONFIG_FIELDS = frozenset(
        {
            'layers',
            'subproblem_count',
            'optimizer_iterations',
            'optimizer_tolerance',
            'simplex_step',
            'max_environment_sweeps',
            'environment_tolerance',
            'energy_tolerance',
            'environment_scale',
            'shots',
            'seed',
            'partition_seed',
            'max_subproblem_variables',
            'symmetry_breaking',
            'initial_parameters',
        }
    )

    def _resolve_config(self, config):
        """Validate SCMF-owned options without invoking QAOA's field list."""
        resolved = BaseQuboSolver._resolve_config(self, config)
        self._reject_unknown_scmf_config_fields(resolved)

        seed = resolved.get('seed', self.DEFAULT_SEED)
        settings = {
            'layers': resolved.get('layers', self.DEFAULT_LAYERS),
            'subproblem_count': resolved.get(
                'subproblem_count',
                self.DEFAULT_SUBPROBLEM_COUNT,
            ),
            'optimizer_iterations': resolved.get(
                'optimizer_iterations',
                self.DEFAULT_OPTIMIZER_ITERATIONS,
            ),
            'optimizer_tolerance': resolved.get(
                'optimizer_tolerance',
                self.DEFAULT_OPTIMIZER_TOLERANCE,
            ),
            'simplex_step': resolved.get(
                'simplex_step',
                self.DEFAULT_SIMPLEX_STEP,
            ),
            'max_environment_sweeps': resolved.get(
                'max_environment_sweeps',
                self.DEFAULT_MAX_ENVIRONMENT_SWEEPS,
            ),
            'environment_tolerance': resolved.get(
                'environment_tolerance',
                self.DEFAULT_ENVIRONMENT_TOLERANCE,
            ),
            'energy_tolerance': resolved.get(
                'energy_tolerance',
                self.DEFAULT_ENERGY_TOLERANCE,
            ),
            'environment_scale': resolved.get(
                'environment_scale',
                self.DEFAULT_ENVIRONMENT_SCALE,
            ),
            'shots': resolved.get('shots', self.DEFAULT_SHOTS),
            'seed': seed,
            'partition_seed': resolved.get('partition_seed', seed),
            'max_subproblem_variables': resolved.get(
                'max_subproblem_variables',
                self.DEFAULT_MAX_SUBPROBLEM_VARIABLES,
            ),
            'symmetry_breaking': resolved.get(
                'symmetry_breaking',
                self.DEFAULT_SYMMETRY_BREAKING,
            ),
            'initial_parameters': resolved.get('initial_parameters'),
        }
        self._validate_scmf_settings(settings)
        return settings

    def _run(self, problem, config):
        """Execute the nested paper workflow in pseudocode-level steps."""
        ising_model = self._qubo_to_ising(problem)
        active_model = self._apply_symmetry_breaking(ising_model, config)

        if not active_model.original_indices:
            return self._solve_trivial_model(problem, active_model, config)

        subproblem_count = self._resolve_subproblem_count(
            config['subproblem_count'],
            len(active_model.original_indices),
        )
        partitions = self._build_balanced_partitions(
            len(active_model.original_indices),
            subproblem_count,
            config['partition_seed'],
        )
        self._guard_subproblem_sizes(
            partitions,
            config['max_subproblem_variables'],
        )
        update_schedule = self._build_update_schedule(
            subproblem_count,
            config['max_environment_sweeps'],
            config['seed'],
        )

        optimisation = self._optimise_shared_parameters(
            active_model,
            partitions,
            update_schedule,
            config,
        )
        sampling = self._sample_product_state(
            problem,
            active_model,
            optimisation.evaluation,
            config['shots'],
            config['seed'],
        )

        return self._build_scmf_outcome(
            active_model,
            partitions,
            optimisation,
            sampling,
            config,
        )

    def _reject_unknown_scmf_config_fields(self, config):
        """Reject misspellings before an experiment silently changes meaning."""
        unknown = set(config) - self._ALLOWED_CONFIG_FIELDS
        if unknown:
            names = ', '.join(sorted(unknown))
            raise ValueError(f'Unknown SCMF-QAOA config fields: {names}')

    def _validate_scmf_settings(self, settings):
        """Keep individual validation rules outside the algorithm workflow."""
        self._validate_positive_integer(settings['layers'], 'layers')
        self._validate_optional_positive_integer(
            settings['subproblem_count'],
            'subproblem_count',
        )
        self._validate_non_negative_integer(
            settings['optimizer_iterations'],
            'optimizer_iterations',
        )
        self._validate_positive_finite_number(
            settings['optimizer_tolerance'],
            'optimizer_tolerance',
        )
        self._validate_positive_finite_number(
            settings['simplex_step'],
            'simplex_step',
        )
        self._validate_non_negative_integer(
            settings['max_environment_sweeps'],
            'max_environment_sweeps',
        )
        self._validate_positive_finite_number(
            settings['environment_tolerance'],
            'environment_tolerance',
        )
        self._validate_positive_finite_number(
            settings['energy_tolerance'],
            'energy_tolerance',
        )
        self._validate_positive_finite_number(
            settings['environment_scale'],
            'environment_scale',
        )
        self._validate_optional_positive_integer(settings['shots'], 'shots')
        self._validate_non_negative_integer(settings['seed'], 'seed')
        self._validate_non_negative_integer(
            settings['partition_seed'],
            'partition_seed',
        )
        self._validate_positive_integer(
            settings['max_subproblem_variables'],
            'max_subproblem_variables',
        )
        if settings['symmetry_breaking'] not in self._SYMMETRY_BREAKING_MODES:
            raise ValueError(
                'symmetry_breaking must be one of: '
                + ', '.join(sorted(self._SYMMETRY_BREAKING_MODES))
            )
        self._validate_initial_parameters(
            settings['initial_parameters'],
            settings['layers'],
        )

    def _validate_positive_finite_number(self, value, name):
        """Validate a real-valued option that must be finite and positive."""
        if not self._is_finite_number(value) or value <= 0:
            raise ValueError(f'{name} must be a finite positive number.')

    def _qubo_to_ising(self, problem):
        """Convert canonical QUBO to the paper's Pauli-Z convention.

        With ``z_i = 1 - 2*x_i``, a QUBO interaction becomes

        ``q_ij*x_i*x_j = q_ij/4 * (1 - z_i - z_j + z_i*z_j)``.
        """
        variable_count = problem['num_variables']
        try:
            constant = float(problem['offset'])
            fields = np.zeros(variable_count, dtype=float)
            couplings = np.zeros(
                (variable_count, variable_count),
                dtype=float,
            )
            for left, right, coefficient in problem['terms']:
                weight = float(coefficient)
                if left == right:
                    constant += weight / 2.0
                    fields[left] -= weight / 2.0
                    continue

                quarter = weight / 4.0
                constant += quarter
                fields[left] -= quarter
                fields[right] -= quarter
                couplings[left, right] += quarter
                couplings[right, left] += quarter
        except (OverflowError, ValueError) as error:
            raise ValueError(
                'SCMF-QAOA requires QUBO coefficients to fit binary64.'
            ) from error

        if not (
            math.isfinite(constant)
            and np.all(np.isfinite(fields))
            and np.all(np.isfinite(couplings))
        ):
            raise ValueError(
                'SCMF-QAOA requires QUBO coefficients to fit binary64.'
            )
        return _IsingModel(
            constant=constant,
            fields=fields,
            couplings=couplings,
            original_indices=tuple(range(variable_count)),
            fixed_spins=(),
            source_variable_count=variable_count,
        )

    def _apply_symmetry_breaking(self, model, config):
        """Fix one spin when requested or when zero fields imply Z2 symmetry."""
        if not model.original_indices:
            return model

        mode = config['symmetry_breaking']
        should_fix = mode.startswith('fix-last')
        if mode == 'auto':
            should_fix = (
                len(model.original_indices) > 1
                and self._has_zero_fields(model)
            )
        if not should_fix:
            return model

        fixed_spin = -1 if mode == 'fix-last-negative' else 1
        return self._fix_last_spin(model, fixed_spin)

    def _has_zero_fields(self, model):
        """Recognise numerical zero after a reversible QUBO/Ising conversion."""
        tolerance = (
            16.0
            * np.finfo(float).eps
            * max(1, len(model.original_indices))
        )
        return bool(np.all(np.abs(model.fields) <= tolerance))

    def _fix_last_spin(self, model, fixed_spin):
        """Condition the Ising model and retain the restoration witness."""
        fixed_position = len(model.original_indices) - 1
        fixed_original_index = model.original_indices[fixed_position]
        active_positions = tuple(range(fixed_position))

        constant = (
            model.constant
            + model.fields[fixed_position] * fixed_spin
        )
        fields = (
            model.fields[list(active_positions)]
            + model.couplings[list(active_positions), fixed_position]
            * fixed_spin
        )
        couplings = model.couplings[
            np.ix_(active_positions, active_positions)
        ]
        return _IsingModel(
            constant=float(constant),
            fields=np.asarray(fields, dtype=float),
            couplings=np.asarray(couplings, dtype=float),
            original_indices=tuple(
                model.original_indices[position]
                for position in active_positions
            ),
            fixed_spins=(
                *model.fixed_spins,
                (fixed_original_index, fixed_spin),
            ),
            source_variable_count=model.source_variable_count,
        )

    def _resolve_subproblem_count(self, requested_count, active_count):
        """Choose a useful default while treating explicit oversizing as error."""
        if requested_count is None:
            return min(2, active_count)
        if requested_count > active_count:
            raise ValueError(
                'subproblem_count cannot exceed the number of active '
                f'variables ({active_count}).'
            )
        return requested_count

    def _build_balanced_partitions(
        self,
        variable_count,
        subproblem_count,
        partition_seed,
    ):
        """Randomly partition once, then sort each factor for stable encoding."""
        random_generator = np.random.default_rng(
            np.random.SeedSequence([partition_seed, 0])
        )
        permutation = random_generator.permutation(variable_count)
        return tuple(
            tuple(sorted(int(value) for value in chunk))
            for chunk in np.array_split(permutation, subproblem_count)
        )

    def _guard_subproblem_sizes(self, partitions, max_variables):
        """Apply the exponential statevector guard to every factor."""
        largest = max(map(len, partitions), default=0)
        if largest > max_variables:
            raise ValueError(
                f'SCMF-QAOA largest subproblem has {largest} variables, '
                f'exceeding max_subproblem_variables={max_variables}.'
            )

    def _build_update_schedule(self, count, sweeps, seed):
        """Precompute random-order sweeps so every objective is deterministic.

        The paper selects a random subproblem at every inner iteration.  A
        random permutation per sweep preserves asynchronous updates while
        ensuring every factor is visited before convergence is tested.
        """
        random_generator = np.random.default_rng(
            np.random.SeedSequence([seed, 1])
        )
        return tuple(
            tuple(int(value) for value in random_generator.permutation(count))
            for _ in range(sweeps)
        )

    def _optimise_shared_parameters(
        self,
        model,
        partitions,
        update_schedule,
        config,
    ):
        """Minimise the self-consistent global expectation by Nelder--Mead."""
        initial = self._initial_scmf_parameters(model, config)
        maximum_iterations = config['optimizer_iterations']
        cache = {}
        total_environment_sweeps = 0

        def evaluate(parameters):
            nonlocal total_environment_sweeps
            normalised = self._normalise_parameters(
                np.asarray(parameters, dtype=float)
            )
            key = tuple(float(value) for value in normalised)
            if key not in cache:
                evaluation = self._converge_environment(
                    model,
                    partitions,
                    normalised,
                    update_schedule,
                    config,
                )
                cache[key] = evaluation
                total_environment_sweeps += evaluation.sweeps
            return normalised, cache[key]

        if maximum_iterations == 0:
            parameters, evaluation = evaluate(initial)
            return _ParameterOptimisation(
                parameters=parameters,
                evaluation=evaluation,
                evaluations=len(cache),
                iterations=0,
                converged=False,
                total_environment_sweeps=total_environment_sweeps,
            )

        simplex = self._initial_simplex(
            initial,
            config['simplex_step'],
        )
        records = [evaluate(point) for point in simplex]
        converged = False
        iterations = 0

        for iterations in range(1, maximum_iterations + 1):
            records = self._sort_simplex_records(records)
            if self._simplex_has_converged(
                records,
                config['optimizer_tolerance'],
            ):
                converged = True
                iterations -= 1
                break
            records = self._nelder_mead_step(records, evaluate)

        records = self._sort_simplex_records(records)
        best_parameters, best_evaluation = records[0]
        return _ParameterOptimisation(
            parameters=best_parameters,
            evaluation=best_evaluation,
            evaluations=len(cache),
            iterations=iterations,
            converged=converged,
            total_environment_sweeps=total_environment_sweeps,
        )

    def _initial_scmf_parameters(self, model, config):
        """Use supplied angles or the paper's p=1 SK concentration scale."""
        if config['initial_parameters'] is not None:
            return np.asarray(config['initial_parameters'], dtype=float)

        scale = max(1, model.source_variable_count)
        gamma = 0.5 / math.sqrt(scale)
        beta = math.pi / 8.0
        return np.asarray(
            [gamma, beta] * config['layers'],
            dtype=float,
        )

    def _initial_simplex(self, initial, step):
        """Create a deterministic axis-aligned simplex around one angle set."""
        simplex = [self._normalise_parameters(initial.copy())]
        for coordinate in range(len(initial)):
            point = initial.copy()
            point[coordinate] += step
            simplex.append(self._normalise_parameters(point))
        return simplex

    def _sort_simplex_records(self, records):
        """Order vertices by energy and then angles for deterministic ties."""
        return sorted(
            records,
            key=lambda record: (
                record[1].expectation,
                tuple(float(value) for value in record[0]),
            ),
        )

    def _simplex_has_converged(self, records, tolerance):
        """Require both objective values and angle positions to contract."""
        best_parameters = records[0][0]
        energy_spread = max(
            abs(record[1].expectation - records[0][1].expectation)
            for record in records
        )
        parameter_spread = max(
            float(np.max(np.abs(record[0] - best_parameters)))
            for record in records
        )
        return energy_spread <= tolerance and parameter_spread <= tolerance

    def _nelder_mead_step(self, records, evaluate):
        """Perform one standard reflection/expansion/contraction iteration."""
        dimension = len(records) - 1
        best = records[0]
        worst = records[-1]
        second_worst = records[-2]
        centroid = np.mean(
            [record[0] for record in records[:-1]],
            axis=0,
        )

        reflected = evaluate(centroid + (centroid - worst[0]))
        if reflected[1].expectation < best[1].expectation:
            expanded = evaluate(
                centroid + 2.0 * (reflected[0] - centroid)
            )
            replacement = min(
                (reflected, expanded),
                key=lambda record: record[1].expectation,
            )
            return [*records[:-1], replacement]

        if reflected[1].expectation < second_worst[1].expectation:
            return [*records[:-1], reflected]

        if reflected[1].expectation < worst[1].expectation:
            contracted_point = centroid + 0.5 * (
                reflected[0] - centroid
            )
            contraction_limit = reflected[1].expectation
        else:
            contracted_point = centroid + 0.5 * (
                worst[0] - centroid
            )
            contraction_limit = worst[1].expectation

        contracted = evaluate(contracted_point)
        if contracted[1].expectation <= contraction_limit:
            return [*records[:-1], contracted]

        shrunk = [best]
        for record in records[1:]:
            shrunk.append(
                evaluate(best[0] + 0.5 * (record[0] - best[0]))
            )
        if len(shrunk) != dimension + 1:
            raise RuntimeError('Nelder--Mead simplex lost a vertex.')
        return shrunk

    def _converge_environment(
        self,
        model,
        partitions,
        parameters,
        update_schedule,
        config,
    ):
        """Run the inner asynchronous fixed-point iteration for one angle set."""
        environment = np.zeros(len(model.original_indices), dtype=float)
        states = self._solve_all_subproblems(
            model,
            partitions,
            environment,
            parameters,
        )
        expectation, one_body = self._global_expectation(model, states)

        if not update_schedule:
            return _ScmfEvaluation(
                expectation=expectation,
                environment=environment,
                one_body=one_body,
                subproblem_states=states,
                sweeps=0,
                converged=False,
                max_environment_delta=0.0,
                energy_delta=None,
            )

        previous_energy = None
        max_delta = math.inf
        energy_delta = None
        converged = False
        sweeps = 0

        for sweeps, update_order in enumerate(update_schedule, start=1):
            previous_environment = environment.copy()
            for partition_index in update_order:
                state = self._solve_subproblem(
                    model,
                    partitions[partition_index],
                    environment,
                    parameters,
                )
                environment[list(state.variables)] = (
                    config['environment_scale'] * state.one_body
                )

            states = self._solve_all_subproblems(
                model,
                partitions,
                environment,
                parameters,
            )
            expectation, one_body = self._global_expectation(model, states)
            max_delta = float(
                np.max(np.abs(environment - previous_environment))
            )
            if previous_energy is not None:
                energy_delta = abs(expectation - previous_energy)
                if (
                    max_delta <= config['environment_tolerance']
                    and self._energy_has_converged(
                        expectation,
                        previous_energy,
                        config['energy_tolerance'],
                    )
                ):
                    converged = True
                    break
            previous_energy = expectation

        return _ScmfEvaluation(
            expectation=expectation,
            environment=environment,
            one_body=one_body,
            subproblem_states=states,
            sweeps=sweeps,
            converged=converged,
            max_environment_delta=max_delta,
            energy_delta=energy_delta,
        )

    def _energy_has_converged(self, current, previous, tolerance):
        """Use a zero-safe relative threshold for the paper's energy test."""
        scale = max(1.0, abs(current), abs(previous))
        return abs(current - previous) <= tolerance * scale

    def _solve_all_subproblems(
        self,
        model,
        partitions,
        environment,
        parameters,
    ):
        """Evaluate every independent product factor at one environment."""
        return tuple(
            self._solve_subproblem(
                model,
                partition,
                environment,
                parameters,
            )
            for partition in partitions
        )

    def _solve_subproblem(
        self,
        model,
        variables,
        environment,
        parameters,
    ):
        """Build the effective field and run exact statevector QAOA locally."""
        variable_array = np.asarray(variables, dtype=int)
        outside = np.asarray(
            [
                index
                for index in range(len(model.original_indices))
                if index not in variables
            ],
            dtype=int,
        )
        effective_fields = model.fields[variable_array].copy()
        if outside.size:
            effective_fields += (
                model.couplings[np.ix_(variable_array, outside)]
                @ environment[outside]
            )

        spins = self._basis_spins(len(variables))
        local_couplings = model.couplings[
            np.ix_(variable_array, variable_array)
        ]
        energies = (
            spins @ effective_fields
            + 0.5
            * np.sum((spins @ local_couplings) * spins, axis=1)
        )
        statevector = self._prepare_qaoa_state(
            energies,
            len(variables),
            self._statevector_parameters(parameters),
        )
        probabilities = self._measurement_probabilities(statevector)
        one_body = probabilities @ spins
        correlations = (spins.T * probabilities) @ spins

        return _SubproblemState(
            variables=tuple(variables),
            probabilities=np.asarray(probabilities, dtype=float),
            energies=np.asarray(energies, dtype=float),
            one_body=np.asarray(one_body, dtype=float),
            correlations=np.asarray(correlations, dtype=float),
        )

    def _statevector_parameters(self, paper_parameters):
        """Translate the paper's gamma sign into the shared QAOA kernel.

        Equation (8) of arXiv:2603.09838v1 gives, at ``p=1``,

        ``<Z_i> = -sin(2 beta) sin(2 gamma h_i) product_j cos(2 gamma W_ij)``.

        The repository's earlier QAOA statevector exposes the opposite cost
        angle orientation for the same diagonal spectrum.  Negating only the
        gamma entries makes SCMF's public angles follow the paper equation
        without changing the already-published generic QAOA convention.
        """
        translated = np.asarray(paper_parameters, dtype=float).copy()
        translated[0::2] *= -1.0
        return translated

    def _basis_spins(self, variable_count):
        """Return Pauli-Z eigenvalues in canonical most-significant-bit order."""
        dimension = 1 << variable_count
        basis_indices = np.arange(dimension, dtype=np.uint64)[:, None]
        shifts = np.arange(
            variable_count - 1,
            -1,
            -1,
            dtype=np.uint64,
        )
        bits = (basis_indices >> shifts) & 1
        return 1.0 - 2.0 * bits.astype(float)

    def _global_expectation(self, model, states):
        """Evaluate the original Ising Hamiltonian on the product state.

        Intra-partition ``<Z_i Z_j>`` values remain exact within each QAOA
        factor.  Only cross-partition correlations are factorized into
        ``<Z_i><Z_j>``, which is the defining mean-field approximation.
        """
        one_body = np.zeros(len(model.original_indices), dtype=float)
        for state in states:
            one_body[list(state.variables)] = state.one_body

        interaction = 0.5 * float(
            one_body @ model.couplings @ one_body
        )
        for state in states:
            variables = state.variables
            for local_left, left in enumerate(variables):
                for local_right in range(local_left + 1, len(variables)):
                    right = variables[local_right]
                    interaction += model.couplings[left, right] * (
                        state.correlations[local_left, local_right]
                        - one_body[left] * one_body[right]
                    )

        expectation = (
            model.constant
            + float(model.fields @ one_body)
            + interaction
        )
        if not math.isfinite(expectation):
            raise ValueError('SCMF-QAOA produced a non-finite expectation.')
        return float(expectation), one_body

    def _sample_product_state(
        self,
        problem,
        model,
        evaluation,
        shots,
        seed,
    ):
        """Independently sample every factor, stitch, then score globally."""
        random_generator = np.random.default_rng(
            np.random.SeedSequence([seed, 2])
        )
        observation_count = 1 if shots is None else shots
        local_observations = []

        for state in evaluation.subproblem_states:
            if shots is None:
                observation = self._most_likely_basis_index(
                    state.probabilities,
                    state.energies,
                )
                local_observations.append(
                    np.asarray([observation], dtype=int)
                )
            else:
                local_observations.append(
                    random_generator.choice(
                        len(state.probabilities),
                        size=shots,
                        p=state.probabilities,
                    )
                )

        best_sample = None
        best_energy = None
        best_probability = 0.0
        occurrences = {}

        for observation_index in range(observation_count):
            active_bits = [0] * len(model.original_indices)
            probability = 1.0
            for state, observations in zip(
                evaluation.subproblem_states,
                local_observations,
            ):
                basis_index = int(observations[observation_index])
                local_bits = self._decode_basis_index(
                    basis_index,
                    len(state.variables),
                )
                probability *= float(state.probabilities[basis_index])
                for local_index, variable in enumerate(state.variables):
                    active_bits[variable] = local_bits[local_index]

            sample = self._restore_canonical_sample(active_bits, model)
            key = tuple(sample)
            occurrences[key] = occurrences.get(key, 0) + 1
            energy = _evaluate_qubo(problem, sample)
            if (
                best_energy is None
                or energy < best_energy
                or (energy == best_energy and sample < best_sample)
            ):
                best_sample = sample
                best_energy = energy
                best_probability = probability

        return _SamplingResult(
            sample=best_sample,
            selected_probability=float(best_probability),
            unique_samples=(None if shots is None else len(occurrences)),
            selected_occurrences=(
                None
                if shots is None
                else occurrences[tuple(best_sample)]
            ),
        )

    def _restore_canonical_sample(self, active_bits, model):
        """Insert conditioned spins and recover the source QUBO bit order."""
        sample = [0] * model.source_variable_count
        for active_position, original_index in enumerate(
            model.original_indices
        ):
            sample[original_index] = int(active_bits[active_position])
        for original_index, spin in model.fixed_spins:
            sample[original_index] = 0 if spin == 1 else 1
        return sample

    def _solve_trivial_model(self, problem, model, config):
        """Return the sole conditioned assignment without allocating a state."""
        sample = self._restore_canonical_sample([], model)
        return QuboSolveOutcome(
            status='feasible',
            best_sample=sample,
            termination_reason='trivial_conditioned_problem',
            metrics={
                'expectation': float(_evaluate_qubo(problem, sample)),
                'subproblem_count': 0,
                'active_variables': 0,
                'environment_converged': True,
            },
            metadata={
                'algorithm': 'self_consistent_mean_field_qaoa',
                'paper_revision': _PAPER_REVISION,
                'layers': config['layers'],
                'seed': config['seed'],
                'fixed_spins': [list(item) for item in model.fixed_spins],
                'partitions': [],
                'parameters': [],
            },
        )

    def _build_scmf_outcome(
        self,
        model,
        partitions,
        optimisation,
        sampling,
        config,
    ):
        """Translate paper diagnostics into a canonical feasible outcome."""
        evaluation = optimisation.evaluation
        metrics = {
            'expectation': evaluation.expectation,
            'optimizer_evaluations': optimisation.evaluations,
            'optimizer_iterations': optimisation.iterations,
            'optimizer_converged': optimisation.converged,
            'environment_sweeps': evaluation.sweeps,
            'total_environment_sweeps': (
                optimisation.total_environment_sweeps
            ),
            'environment_updates': evaluation.sweeps * len(partitions),
            'environment_converged': evaluation.converged,
            'environment_max_delta': evaluation.max_environment_delta,
            'subproblem_count': len(partitions),
            'largest_subproblem_variables': max(map(len, partitions)),
            'active_variables': len(model.original_indices),
            'selected_probability': sampling.selected_probability,
        }
        if evaluation.energy_delta is not None:
            metrics['environment_energy_delta'] = evaluation.energy_delta
        if sampling.unique_samples is not None:
            metrics['unique_samples_observed'] = sampling.unique_samples
            metrics['selected_occurrences'] = sampling.selected_occurrences

        if config['max_environment_sweeps'] == 0:
            termination_reason = 'independent_subproblems_completed'
        elif evaluation.converged:
            termination_reason = 'self_consistent_environment_reached'
        else:
            termination_reason = 'environment_iteration_limit'

        return QuboSolveOutcome(
            status='feasible',
            best_sample=sampling.sample,
            termination_reason=termination_reason,
            metrics=metrics,
            metadata={
                'algorithm': 'self_consistent_mean_field_qaoa',
                'paper_title': _PAPER_TITLE,
                'paper_revision': _PAPER_REVISION,
                'layers': config['layers'],
                'shots': config['shots'],
                'seed': config['seed'],
                'partition_seed': config['partition_seed'],
                'symmetry_breaking': config['symmetry_breaking'],
                'environment_scale': config['environment_scale'],
                'parameters': optimisation.parameters.tolist(),
                'partitions': [
                    [
                        model.original_indices[position]
                        for position in partition
                    ]
                    for partition in partitions
                ],
                'fixed_spins': [
                    list(item)
                    for item in model.fixed_spins
                ],
                'environment': evaluation.environment.tolist(),
                'spin_expectations': evaluation.one_body.tolist(),
                'update_policy': 'seeded-random-order-asynchronous-sweeps',
            },
        )


__all__ = ['ScmfQaoaSolver']
