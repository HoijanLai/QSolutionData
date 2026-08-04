"""A small, dependency-light QAOA solver backed by a NumPy statevector.

This module is intended as a transparent reference implementation.  It keeps
the quantum state, QUBO phase separator, mixer, classical angle search and
measurement policy visible in Python instead of hiding them behind a provider
SDK.  That makes it useful for reproducing QAOA experiments and testing the
canonical solver contract, but not for simulating large quantum systems.
"""

import math

import numpy as np

from .base import BaseQuboSolver, QuboSolveOutcome


class QaoaQuboSolver(BaseQuboSolver):
    """Approximately solve a QUBO with statevector QAOA.

    The classical optimiser is deterministic coordinate search.  It is modest
    by design: the important first milestone is a fully inspectable QAOA path,
    after which SciPy, Qiskit Runtime or hardware optimisers can be introduced
    behind the same protected method boundary.

    Supported configuration fields:

    ``layers``
        Number of alternating cost and mixer layers. Defaults to ``1``.
    ``optimizer_iterations``
        Coordinate-search sweeps per restart. Defaults to ``30``.
    ``restarts``
        Independent parameter initialisations. Defaults to ``4``.
    ``shots``
        Number of simulated measurements. Defaults to ``1024``. Use ``None``
        to select directly from exact statevector probabilities.
    ``seed``
        Random seed used by parameter initialisation and measurement.
    ``max_variables``
        Statevector safety limit. Defaults to ``16``.
    ``initial_parameters``
        Optional flat ``[gamma_1, beta_1, ...]`` starting point. It replaces
        the first random start and must contain ``2 * layers`` finite numbers.
    """

    SOLVER_NAME = 'qaoa-statevector'
    SOLVER_VERSION = '0.1.0'
    BACKEND = 'numpy-statevector'

    DEFAULT_LAYERS = 1
    DEFAULT_OPTIMIZER_ITERATIONS = 30
    DEFAULT_RESTARTS = 4
    DEFAULT_SHOTS = 1024
    DEFAULT_SEED = 0
    DEFAULT_MAX_VARIABLES = 16
    _ALLOWED_CONFIG_FIELDS = frozenset(
        {
            'layers',
            'optimizer_iterations',
            'restarts',
            'shots',
            'seed',
            'max_variables',
            'initial_parameters',
        }
    )

    def _resolve_config(self, config):
        """Validate and normalise all QAOA-owned options."""
        resolved = super()._resolve_config(config)
        self._reject_unknown_config_fields(resolved)

        settings = {
            'layers': resolved.get('layers', self.DEFAULT_LAYERS),
            'optimizer_iterations': resolved.get(
                'optimizer_iterations',
                self.DEFAULT_OPTIMIZER_ITERATIONS,
            ),
            'restarts': resolved.get('restarts', self.DEFAULT_RESTARTS),
            'shots': resolved.get('shots', self.DEFAULT_SHOTS),
            'seed': resolved.get('seed', self.DEFAULT_SEED),
            'max_variables': resolved.get(
                'max_variables',
                self.DEFAULT_MAX_VARIABLES,
            ),
            'initial_parameters': resolved.get('initial_parameters'),
        }
        self._validate_settings(settings)
        return settings

    def _run(self, problem, config):
        """Run QAOA; the high-level workflow intentionally resembles pseudocode."""
        variable_count = problem['num_variables']
        self._guard_statevector_size(variable_count, config['max_variables'])

        energies = self._build_cost_spectrum(problem)
        random_generator = self._make_random_generator(config['seed'])

        parameters, expectation, evaluations = self._optimise_parameters(
            energies,
            variable_count,
            config,
            random_generator,
        )
        final_state = self._prepare_qaoa_state(
            energies,
            variable_count,
            parameters,
        )
        best_sample, selected_probability, unique_samples = self._select_sample(
            final_state,
            energies,
            variable_count,
            config['shots'],
            random_generator,
        )

        return self._build_qaoa_outcome(
            best_sample,
            parameters,
            expectation,
            evaluations,
            selected_probability,
            unique_samples,
            config,
        )

    def _reject_unknown_config_fields(self, config):
        """Reject misspellings so an experiment never uses silent defaults."""
        unknown = set(config) - self._ALLOWED_CONFIG_FIELDS
        if unknown:
            names = ', '.join(sorted(unknown))
            raise ValueError(f'Unknown QAOA solver config fields: {names}')

    def _validate_settings(self, settings):
        """Delegate individual validation rules away from the main workflow."""
        self._validate_positive_integer(settings['layers'], 'layers')
        self._validate_non_negative_integer(
            settings['optimizer_iterations'],
            'optimizer_iterations',
        )
        self._validate_positive_integer(settings['restarts'], 'restarts')
        self._validate_optional_positive_integer(settings['shots'], 'shots')
        self._validate_non_negative_integer(settings['seed'], 'seed')
        self._validate_non_negative_integer(
            settings['max_variables'],
            'max_variables',
        )
        self._validate_initial_parameters(
            settings['initial_parameters'],
            settings['layers'],
        )

    def _validate_positive_integer(self, value, name):
        """Validate an integer configuration field that must be above zero."""
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f'{name} must be a positive integer.')

    def _validate_non_negative_integer(self, value, name):
        """Validate an integer configuration field that may be zero."""
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f'{name} must be a non-negative integer.')

    def _validate_optional_positive_integer(self, value, name):
        """Validate a positive integer or the sentinel ``None``."""
        if value is not None:
            self._validate_positive_integer(value, name)

    def _validate_initial_parameters(self, parameters, layers):
        """Validate the optional interleaved gamma/beta starting point."""
        if parameters is None:
            return
        if isinstance(parameters, (str, bytes)) or not isinstance(
            parameters,
            (list, tuple),
        ):
            raise TypeError('initial_parameters must be a list, tuple, or None.')
        if len(parameters) != 2 * layers:
            raise ValueError(
                'initial_parameters must contain exactly 2 * layers values.'
            )
        if any(not self._is_finite_number(value) for value in parameters):
            raise ValueError('initial_parameters must contain finite numbers.')

    def _is_finite_number(self, value):
        """Recognise finite real numbers without treating booleans as angles."""
        return (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(value)
        )

    def _guard_statevector_size(self, variable_count, max_variables):
        """Prevent accidental allocation of an exponential statevector."""
        if variable_count > max_variables:
            raise ValueError(
                f'QAOA statevector received {variable_count} variables, '
                f'exceeding max_variables={max_variables}.'
            )

    def _build_cost_spectrum(self, problem):
        """Evaluate the QUBO once for every computational-basis state."""
        try:
            spectrum = np.asarray(
                [
                    self._evaluate_basis_state(problem, basis_index)
                    for basis_index in range(1 << problem['num_variables'])
                ],
                dtype=float,
            )
        except (OverflowError, ValueError) as error:
            raise ValueError(
                'QAOA requires every basis energy to fit finite binary64; '
                'rescale the QUBO coefficients.'
            ) from error
        if not np.all(np.isfinite(spectrum)):
            raise ValueError(
                'QAOA requires every basis energy to fit finite binary64; '
                'rescale the QUBO coefficients.'
            )
        return spectrum

    def _evaluate_basis_state(self, problem, basis_index):
        """Evaluate one basis state directly from its integer bit pattern."""
        variable_count = problem['num_variables']
        return float(
            math.fsum(
                [
                    float(problem['offset']),
                    *(
                        float(coefficient)
                        * self._read_bit(
                            basis_index,
                            left,
                            variable_count,
                        )
                        * self._read_bit(
                            basis_index,
                            right,
                            variable_count,
                        )
                        for left, right, coefficient in problem['terms']
                    ),
                ]
            )
        )

    def _read_bit(self, basis_index, variable_index, variable_count):
        """Read a variable using the canonical sample's most-significant order."""
        shift = variable_count - variable_index - 1
        return (basis_index >> shift) & 1

    def _make_random_generator(self, seed):
        """Keep all stochastic behaviour under one reproducible generator."""
        return np.random.default_rng(seed)

    def _optimise_parameters(
        self,
        energies,
        variable_count,
        config,
        random_generator,
    ):
        """Find low-expectation angles with multi-start coordinate search."""
        best_parameters = None
        best_expectation = math.inf
        total_evaluations = 0

        for restart in range(config['restarts']):
            parameters = self._initial_parameters(
                restart,
                config,
                random_generator,
            )
            parameters, expectation, evaluations = self._coordinate_search(
                energies,
                variable_count,
                parameters,
                config['optimizer_iterations'],
            )
            total_evaluations += evaluations

            if expectation < best_expectation:
                best_parameters = parameters
                best_expectation = expectation

        return best_parameters, best_expectation, total_evaluations

    def _initial_parameters(self, restart, config, random_generator):
        """Use the supplied first start, then reproducible random restarts."""
        supplied = config['initial_parameters']
        if restart == 0 and supplied is not None:
            return np.asarray(supplied, dtype=float)

        parameters = np.empty(2 * config['layers'], dtype=float)
        parameters[0::2] = random_generator.uniform(
            0.0,
            2.0 * math.pi,
            config['layers'],
        )
        parameters[1::2] = random_generator.uniform(
            0.0,
            math.pi,
            config['layers'],
        )
        return parameters

    def _coordinate_search(
        self,
        energies,
        variable_count,
        initial_parameters,
        iterations,
    ):
        """Refine each angle in both directions with a shrinking step size."""
        parameters = initial_parameters.copy()
        expectation = self._expectation(
            energies,
            variable_count,
            parameters,
        )
        evaluations = 1
        step_size = math.pi / 2.0

        for _ in range(iterations):
            improved = False
            for parameter_index in range(len(parameters)):
                candidate, candidate_expectation, used_evaluations = (
                    self._best_coordinate_move(
                        energies,
                        variable_count,
                        parameters,
                        expectation,
                        parameter_index,
                        step_size,
                    )
                )
                evaluations += used_evaluations
                if candidate_expectation < expectation:
                    parameters = candidate
                    expectation = candidate_expectation
                    improved = True
            if not improved:
                step_size /= 2.0

        return parameters, expectation, evaluations

    def _best_coordinate_move(
        self,
        energies,
        variable_count,
        parameters,
        expectation,
        parameter_index,
        step_size,
    ):
        """Test the positive and negative move for one parameter."""
        best_parameters = parameters
        best_expectation = expectation
        evaluations = 0

        for direction in (-1.0, 1.0):
            candidate = parameters.copy()
            candidate[parameter_index] += direction * step_size
            candidate = self._normalise_parameters(candidate)
            candidate_expectation = self._expectation(
                energies,
                variable_count,
                candidate,
            )
            evaluations += 1
            if candidate_expectation < best_expectation:
                best_parameters = candidate
                best_expectation = candidate_expectation

        return best_parameters, best_expectation, evaluations

    def _normalise_parameters(self, parameters):
        """Wrap only mixer angles into their problem-independent period.

        The X mixer has period ``pi`` up to a global phase, so normalising beta
        is always safe.  A cost angle gamma has a common ``2*pi`` period only
        when all QUBO energy differences live on a compatible integer lattice.
        Canonical ``qubo.v1`` coefficients are arbitrary finite reals, hence
        wrapping gamma unconditionally can change the represented state.
        """
        normalised = parameters.copy()
        normalised[1::2] %= math.pi
        return normalised

    def _expectation(self, energies, variable_count, parameters):
        """Return the expected QUBO energy of one QAOA state."""
        state = self._prepare_qaoa_state(
            energies,
            variable_count,
            parameters,
        )
        probabilities = self._measurement_probabilities(state)
        return float(np.dot(probabilities, energies))

    def _prepare_qaoa_state(self, energies, variable_count, parameters):
        """Create |+>^n and apply each cost/mixer layer in sequence."""
        state = self._uniform_superposition(variable_count)
        for layer in range(len(parameters) // 2):
            gamma = parameters[2 * layer]
            beta = parameters[2 * layer + 1]
            state = self._apply_cost_layer(state, energies, gamma)
            state = self._apply_mixer_layer(state, variable_count, beta)
        return state

    def _uniform_superposition(self, variable_count):
        """Construct the normalised |+> state for every binary variable."""
        dimension = 1 << variable_count
        amplitude = 1.0 / math.sqrt(dimension)
        return np.full(dimension, amplitude, dtype=complex)

    def _apply_cost_layer(self, state, energies, gamma):
        """Apply exp(-i * gamma * C) for diagonal QUBO Hamiltonian C."""
        return state * np.exp(-1j * gamma * energies)

    def _apply_mixer_layer(self, state, variable_count, beta):
        """Apply exp(-i * beta * X) independently to every qubit."""
        mixed = state.copy()
        cosine = math.cos(beta)
        sine = math.sin(beta)

        for variable_index in range(variable_count):
            bit_mask = 1 << (variable_count - variable_index - 1)
            for basis_index in range(len(mixed)):
                if basis_index & bit_mask:
                    continue
                paired_index = basis_index | bit_mask
                zero_amplitude = mixed[basis_index]
                one_amplitude = mixed[paired_index]
                mixed[basis_index] = (
                    cosine * zero_amplitude - 1j * sine * one_amplitude
                )
                mixed[paired_index] = (
                    -1j * sine * zero_amplitude + cosine * one_amplitude
                )

        return mixed

    def _measurement_probabilities(self, state):
        """Convert amplitudes to a normalised real probability vector."""
        probabilities = np.abs(state) ** 2
        return probabilities / probabilities.sum()

    def _select_sample(
        self,
        state,
        energies,
        variable_count,
        shots,
        random_generator,
    ):
        """Select the lowest-energy observed state or exact most-likely state."""
        probabilities = self._measurement_probabilities(state)

        if shots is None:
            basis_index = self._most_likely_basis_index(
                probabilities,
                energies,
            )
            unique_samples = None
        else:
            observations = random_generator.choice(
                len(probabilities),
                size=shots,
                p=probabilities,
            )
            basis_index = self._best_observed_basis_index(
                observations,
                energies,
            )
            unique_samples = int(len(np.unique(observations)))

        sample = self._decode_basis_index(basis_index, variable_count)
        return sample, float(probabilities[basis_index]), unique_samples

    def _most_likely_basis_index(self, probabilities, energies):
        """Resolve probability ties by energy, then canonical basis order."""
        largest_probability = float(np.max(probabilities))
        candidates = np.flatnonzero(
            np.isclose(probabilities, largest_probability)
        )
        return int(
            min(candidates, key=lambda index: (energies[index], index))
        )

    def _best_observed_basis_index(self, observations, energies):
        """Return the lowest-energy measured basis state deterministically."""
        return int(min(observations, key=lambda index: (energies[index], index)))

    def _decode_basis_index(self, basis_index, variable_count):
        """Convert a NumPy/Python basis index to JSON-native binary integers."""
        return [
            int(self._read_bit(int(basis_index), index, variable_count))
            for index in range(variable_count)
        ]

    def _build_qaoa_outcome(
        self,
        best_sample,
        parameters,
        expectation,
        evaluations,
        selected_probability,
        unique_samples,
        config,
    ):
        """Build a canonical approximate result without claiming optimality."""
        metrics = {
            'expectation': expectation,
            'optimizer_evaluations': evaluations,
            'selected_probability': selected_probability,
        }
        if unique_samples is not None:
            metrics['unique_samples_observed'] = unique_samples

        return QuboSolveOutcome(
            status='feasible',
            best_sample=best_sample,
            termination_reason='optimizer_completed',
            metrics=metrics,
            metadata={
                'algorithm': 'qaoa',
                'layers': config['layers'],
                'shots': config['shots'],
                'seed': config['seed'],
                'parameters': parameters.tolist(),
            },
        )
