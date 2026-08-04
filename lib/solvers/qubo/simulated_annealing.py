"""Dependency-light simulated annealing for canonical QUBO problems.

The implementation is a readable NumPy reference rather than an adapter to a
third-party sampler.  Each read performs single-bit Metropolis updates over an
inverse-temperature schedule and retains every improving state it observes.
It deliberately returns only the best candidate through ``qubo-result.v1``;
a future sample-set contract should be introduced separately.
"""

import math
import time
from dataclasses import dataclass
from fractions import Fraction

import numpy as np

from ...contracts.validation import _evaluate_qubo
from .base import BaseQuboSolver, QuboSolveOutcome


@dataclass(frozen=True)
class _VariableNeighborhood:
    """One variable's diagonal bias and incident off-diagonal terms."""

    linear_bias: Fraction
    interactions: tuple[tuple[int, Fraction], ...]


@dataclass
class _ReadState:
    """Mutable state kept local to one annealing read."""

    sample: np.ndarray
    energy_without_offset: Fraction
    best_sample: list[int]
    best_energy_without_offset: Fraction


@dataclass(frozen=True)
class _SweepProgress:
    """Counters and termination state produced by one attempted sweep."""

    flips_proposed: int
    flips_accepted: int
    timed_out: bool


@dataclass(frozen=True)
class _ReadResult:
    """Best observation and progress produced by one independent read."""

    best_sample: list[int]
    best_energy_without_offset: Fraction
    sweeps_completed: int
    flips_proposed: int
    flips_accepted: int
    timed_out: bool
    trace: tuple[dict, ...]


class SimulatedAnnealingQuboSolver(BaseQuboSolver):
    """Approximately minimise ``qubo.v1`` with Metropolis annealing.

    Supported configuration fields:

    ``num_reads`` (default ``32``)
        Number of independent annealing trajectories.
    ``sweeps`` (default ``1000``)
        Number of complete variable-update sweeps in each read.
    ``beta_schedule_type`` (default ``"geometric"``)
        One of ``"linear"``, ``"geometric"`` or ``"custom"``.
    ``beta_start`` / ``beta_end`` (default ``None``)
        Optional non-negative inverse-temperature endpoints. Missing endpoints
        are estimated deterministically from the QUBO coefficients.
    ``beta_schedule`` (default ``None``)
        A finite, non-negative, non-decreasing list used only for ``"custom"``.
        Its length must equal ``sweeps``.
    ``seed`` (default ``0``)
        Non-negative seed controlling initial states, update order and moves.
    ``initial_samples`` (default ``None``)
        Optional non-empty list of binary samples. They seed reads in order and
        are cycled when fewer samples than ``num_reads`` are supplied.
    ``update_order`` (default ``"random"``)
        Either a new random permutation per sweep or ``"sequential"`` order.
    ``timeout_seconds`` (default ``None``)
        Optional positive wall-clock deadline for the complete solve.
    ``trace_interval`` (default ``None``)
        Record a canonical incumbent after each configured number of sweeps.

    Simulated annealing is heuristic. Even when it happens to observe a global
    optimum, normal completion is reported as ``feasible``, never ``optimal``.
    """

    SOLVER_NAME = 'simulated-annealing'
    SOLVER_VERSION = '0.1.0'
    BACKEND = 'numpy-local'

    DEFAULT_NUM_READS = 32
    DEFAULT_SWEEPS = 1000
    DEFAULT_BETA_SCHEDULE_TYPE = 'geometric'
    DEFAULT_SEED = 0
    DEFAULT_UPDATE_ORDER = 'random'

    _SCHEDULE_TYPES = frozenset({'linear', 'geometric', 'custom'})
    _UPDATE_ORDERS = frozenset({'random', 'sequential'})
    _ALLOWED_CONFIG_FIELDS = frozenset(
        {
            'num_reads',
            'sweeps',
            'beta_schedule_type',
            'beta_start',
            'beta_end',
            'beta_schedule',
            'seed',
            'initial_samples',
            'update_order',
            'timeout_seconds',
            'trace_interval',
        }
    )

    def _resolve_config(self, config):
        """Copy, validate and normalise every solver-owned option."""
        resolved = super()._resolve_config(config)
        self._reject_unknown_config_fields(resolved)

        settings = {
            'num_reads': resolved.get('num_reads', self.DEFAULT_NUM_READS),
            'sweeps': resolved.get('sweeps', self.DEFAULT_SWEEPS),
            'beta_schedule_type': resolved.get(
                'beta_schedule_type',
                self.DEFAULT_BETA_SCHEDULE_TYPE,
            ),
            'beta_start': resolved.get('beta_start'),
            'beta_end': resolved.get('beta_end'),
            'beta_schedule': resolved.get('beta_schedule'),
            'seed': resolved.get('seed', self.DEFAULT_SEED),
            'initial_samples': resolved.get('initial_samples'),
            'update_order': resolved.get(
                'update_order',
                self.DEFAULT_UPDATE_ORDER,
            ),
            'timeout_seconds': resolved.get('timeout_seconds'),
            'trace_interval': resolved.get('trace_interval'),
        }
        self._validate_settings(settings)
        return settings

    def _run(self, problem, config):
        """Prepare, anneal, collect and report; details stay behind helpers."""
        started_at = time.perf_counter()
        deadline = self._make_deadline(
            started_at,
            config['timeout_seconds'],
        )
        beta_schedule = self._resolve_beta_schedule(problem, config)
        adjacency = self._build_adjacency(problem)
        random_generator = self._make_random_generator(config['seed'])
        initial_samples = self._prepare_initial_samples(
            problem['num_variables'],
            config,
            random_generator,
        )

        incumbent_sample = None
        incumbent_energy = None
        progress = self._empty_progress(config, beta_schedule)
        trace = []

        for read_index, initial_sample in enumerate(initial_samples):
            read_result = self._run_read(
                problem,
                initial_sample,
                beta_schedule,
                adjacency,
                config,
                random_generator,
                deadline,
                started_at,
                read_index,
                progress['sweeps_completed'],
            )
            incumbent_sample, incumbent_energy = self._update_incumbent(
                read_result.best_sample,
                read_result.best_energy_without_offset,
                incumbent_sample,
                incumbent_energy,
            )
            self._accumulate_progress(progress, read_result)
            trace.extend(read_result.trace)

            if read_result.timed_out:
                return self._build_annealing_outcome(
                    'timeout',
                    incumbent_sample,
                    progress,
                    trace,
                    beta_schedule,
                    config,
                )

        return self._build_annealing_outcome(
            'feasible',
            incumbent_sample,
            progress,
            trace,
            beta_schedule,
            config,
        )

    def _reject_unknown_config_fields(self, config):
        """Reject misspellings instead of silently changing an experiment."""
        unknown = set(config) - self._ALLOWED_CONFIG_FIELDS
        if unknown:
            names = ', '.join(sorted(unknown))
            raise ValueError(
                f'Unknown simulated annealing config fields: {names}'
            )

    def _validate_settings(self, settings):
        """Keep individual configuration rules out of the algorithm flow."""
        self._validate_positive_integer(settings['num_reads'], 'num_reads')
        self._validate_positive_integer(settings['sweeps'], 'sweeps')
        self._validate_non_negative_integer(settings['seed'], 'seed')
        self._validate_choice(
            settings['beta_schedule_type'],
            self._SCHEDULE_TYPES,
            'beta_schedule_type',
        )
        self._validate_choice(
            settings['update_order'],
            self._UPDATE_ORDERS,
            'update_order',
        )
        self._validate_optional_non_negative_number(
            settings['beta_start'],
            'beta_start',
        )
        self._validate_optional_non_negative_number(
            settings['beta_end'],
            'beta_end',
        )
        self._validate_timeout(settings['timeout_seconds'])
        self._validate_optional_positive_integer(
            settings['trace_interval'],
            'trace_interval',
        )
        self._validate_initial_samples(
            settings['initial_samples'],
            settings['num_reads'],
        )
        self._validate_schedule_options(settings)

    def _validate_positive_integer(self, value, name):
        """Require a genuine Python integer strictly above zero."""
        if type(value) is not int or value <= 0:
            raise ValueError(f'{name} must be a positive integer.')

    def _validate_non_negative_integer(self, value, name):
        """Require a genuine Python integer greater than or equal to zero."""
        if type(value) is not int or value < 0:
            raise ValueError(f'{name} must be a non-negative integer.')

    def _validate_optional_positive_integer(self, value, name):
        """Validate a positive integer or the absence sentinel ``None``."""
        if value is not None:
            self._validate_positive_integer(value, name)

    def _validate_choice(self, value, choices, name):
        """Validate one closed string-valued experimental option."""
        if value not in choices:
            available = ', '.join(sorted(choices))
            raise ValueError(f'{name} must be one of: {available}.')

    def _validate_optional_non_negative_number(self, value, name):
        """Validate a finite non-negative inverse temperature or ``None``."""
        if value is None:
            return
        if (
            not self._is_finite_number(value)
            or value < 0
        ):
            raise ValueError(f'{name} must be finite and non-negative or None.')

    def _validate_timeout(self, value):
        """Accept no deadline or a finite, strictly positive duration."""
        if value is None:
            return
        if not self._is_finite_number(value) or value <= 0:
            raise ValueError(
                'timeout_seconds must be a finite positive number or None.'
            )

    def _is_finite_number(self, value):
        """Recognise values representable as finite binary64 config numbers."""
        if type(value) not in {int, float}:
            return False
        try:
            converted = float(value)
        except OverflowError:
            return False
        return math.isfinite(converted)

    def _validate_initial_samples(self, samples, num_reads):
        """Validate binary values before variable-count validation is possible."""
        if samples is None:
            return
        if not isinstance(samples, (list, tuple)) or not samples:
            raise ValueError(
                'initial_samples must be a non-empty list of binary samples.'
            )
        if len(samples) > num_reads:
            raise ValueError(
                'initial_samples cannot contain more samples than num_reads.'
            )
        for sample in samples:
            if not isinstance(sample, (list, tuple)):
                raise ValueError(
                    'initial_samples must contain list or tuple samples.'
                )
            if any(type(value) is not int or value not in {0, 1}
                   for value in sample):
                raise ValueError(
                    'initial_samples must contain only integer 0 or 1 values.'
                )

    def _validate_schedule_options(self, settings):
        """Require schedule fields to agree with the selected schedule type."""
        schedule_type = settings['beta_schedule_type']
        custom_schedule = settings['beta_schedule']

        if schedule_type == 'custom':
            if (
                settings['beta_start'] is not None
                or settings['beta_end'] is not None
            ):
                raise ValueError(
                    'beta_start and beta_end must be None for a custom schedule.'
                )
            if not isinstance(custom_schedule, (list, tuple)):
                raise ValueError(
                    'beta_schedule must be a list or tuple for a custom schedule.'
                )
            return

        if custom_schedule is not None:
            raise ValueError(
                'beta_schedule is only valid when beta_schedule_type is custom.'
            )

    def _resolve_beta_schedule(self, problem, config):
        """Build the complete deterministic inverse-temperature schedule."""
        if config['beta_schedule_type'] == 'custom':
            return self._validate_custom_schedule(
                config['beta_schedule'],
                config['sweeps'],
            )

        automatic_start, automatic_end = self._estimate_beta_range(problem)
        beta_start = (
            automatic_start
            if config['beta_start'] is None
            else self._as_finite_beta(config['beta_start'], 'beta_start')
        )
        beta_end = (
            automatic_end
            if config['beta_end'] is None
            else self._as_finite_beta(config['beta_end'], 'beta_end')
        )
        if beta_end < beta_start:
            raise ValueError(
                'beta_end must be greater than or equal to beta_start.'
            )

        if config['beta_schedule_type'] == 'linear':
            return np.linspace(beta_start, beta_end, config['sweeps'])
        if beta_start <= 0 or beta_end <= 0:
            raise ValueError(
                'A geometric beta schedule requires positive endpoints.'
            )
        return np.geomspace(beta_start, beta_end, config['sweeps'])

    def _as_finite_beta(self, value, name):
        """Convert a validated JSON number to the NumPy backend's scalar."""
        try:
            converted = float(value)
        except OverflowError as error:
            raise ValueError(f'{name} must fit a finite binary64 number.') from error
        if not math.isfinite(converted):
            raise ValueError(f'{name} must fit a finite binary64 number.')
        return converted

    def _validate_custom_schedule(self, schedule, sweeps):
        """Return a copied custom schedule after shape and order checks."""
        if len(schedule) != sweeps:
            raise ValueError('beta_schedule length must equal sweeps.')
        if any(
            not self._is_finite_number(value) or value < 0
            for value in schedule
        ):
            raise ValueError(
                'beta_schedule values must be finite and non-negative.'
            )
        try:
            values = np.asarray(schedule, dtype=float)
        except (OverflowError, TypeError, ValueError) as error:
            raise ValueError(
                'beta_schedule values must fit finite binary64 numbers.'
            ) from error
        if not np.all(np.isfinite(values)):
            raise ValueError(
                'beta_schedule values must fit finite binary64 numbers.'
            )
        if np.any(values[1:] < values[:-1]):
            raise ValueError('beta_schedule must be non-decreasing.')
        return values.copy()

    def _estimate_beta_range(self, problem):
        """Estimate endpoints solely from a bound on one-flip energy changes."""
        variable_bounds = [Fraction(0) for _ in range(problem['num_variables'])]
        for left, right, coefficient in problem['terms']:
            magnitude = abs(Fraction(coefficient))
            variable_bounds[left] += magnitude
            if right != left:
                variable_bounds[right] += magnitude

        scale = max(variable_bounds, default=Fraction(0))
        if scale == 0:
            return 0.1, 5.0
        return (
            self._finite_positive_ratio(Fraction(1, 10), scale),
            self._finite_positive_ratio(Fraction(5), scale),
        )

    def _finite_positive_ratio(self, numerator, denominator):
        """Convert a positive exact ratio to a usable binary64 temperature."""
        try:
            value = float(numerator / denominator)
        except OverflowError:
            value = np.finfo(float).max
        if math.isinf(value):
            value = np.finfo(float).max
        if value == 0:
            value = np.nextafter(0.0, 1.0)
        return value

    def _build_adjacency(self, problem):
        """Build exact sparse neighborhoods, excluding the constant offset."""
        linear_biases = [
            Fraction(0) for _ in range(problem['num_variables'])
        ]
        interactions = [
            [] for _ in range(problem['num_variables'])
        ]

        for left, right, coefficient in problem['terms']:
            exact_coefficient = Fraction(coefficient)
            if left == right:
                linear_biases[left] += exact_coefficient
            else:
                interactions[left].append((right, exact_coefficient))
                interactions[right].append((left, exact_coefficient))

        return tuple(
            _VariableNeighborhood(
                linear_bias=linear_biases[index],
                interactions=tuple(interactions[index]),
            )
            for index in range(problem['num_variables'])
        )

    def _delta_energy(self, sample, variable_index, adjacency):
        """Return the exact non-constant energy change for one bit flip."""
        neighborhood = adjacency[variable_index]
        local_field = neighborhood.linear_bias
        for neighbor, coefficient in neighborhood.interactions:
            if int(sample[neighbor]):
                local_field += coefficient
        direction = 1 - 2 * int(sample[variable_index])
        return direction * local_field

    def _energy_without_offset(self, sample, adjacency):
        """Evaluate one initial state exactly without the irrelevant offset."""
        energy = Fraction(0)
        for variable_index, neighborhood in enumerate(adjacency):
            if int(sample[variable_index]):
                energy += neighborhood.linear_bias
                energy += sum(
                    coefficient
                    for neighbor, coefficient in neighborhood.interactions
                    if neighbor > variable_index and int(sample[neighbor])
                )
        return energy

    def _make_random_generator(self, seed):
        """Own all stochastic decisions with one reproducible generator."""
        return np.random.default_rng(seed)

    def _prepare_initial_samples(self, variable_count, config, random_generator):
        """Create independent writable starts, cycling supplied states if needed."""
        supplied = config['initial_samples']
        if supplied is None:
            generated = random_generator.integers(
                0,
                2,
                size=(config['num_reads'], variable_count),
                dtype=np.int8,
            )
            return [generated[index].copy() for index in range(len(generated))]

        for sample in supplied:
            if len(sample) != variable_count:
                raise ValueError(
                    'Every initial sample length must equal num_variables.'
                )
        return [
            np.asarray(supplied[index % len(supplied)], dtype=np.int8).copy()
            for index in range(config['num_reads'])
        ]

    def _make_deadline(self, started_at, timeout_seconds):
        """Translate a relative timeout into one monotonic-clock deadline."""
        if timeout_seconds is None:
            return None
        return started_at + timeout_seconds

    def _deadline_reached(self, deadline):
        """Isolate wall-clock access so timeout behaviour remains testable."""
        return deadline is not None and time.perf_counter() >= deadline

    def _run_read(
        self,
        problem,
        initial_sample,
        beta_schedule,
        adjacency,
        config,
        random_generator,
        deadline,
        started_at,
        read_index,
        sweep_offset,
    ):
        """Run one trajectory and preserve its best state, including on timeout."""
        initial_energy = self._energy_without_offset(
            initial_sample,
            adjacency,
        )
        state = _ReadState(
            sample=initial_sample.copy(),
            energy_without_offset=initial_energy,
            best_sample=self._sample_to_list(initial_sample),
            best_energy_without_offset=initial_energy,
        )
        sweeps_completed = 0
        flips_proposed = 0
        flips_accepted = 0
        trace = []

        for sweep_index, beta in enumerate(beta_schedule):
            if self._deadline_reached(deadline):
                return self._read_result(
                    state,
                    sweeps_completed,
                    flips_proposed,
                    flips_accepted,
                    True,
                    trace,
                )

            sweep_progress = self._run_sweep(
                state,
                float(beta),
                adjacency,
                config['update_order'],
                random_generator,
                deadline,
            )
            flips_proposed += sweep_progress.flips_proposed
            flips_accepted += sweep_progress.flips_accepted
            if sweep_progress.timed_out:
                return self._read_result(
                    state,
                    sweeps_completed,
                    flips_proposed,
                    flips_accepted,
                    True,
                    trace,
                )

            sweeps_completed += 1
            global_step = sweep_offset + sweeps_completed
            if self._should_record_trace(
                global_step,
                config['trace_interval'],
            ):
                trace.append(
                    self._build_trace_entry(
                        problem,
                        state.best_sample,
                        global_step,
                        started_at,
                        read_index,
                        sweep_index,
                    )
                )

        return self._read_result(
            state,
            sweeps_completed,
            flips_proposed,
            flips_accepted,
            False,
            trace,
        )

    def _run_sweep(
        self,
        state,
        beta,
        adjacency,
        update_order,
        random_generator,
        deadline,
    ):
        """Attempt one Metropolis update for every variable."""
        proposed = 0
        accepted = 0
        for variable_index in self._variable_order(
            len(state.sample),
            update_order,
            random_generator,
        ):
            if self._deadline_reached(deadline):
                return _SweepProgress(proposed, accepted, True)

            delta = self._delta_energy(
                state.sample,
                variable_index,
                adjacency,
            )
            proposed += 1
            if self._accept_flip(delta, beta, random_generator):
                state.sample[variable_index] = 1 - state.sample[variable_index]
                state.energy_without_offset += delta
                accepted += 1
                state.best_sample, state.best_energy_without_offset = (
                    self._update_incumbent(
                        state.sample,
                        state.energy_without_offset,
                        state.best_sample,
                        state.best_energy_without_offset,
                    )
                )

        return _SweepProgress(proposed, accepted, False)

    def _variable_order(self, variable_count, update_order, random_generator):
        """Return canonical or independently shuffled indices for one sweep."""
        if update_order == 'sequential':
            return range(variable_count)
        return (
            int(index)
            for index in random_generator.permutation(variable_count)
        )

    def _accept_flip(self, delta_energy, beta, random_generator):
        """Apply the single-bit Metropolis acceptance rule."""
        if delta_energy <= 0:
            return True
        if beta == 0:
            return True
        try:
            scaled_delta = float(Fraction(float(beta)) * delta_energy)
        except OverflowError:
            return False
        probability = (
            0.0
            if not math.isfinite(scaled_delta)
            else math.exp(-scaled_delta)
        )
        return float(random_generator.random()) < probability

    def _update_incumbent(
        self,
        sample,
        energy_without_offset,
        best_sample,
        best_energy_without_offset,
    ):
        """Update an incumbent with stable lexicographic tie-breaking."""
        candidate = self._sample_to_list(sample)
        if (
            best_sample is None
            or energy_without_offset < best_energy_without_offset
            or (
                energy_without_offset == best_energy_without_offset
                and candidate < best_sample
            )
        ):
            return candidate, energy_without_offset
        return best_sample, best_energy_without_offset

    def _sample_to_list(self, sample):
        """Convert NumPy state to JSON-native Python binary integers."""
        return [int(value) for value in sample]

    def _should_record_trace(self, completed_sweeps, trace_interval):
        """Decide whether one completed sweep is a requested trace point."""
        return (
            trace_interval is not None
            and completed_sweeps % trace_interval == 0
        )

    def _build_trace_entry(
        self,
        problem,
        sample,
        step,
        started_at,
        read_index,
        sweep_index,
    ):
        """Build one contract-valid observation with canonical exact energy."""
        return {
            'step': step,
            'time_seconds': max(0.0, time.perf_counter() - started_at),
            'energy': _evaluate_qubo(problem, sample),
            'sample': list(sample),
            'metadata': {
                'read': read_index,
                'sweep': sweep_index,
            },
        }

    def _read_result(
        self,
        state,
        sweeps_completed,
        flips_proposed,
        flips_accepted,
        timed_out,
        trace,
    ):
        """Freeze a trajectory result before returning it to orchestration."""
        return _ReadResult(
            best_sample=list(state.best_sample),
            best_energy_without_offset=state.best_energy_without_offset,
            sweeps_completed=sweeps_completed,
            flips_proposed=flips_proposed,
            flips_accepted=flips_accepted,
            timed_out=timed_out,
            trace=tuple(trace),
        )

    def _empty_progress(self, config, beta_schedule):
        """Create JSON-native counters shared by both termination paths."""
        return {
            'reads_completed': 0,
            'reads_requested': config['num_reads'],
            'sweeps_completed': 0,
            'sweeps_requested': config['num_reads'] * len(beta_schedule),
            'flips_proposed': 0,
            'flips_accepted': 0,
        }

    def _accumulate_progress(self, progress, read_result):
        """Merge one trajectory's counters into solve-wide progress."""
        progress['sweeps_completed'] += read_result.sweeps_completed
        progress['flips_proposed'] += read_result.flips_proposed
        progress['flips_accepted'] += read_result.flips_accepted
        if not read_result.timed_out:
            progress['reads_completed'] += 1

    def _build_annealing_outcome(
        self,
        status,
        best_sample,
        progress,
        trace,
        beta_schedule,
        config,
    ):
        """Describe a heuristic result without making an optimality claim."""
        proposed = progress['flips_proposed']
        metrics = {
            **progress,
            'acceptance_rate': (
                progress['flips_accepted'] / proposed if proposed else 0.0
            ),
        }
        metadata = {
            'algorithm': 'simulated_annealing',
            'num_reads': config['num_reads'],
            'sweeps': config['sweeps'],
            'beta_schedule_type': config['beta_schedule_type'],
            'beta_start': float(beta_schedule[0]),
            'beta_end': float(beta_schedule[-1]),
            'update_order': config['update_order'],
            'seed': config['seed'],
        }
        return QuboSolveOutcome(
            status=status,
            best_sample=best_sample,
            termination_reason=(
                'timeout_reached'
                if status == 'timeout'
                else 'annealing_completed'
            ),
            metrics=metrics,
            trace=tuple(trace),
            metadata=metadata,
        )
