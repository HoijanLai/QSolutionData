"""Qiskit Aer matrix-product-state QAOA for canonical ``qubo.v1``."""

from __future__ import annotations

import math
import re
import time
from itertools import pairwise

import numpy as np

from ...contracts.validation import _evaluate_qubo
from .base import BaseQuboSolver, QuboSolveOutcome


class _OptimizationTimeout(RuntimeError):
    pass


class AerMpsQaoaSolver(BaseQuboSolver):
    """Run shallow QAOA with Qiskit Aer's matrix-product-state method.

    Qiskit, Qiskit Aer, and SciPy are optional and imported only when ``solve``
    is executed. The public boundary remains the repository's dependency-free
    ``QuboSolver`` contract.
    """

    SOLVER_NAME = 'qiskit-aer-mps-qaoa'
    SOLVER_VERSION = '0.1.0'
    BACKEND = 'qiskit-aer-matrix-product-state'

    DEFAULT_LAYERS = 1
    DEFAULT_OPTIMIZER_ITERATIONS = 20
    DEFAULT_RESTARTS = 1
    DEFAULT_SHOTS = 4096
    DEFAULT_SEED = 0
    DEFAULT_MAX_BOND_DIMENSION = 128
    DEFAULT_TRUNCATION_THRESHOLD = 1e-8
    _ALLOWED_CONFIG_FIELDS = frozenset(
        {
            'layers',
            'optimizer_iterations',
            'restarts',
            'shots',
            'seed',
            'max_bond_dimension',
            'truncation_threshold',
            'timeout_seconds',
            'initial_parameters',
            'cardinality_partitions',
        }
    )

    def _resolve_config(self, config):
        resolved = super()._resolve_config(config)
        unknown = sorted(set(resolved).difference(self._ALLOWED_CONFIG_FIELDS))
        if unknown:
            raise ValueError(f'Unknown Aer MPS QAOA config fields: {unknown}')
        settings = {
            'layers': resolved.get('layers', self.DEFAULT_LAYERS),
            'optimizer_iterations': resolved.get(
                'optimizer_iterations',
                self.DEFAULT_OPTIMIZER_ITERATIONS,
            ),
            'restarts': resolved.get('restarts', self.DEFAULT_RESTARTS),
            'shots': resolved.get('shots', self.DEFAULT_SHOTS),
            'seed': resolved.get('seed', self.DEFAULT_SEED),
            'max_bond_dimension': resolved.get(
                'max_bond_dimension',
                self.DEFAULT_MAX_BOND_DIMENSION,
            ),
            'truncation_threshold': resolved.get(
                'truncation_threshold',
                self.DEFAULT_TRUNCATION_THRESHOLD,
            ),
            'timeout_seconds': resolved.get('timeout_seconds'),
            'initial_parameters': resolved.get('initial_parameters'),
            'cardinality_partitions': resolved.get('cardinality_partitions'),
        }
        self._validate_settings(settings)
        return settings

    def _run(self, problem, config):
        qiskit = self._load_qiskit_components()
        started = time.perf_counter()
        deadline = (
            None
            if config['timeout_seconds'] is None
            else started + config['timeout_seconds']
        )
        scale = self._coefficient_scale(problem)
        ising = self._qubo_to_ising(problem, scale)
        circuit, observable = self._build_qaoa_circuit(
            problem['num_variables'],
            ising,
            config['layers'],
            config['cardinality_partitions'],
            qiskit,
        )
        backend_options = self._backend_options(config)
        transpile_backend = qiskit['AerSimulator'](**backend_options)
        circuit = qiskit['transpile'](
            circuit,
            transpile_backend,
            optimization_level=1,
        )
        if getattr(circuit, 'layout', None) is not None:
            observable = observable.apply_layout(circuit.layout)
        estimator = qiskit['EstimatorV2'](
            options={
                'backend_options': backend_options,
                'run_options': {'shots': None},
            }
        )

        best = {
            'parameters': self._initial_parameters(config, 0),
            'expectation': math.inf,
            'evaluations': 0,
        }
        timed_out = False
        optimizer_messages = []
        for restart in range(config['restarts']):
            initial = self._initial_parameters(config, restart)
            try:
                result = self._optimize_once(
                    circuit,
                    observable,
                    estimator,
                    initial,
                    config,
                    deadline,
                    best,
                    qiskit,
                )
                optimizer_messages.append(str(result.message))
            except _OptimizationTimeout:
                timed_out = True
                optimizer_messages.append('timeout')
                break

        sample, sampling = self._sample_best_parameters(
            problem,
            circuit,
            best['parameters'],
            config,
            backend_options,
            qiskit,
        )
        return QuboSolveOutcome(
            status='timeout' if timed_out else 'feasible',
            best_sample=sample,
            termination_reason=(
                'timeout_reached' if timed_out else 'optimizer_completed'
            ),
            metrics={
                'expectation_normalized_without_offset': (
                    None if not math.isfinite(best['expectation'])
                    else float(best['expectation'])
                ),
                'optimizer_evaluations': int(best['evaluations']),
                'unique_samples_observed': int(sampling['unique_samples']),
                'selected_occurrences': int(sampling['selected_occurrences']),
                'max_bond_dimension_observed': sampling[
                    'max_bond_dimension_observed'
                ],
                'sampling_time_seconds': sampling['sampling_time_seconds'],
            },
            metadata={
                'algorithm': 'qaoa',
                'simulation_method': 'matrix_product_state',
                'layers': config['layers'],
                'shots': config['shots'],
                'seed': config['seed'],
                'max_bond_dimension': config['max_bond_dimension'],
                'truncation_threshold': config['truncation_threshold'],
                'coefficient_scale': scale,
                'parameters': [float(value) for value in best['parameters']],
                'optimizer_messages': optimizer_messages,
                'mixer': (
                    'partitioned_xy'
                    if config['cardinality_partitions'] is not None
                    else 'transverse_x'
                ),
                'cardinality_partitions': config['cardinality_partitions'],
            },
        )

    def _load_qiskit_components(self):
        try:
            from qiskit import QuantumCircuit, transpile
            from qiskit.circuit import ParameterVector
            from qiskit.circuit.library import StatePreparation
            from qiskit.quantum_info import SparsePauliOp
            from qiskit_aer import AerSimulator
            from qiskit_aer.primitives import EstimatorV2
            from scipy.optimize import minimize
        except ImportError as error:
            raise ImportError(
                'AerMpsQaoaSolver requires qiskit, qiskit-aer, and scipy.'
            ) from error
        return {
            'AerSimulator': AerSimulator,
            'EstimatorV2': EstimatorV2,
            'ParameterVector': ParameterVector,
            'QuantumCircuit': QuantumCircuit,
            'SparsePauliOp': SparsePauliOp,
            'StatePreparation': StatePreparation,
            'minimize': minimize,
            'transpile': transpile,
        }

    def _validate_settings(self, settings):
        for name in (
            'layers',
            'restarts',
            'shots',
            'max_bond_dimension',
        ):
            value = settings[name]
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f'{name} must be a positive integer.')
        iterations = settings['optimizer_iterations']
        if not isinstance(iterations, int) or isinstance(iterations, bool) or iterations < 0:
            raise ValueError('optimizer_iterations must be a non-negative integer.')
        seed = settings['seed']
        if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
            raise ValueError('seed must be a non-negative integer.')
        threshold = settings['truncation_threshold']
        if (
            not isinstance(threshold, (int, float))
            or isinstance(threshold, bool)
            or not math.isfinite(float(threshold))
            or threshold < 0
        ):
            raise ValueError('truncation_threshold must be finite and non-negative.')
        timeout = settings['timeout_seconds']
        if timeout is not None and (
            not isinstance(timeout, (int, float))
            or isinstance(timeout, bool)
            or not math.isfinite(float(timeout))
            or timeout <= 0
        ):
            raise ValueError('timeout_seconds must be positive or None.')
        parameters = settings['initial_parameters']
        if parameters is not None:
            if not isinstance(parameters, (list, tuple)):
                raise TypeError('initial_parameters must be a list, tuple, or None.')
            if len(parameters) != 2 * settings['layers']:
                raise ValueError('initial_parameters must contain 2 * layers values.')
            if any(not math.isfinite(float(value)) for value in parameters):
                raise ValueError('initial_parameters must contain finite numbers.')
        self._validate_cardinality_partitions(settings['cardinality_partitions'])

    def _validate_cardinality_partitions(self, partitions):
        if partitions is None:
            return
        if not isinstance(partitions, list) or not partitions:
            raise TypeError('cardinality_partitions must be a non-empty list or None.')
        used = set()
        for position, partition in enumerate(partitions):
            if not isinstance(partition, dict):
                raise TypeError('Each cardinality partition must be a dictionary.')
            if set(partition) != {'indices', 'count'}:
                raise ValueError(
                    'Each cardinality partition requires only indices and count.'
                )
            indices = partition['indices']
            count = partition['count']
            if (
                not isinstance(indices, list)
                or not indices
                or any(
                    not isinstance(index, int)
                    or isinstance(index, bool)
                    or index < 0
                    for index in indices
                )
                or len(indices) != len(set(indices))
            ):
                raise ValueError(f'Partition {position} has invalid indices.')
            overlap = used.intersection(indices)
            if overlap:
                raise ValueError(f'Cardinality partitions overlap at {sorted(overlap)}.')
            used.update(indices)
            if (
                not isinstance(count, int)
                or isinstance(count, bool)
                or not 0 <= count <= len(indices)
            ):
                raise ValueError(f'Partition {position} has an invalid count.')

    def _coefficient_scale(self, problem):
        magnitudes = [abs(float(term[2])) for term in problem['terms'] if term[2]]
        return max(magnitudes, default=1.0)

    def _qubo_to_ising(self, problem, scale):
        linear = [0.0] * problem['num_variables']
        quadratic = {}
        for left, right, coefficient in problem['terms']:
            value = float(coefficient) / scale
            if left == right:
                linear[left] -= value / 2
            else:
                linear[left] -= value / 4
                linear[right] -= value / 4
                quadratic[(left, right)] = (
                    quadratic.get((left, right), 0.0) + value / 4
                )
        return {'linear': linear, 'quadratic': quadratic}

    def _build_qaoa_circuit(
        self,
        variable_count,
        ising,
        layers,
        cardinality_partitions,
        qiskit,
    ):
        parameters = qiskit['ParameterVector']('theta', 2 * layers)
        circuit = qiskit['QuantumCircuit'](variable_count)
        if cardinality_partitions is None:
            circuit.h(range(variable_count))
        else:
            covered = set()
            for partition in cardinality_partitions:
                indices = partition['indices']
                covered.update(indices)
                self._prepare_dicke_state(
                    circuit,
                    indices,
                    partition['count'],
                    qiskit,
                )
            if covered != set(range(variable_count)):
                missing = sorted(set(range(variable_count)).difference(covered))
                extra = sorted(covered.difference(range(variable_count)))
                raise ValueError(
                    'cardinality_partitions must cover every QUBO variable; '
                    f'missing={missing}, extra={extra}'
                )
        for layer in range(layers):
            gamma = parameters[2 * layer]
            beta = parameters[2 * layer + 1]
            for index, coefficient in enumerate(ising['linear']):
                if coefficient:
                    circuit.rz(2 * gamma * coefficient, index)
            for (left, right), coefficient in ising['quadratic'].items():
                if coefficient:
                    circuit.rzz(2 * gamma * coefficient, left, right)
            if cardinality_partitions is None:
                circuit.rx(2 * beta, range(variable_count))
            else:
                self._apply_partitioned_xy_mixer(
                    circuit,
                    cardinality_partitions,
                    beta,
                )

        sparse_terms = []
        for index, coefficient in enumerate(ising['linear']):
            if coefficient:
                sparse_terms.append(('Z', [index], coefficient))
        for (left, right), coefficient in ising['quadratic'].items():
            if coefficient:
                sparse_terms.append(('ZZ', [left, right], coefficient))
        if not sparse_terms:
            sparse_terms.append(('I', [0], 0.0))
        observable = qiskit['SparsePauliOp'].from_sparse_list(
            sparse_terms,
            num_qubits=variable_count,
        )
        return circuit, observable

    def _prepare_dicke_state(self, circuit, indices, count, qiskit):
        if count == 0:
            return
        if count == len(indices):
            circuit.x(indices)
            return
        dimension = 1 << len(indices)
        population = math.comb(len(indices), count)
        amplitude = 1 / math.sqrt(population)
        state = np.zeros(dimension, dtype=complex)
        for basis_index in range(dimension):
            if basis_index.bit_count() == count:
                state[basis_index] = amplitude
        circuit.append(qiskit['StatePreparation'](state), indices)

    def _apply_partitioned_xy_mixer(self, circuit, partitions, beta):
        for partition in partitions:
            indices = partition['indices']
            if len(indices) < 2 or partition['count'] in {0, len(indices)}:
                continue
            edges = list(pairwise(indices))
            if len(indices) > 2:
                edges.append((indices[-1], indices[0]))
            for left, right in edges:
                circuit.rxx(2 * beta, left, right)
                circuit.ryy(2 * beta, left, right)

    def _backend_options(self, config):
        return {
            'method': 'matrix_product_state',
            'matrix_product_state_max_bond_dimension': config[
                'max_bond_dimension'
            ],
            'matrix_product_state_truncation_threshold': config[
                'truncation_threshold'
            ],
            'mps_log_data': True,
            'seed_simulator': config['seed'],
        }

    def _initial_parameters(self, config, restart):
        if restart == 0 and config['initial_parameters'] is not None:
            return np.asarray(config['initial_parameters'], dtype=float)
        generator = np.random.default_rng(config['seed'] + restart)
        parameters = []
        for _ in range(config['layers']):
            parameters.extend(
                [generator.uniform(0.0, 2 * math.pi), generator.uniform(0.0, math.pi)]
            )
        return np.asarray(parameters, dtype=float)

    def _optimize_once(
        self,
        circuit,
        observable,
        estimator,
        initial,
        config,
        deadline,
        best,
        qiskit,
    ):
        def objective(parameters):
            if deadline is not None and time.perf_counter() >= deadline:
                raise _OptimizationTimeout
            pub_result = estimator.run(
                [(circuit, observable, np.asarray(parameters, dtype=float))]
            ).result()[0]
            value = float(np.asarray(pub_result.data.evs).reshape(-1)[0])
            best['evaluations'] += 1
            if value < best['expectation']:
                best['expectation'] = value
                best['parameters'] = np.asarray(parameters, dtype=float).copy()
            if deadline is not None and time.perf_counter() >= deadline:
                raise _OptimizationTimeout
            return value

        if config['optimizer_iterations'] == 0:
            objective(initial)
            return type('OptimizerResult', (), {'message': 'evaluation_only'})()
        return qiskit['minimize'](
            objective,
            initial,
            method='COBYLA',
            options={'maxiter': config['optimizer_iterations'], 'rhobeg': 0.5},
        )

    def _sample_best_parameters(
        self,
        problem,
        circuit,
        parameters,
        config,
        backend_options,
        qiskit,
    ):
        parameter_map = dict(zip(circuit.parameters, parameters))
        measured = circuit.assign_parameters(parameter_map, inplace=False)
        measured.measure_all()
        backend = qiskit['AerSimulator'](**backend_options)
        executable = qiskit['transpile'](measured, backend, optimization_level=1)
        started = time.perf_counter()
        native_result = backend.run(executable, shots=config['shots']).result()
        sampling_time = time.perf_counter() - started
        counts = native_result.get_counts()
        candidates = []
        for bitstring, occurrences in counts.items():
            compact = bitstring.replace(' ', '')
            sample = [int(value) for value in reversed(compact)]
            candidates.append(
                (_evaluate_qubo(problem, sample), sample, int(occurrences))
            )
        _, sample, occurrences = min(
            candidates,
            key=lambda item: (item[0], item[1]),
        )
        native_metadata = native_result.results[0].metadata
        return sample, {
            'unique_samples': len(counts),
            'selected_occurrences': occurrences,
            'max_bond_dimension_observed': self._max_logged_bond_dimension(
                native_metadata.get('MPS_log_data', '')
            ),
            'sampling_time_seconds': sampling_time,
        }

    def _max_logged_bond_dimension(self, log_data):
        values = []
        for match in re.findall(r'BD=\[([^\]]*)\]', str(log_data)):
            values.extend(int(value) for value in re.findall(r'\d+', match))
        return max(values) if values else None


__all__ = ['AerMpsQaoaSolver']
