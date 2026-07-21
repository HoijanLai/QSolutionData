# Optimization interface contracts

This directory is the versioned boundary between model construction, QUBO
compilation, solvers, and solver adapters. The JSON field names and numerical
conventions in this directory are part of the public contract.

## Contract flow

```text
cbqm.v1
  -> compile_qubo(...)
qubo.v1
  -> solver.solve(problem, config)
qubo-result.v1
  -> decode(...)
```

Backend-specific representations, including Qiskit `QuadraticProgram`,
PennyLane Hamiltonians, and Q-RBnBR `MaxCutProblem`, sit behind adapters and
must not change these contracts.

## Files

- `schemas/cbqm.v1.schema.json`: constrained binary quadratic model.
- `schemas/qubo.v1.schema.json`: canonical solver input.
- `schemas/qubo-result.v1.schema.json`: canonical solver output.
- `python/solver_protocol.py`: dependency-free Python solver signature.
- `examples/`: mutually consistent minimal payloads.

## Shared conventions

- Every payload carries a literal `schema` version and a stable `problem_id`.
- Variables are addressed by zero-based integer indices. An index is stable
  within one problem and must equal its position in the variable list.
- All coefficients and energies must be finite real numbers; `NaN` and
  infinities are invalid even if a JSON implementation accepts them.
- Sparse terms with a zero coefficient should be omitted.
- Duplicate sparse terms are invalid. Producers must combine them before
  serialization.
- Unknown fields are rejected at the defined structural levels. Extension data
  belongs in `metadata`.

## `cbqm.v1`

`cbqm.v1` is the neutral constrained-model format. It follows the concepts of
Qiskit `QuadraticProgram` without requiring Qiskit.

The objective is evaluated as

```text
offset + sum(linear[i] * x[i]) + sum(quadratic[i,j] * x[i] * x[j])
```

Quadratic objective terms use `i <= j`. A diagonal quadratic term is legal;
for binary variables it is algebraically equivalent to a linear term, but the
producer's representation is preserved at this layer.

Each constraint is linear and means

```text
lower_bound <= sum(linear[i] * x[i]) <= upper_bound
```

At least one bound must be present. Equal lower and upper bounds represent an
equality. `fixed_values` are hard assignments and are not penalty hints.

`cbqm.v1` deliberately does not specify penalty strengths, slack encodings, or
constraint compilation policy. Those choices belong to `compile_qubo` and
must be recorded in the generated QUBO metadata.

## `qubo.v1`

`qubo.v1` is the only input a generic QUBO solver must understand. It always
represents minimization and its exact energy is

```text
E(x) = offset + sum([i, j, coefficient] in terms)
                    coefficient * x[i] * x[j]
```

Every term must satisfy `0 <= i <= j < num_variables`. Diagonal terms are the
linear QUBO coefficients. Off-diagonal coefficients are used exactly once:
there is no implicit factor of two.

The returned energy must include `offset`. A solver may omit the constant while
optimizing internally, but it must restore it in `qubo-result.v1`.

## Solver signature

The canonical Python call is:

```python
result = solver.solve(problem, config=None)
```

- `problem` conforms to `qubo.v1`.
- `config` is solver-owned configuration. Portable callers may pass `None` or
  an empty mapping.
- `result` conforms to `qubo-result.v1`.
- Invalid input should raise `ValueError` or a more specific input exception.
- A normal algorithmic termination, including timeout or infeasibility, should
  be returned as a result status rather than raised as an exception.
- Solvers must not mutate `problem` or `config`.

The dependency-free structural interface is defined in
`python/solver_protocol.py`. A solver need not inherit from a project base
class; satisfying the method signature is sufficient.

## `qubo-result.v1`

`best_sample` is ordered by the QUBO variable indices and contains only `0` or
`1`. `best_energy` is the canonical QUBO energy, including offset. Both are
`null` when no candidate solution exists.

Statuses have the following meanings:

- `optimal`: optimality was established.
- `feasible`: a candidate was returned without proof of optimality.
- `infeasible`: the solver established that no candidate exists.
- `timeout`: stopped by a time limit; a candidate may still be present.
- `error`: the backend failed after accepting the input.
- `unknown`: no stronger conclusion is available.

`trace` is optional and ordered chronologically. It can preserve the
breadcrumb/history idea used by Q-RBnBR without making tracing mandatory.

## Adapter boundary

A Q-RBnBR adapter should expose the equivalent of:

```python
maxcut_problem, context = adapter.from_qubo(qubo_problem)
native_solution = solver.solve(maxcut_problem)
result = adapter.to_result(native_solution, context)
```

`context` must retain the QUBO offset, minimization/maximization sign change,
variable-to-node mapping, auxiliary anchor node, and any QUBO-to-Ising/MaxCut
energy transformation. The final adapter output must again satisfy
`qubo-result.v1` and report energy in the original QUBO convention.
