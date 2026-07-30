# Optimization interface contracts

This directory contains the versioned boundaries between model construction,
representation-specific compilation, solvers, samplers, and adapters. The JSON
field names and numerical conventions in this directory are public contracts.

## Implemented contract flows

```text
cbqm.v1
  -> compile_qubo(...)
qubo.v1
  -> solver.solve(problem, config)
qubo-result.v1
  -> decode(...)

cbqm.v1
  -> native solver.solve(problem, config)
cbqm-result.v1

binary problem
  -> sampler.sample(problem, config)
binary-sample-set.v1

mis.v1
  -> native solver.solve(problem, config)
mis-result.v1
```

Backend-specific representations, including Qiskit `QuadraticProgram`,
PennyLane Hamiltonians, and Q-RBnBR `MaxCutProblem`, sit behind adapters and
must not change these contracts.

These implemented paths are not a restriction on future solvers. The Python
contract layer also exposes a representation-neutral
`Solver[ProblemT, ResultT]` call shape. New problem families must bind that
shape to their own versioned input/output schemas; they must not label a
non-QUBO result as `qubo-result.v1`.

## Files

- `schemas/cbqm.v1.schema.json`: constrained binary quadratic model.
- `schemas/cbqm-result.v1.schema.json`: native constrained solver output.
- `schemas/mis.v1.schema.json`: canonical native independent-set graph.
- `schemas/mis-result.v1.schema.json`: native MIS solver output.
- `schemas/qubo.v1.schema.json`: canonical QUBO solver input.
- `schemas/qubo-result.v1.schema.json`: canonical QUBO solver output.
- `schemas/binary-sample-set.v1.schema.json`: aggregated binary distribution.
- `../lib/contracts/cbqm_protocol.py`: typed native CBQM solver boundary.
- `../lib/contracts/sampling_protocol.py`: generic sampler call shape.
- `../lib/contracts/solver_protocol.py`: dependency-free runtime signature.
- `../lib/contracts/*_validation.py`: strict runtime and cross-field checks.
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

Python callers can apply the same boundary rules directly:

```python
from lib.contracts import (
    validate_binary_sample_set,
    validate_cbqm,
    validate_cbqm_result,
    validate_mis,
    validate_mis_result,
    validate_qubo,
    validate_qubo_result,
)

validate_cbqm(cbqm)
validate_cbqm_result(cbqm, cbqm_result)
validate_mis(mis)
validate_mis_result(mis, mis_result)
validate_qubo(qubo)
validate_qubo_result(qubo, result)
validate_binary_sample_set(sample_set)
```

The runtime validators also reject Python-only JSON values, booleans used as
binary integers, non-finite nested metadata, mismatched problem IDs, and
reported energies that do not match their samples.

Canonical QUBO energy recomputation first accumulates the JSON numeric values
as exact rationals. Integral results remain JSON integers, so a large offset
cannot erase a one-unit objective difference through an early binary64 cast.
Reported energies must equal that canonical JSON-number result; a
scale-dependent relative tolerance is deliberately not used.

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

The portfolio reference builder is `lib.portfolio.build_portfolio_cbqm`. It
requires an explicit objective configuration and converts the prepared
portfolio payload into this contract without adding constraint penalties.

The reference compiler is `lib.compilers.compile_qubo`. It returns both a
`qubo.v1` payload and `qubo-compilation-context.v1`; its penalty and numerical
encoding choices are supplied explicitly by the caller.

## `cbqm-result.v1`

A native constrained solver reports the original CBQM objective separately
from feasibility. A present candidate is an atomic triplet:

```text
best_sample + best_objective + feasibility
```

The runtime validator recomputes the original linear/quadratic objective, every
fixed value, and every explicit lower/upper constraint. `optimal` and
`feasible` require a feasible candidate. `infeasible` requires no candidate.
`timeout`, `unknown`, and `error` may have no candidate; when they retain an
infeasible observed sample, its violation list remains explicit.

Optional bounds and proof evidence are solver attestations. In particular,
`proof.independently_verified: false` is expected for a proof emitted by the
solver itself. Persistent `ProblemCase.best_known.exact` promotion requires a
separate verifier, certificate, or authoritative external registration.

## `mis.v1` and `mis-result.v1`

`mis.v1` is a graph-native maximum independent set contract. Vertices retain
stable contiguous indices and real names; undirected edges are unique
`[u, v]` pairs in canonical order. The objective is explicitly either
`maximum-cardinality` or `maximum-weight`, and optional `fixed_values` are hard
assignments.

The result witness is a strictly increasing vertex-index set. Its objective,
cardinality, total weight and feasibility are recomputed from the source graph.
`ExactMisSolver` provides branch-and-reduce proofs and bounds;
`GreedyMisSolver` provides a deterministic feasible baseline with local
exchanges. A solver-reported `optimal` becomes a persistent exact best-known
only after the ProblemCase execution layer independently enumerates the native
graph within its configured verification limit.

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

## QUBO solver signature

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
`lib/contracts/solver_protocol.py`. `QuboSolver` is the schema-specific
specialization used by this flow; the generic `Solver` protocol only shares the
call shape. A solver need not inherit from either protocol.

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

The result validator confirms status/candidate consistency and recomputes the
candidate energy, but a generic schema boundary cannot prove that an arbitrary
solver exhausted its search space. Thus `status: optimal` remains the solver's
attestation. Before turning that attestation into a persistent
`ProblemCase.best_known.exact` lock, `solve_problem_task` independently
enumerates the selected QUBO within its configured verification limit.

`trace` is optional and ordered chronologically. It can preserve the
breadcrumb/history idea used by Q-RBnBR without making tracing mandatory.

## `binary-sample-set.v1`

This artifact stores a full aggregated binary distribution without hiding it
inside solver metadata or trace. Records contain only:

```text
sample + occurrences + optional metadata
```

They are unique and strictly lexicographically ordered. Empirical and exact
distributions use a positive integer `shots`, and the occurrence total must
equal it. A deterministic producer uses `shots: null`, exactly one record, and
`occurrences: 1`.

Objective or feasibility claims are deliberately absent. Consumers evaluate
records against the canonical source problem. Large raw shot streams should
remain external; repository artifacts store aggregated records or a manifest.

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

The reference implementations live in `lib/adapters/`. In particular,
`QRBnBRSolverAdapter` exposes a native Q-RBnBR solver through the canonical
`solve(problem, config=None)` signature.
