# Solver adapters

Adapters keep optional solver frameworks outside the neutral `cbqm.v1`,
`qubo.v1`, and `qubo-result.v1` contracts.

Adapters contain representation conversion and external-backend wrappers, not
the optimization algorithms owned by `lib/solvers/`.

## Qiskit

`to_qiskit` accepts `cbqm.v1` and returns `(QuadraticProgram, context)`.
`from_qiskit` accepts a Qiskit `QuadraticProgram` and optional context, and
returns `cbqm.v1`.

```python
from lib.adapters import QiskitQuadraticProgramAdapter

adapter = QiskitQuadraticProgramAdapter()
quadratic_program, context = adapter.to_qiskit(cbqm_problem)
restored_cbqm = adapter.from_qiskit(quadratic_program, context)
```

Qiskit is imported only when `to_qiskit` is called. A two-sided CBQM range is
represented by two Qiskit linear constraints. Fixed values are represented by
reserved equality constraints. The returned context restores the original
constraint families, ranges, variable metadata, and model metadata.

## Q-RBnBR

`from_qubo` accepts `qubo.v1` and returns `(MaxCutProblem, context)`.
`to_result` accepts a native Q-RBnBR solution plus that context and returns
`qubo-result.v1`.

```python
from lib.adapters import QRBnBRMaxCutAdapter

adapter = QRBnBRMaxCutAdapter()
maxcut_problem, context = adapter.from_qubo(qubo_problem)
native_solution = solver.solve(maxcut_problem)
result = adapter.to_result(native_solution, context)
```

The QUBO-to-MaxCut conversion adds one anchor node. For QUBO variables `x_i`,
the decoded value is whether graph node `i` differs from the anchor. Signed
edge weights are retained, and the exact energy relation is:

```text
qubo_energy = qubo_offset - cut_value
```

The result adapter always recomputes the energy from the original QUBO rather
than trusting a backend-specific cost convention.

The graph backend stores ordinary finite floats. Edge contributions are first
accumulated as exact rationals and converted only when the final float is
lossless. The adapter rejects overflow, subnormal underflow, and rounded
large-plus-small accumulation rather than emitting a graph that violates the
documented energy relation. It also bounds the sum of all absolute edge
weights with conservative sequential-rounding headroom, ensuring every
possible native cut total remains finite binary64 regardless of edge order.

An existing Q-RBnBR solver can also be exposed directly through the canonical
solver signature:

```python
from lib.adapters import QRBnBRSolverAdapter

solver = QRBnBRSolverAdapter(
    native_solver,
    result_status='feasible',
)
result = solver.solve(qubo_problem, config=None)
```

`config` may contain `native_solve_kwargs`, `status`, and
`termination_reason`. Constructor parameters remain the preferred place for
the native solver's persistent algorithm configuration.

`optimal` is deliberately stronger than a caller-selected label. The current
adapter proves it independently by enumerating every assignment of the source
QUBO:

```python
solver = QRBnBRSolverAdapter(
    native_solver,
    result_status='optimal',
    proves_optimality=True,
    optimality_verification_max_variables=20,
)
```

This check runs for every native `optimal` result, and its JSON proof metadata
is kept in `result['metadata']['optimality_proof']`. The variable limit is a
safety rail against accidental exponential work. Without the independent
check—or when the problem exceeds that configured limit—the adapter refuses to
emit `status='optimal'`. Enumeration compares exact JSON-number rationals, so a
large offset cannot hide a smaller integer improvement.
