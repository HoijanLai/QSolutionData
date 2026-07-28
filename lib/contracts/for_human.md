# Runtime contracts

This module contains dependency-free Python interfaces and semantic validators
for the versioned payloads documented in the top-level `contracts/` directory.

## Solver protocols

`Solver[ProblemT, ResultT]` defines the representation-neutral call shape:

```python
from lib.contracts import Solver


class NativeSolver:
    def solve(self, problem, config=None):
        ...
```

It deliberately does not assign a schema or numerical meaning to `problem` or
the returned value. Every production solver must use a representation-specific
specialization whose input and result contracts are explicit.

### QUBO specialization

Input: `qubo.v1` plus optional solver-owned configuration.

Output: `qubo-result.v1`.

```python
from lib.contracts import QuboSolver


class MySolver:
    def solve(self, problem, config=None):
        ...


solver: QuboSolver = MySolver()
```

`QuboSolver` specializes `Solver` as `qubo.v1 -> qubo-result.v1`. It remains a
structural `Protocol`: inheritance is unnecessary. Matching the typed
`solve(problem, config=None)` contract is sufficient.

Future direct constrained-QAOA and MIS integrations should add their own
versioned result schemas and specialized protocols instead of pretending to be
`QuboSolver`. A common method name does not make different result semantics
interchangeable.

## Validation helpers

`lib.contracts` publicly exports `validate_cbqm`, `validate_qubo`, and
`validate_qubo_result`. Builders, compilers, adapters, and solvers should call
the applicable public validator at their input/output boundary, then validate
only the additional configuration and invariants specific to their algorithm.
The underscore-prefixed helpers inside `validation.py` remain implementation
details and should not be called as public APIs.

Authoritative serializable artifacts:

- `contracts/schemas/cbqm.v1.schema.json`
- `contracts/schemas/qubo.v1.schema.json`
- `contracts/schemas/qubo-result.v1.schema.json`
- `contracts/examples/`
