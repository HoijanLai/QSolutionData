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

`validation.py` is internal support for builders, compilers, and adapters. Its
protected functions validate `cbqm.v1`, `qubo.v1`, sparse term conventions,
indices, and finite coefficients. Solver implementations normally should not
call these helpers directly; they should validate only the additional config
and invariants specific to their algorithm.

Authoritative serializable artifacts:

- `contracts/schemas/cbqm.v1.schema.json`
- `contracts/schemas/qubo.v1.schema.json`
- `contracts/schemas/qubo-result.v1.schema.json`
- `contracts/examples/`
