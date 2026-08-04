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

### CBQM specialization

`CbqmSolver` binds the same generic call shape to:

```text
cbqm.v1 -> cbqm-result.v1
```

Its result reports the original objective and an independently recomputable
feasibility summary. Native constrained solvers must not return
`qubo-result.v1`, because penalty energy is not the business objective.

### MIS specialization

`MisSolver` binds the generic call shape to:

```text
mis.v1 -> mis-result.v1
```

The candidate is a canonical, strictly increasing vertex-index set. Result
validation recomputes cardinality, total weight, fixed-value compliance and
independent-set feasibility from the source graph. Maximum-cardinality and
maximum-weight semantics remain explicit in the input contract.

## Sampler protocol

`Sampler[ProblemT, SampleSetT]` defines:

```python
sample_set = sampler.sample(problem, config=None)
```

`BinarySampleSet` describes `binary-sample-set.v1`. One class may implement
both `sample()` for an aggregated distribution and `solve()` for a standard
best-candidate result. The existing `solve()` API does not need to change.

## Validation helpers

`lib.contracts` publicly exports `validate_cbqm`, `validate_cbqm_result`,
`validate_mis`, `validate_mis_result`, `evaluate_mis_solution`,
`validate_qubo`, `validate_qubo_result`, and
`validate_binary_sample_set`. Builders, compilers, adapters, solvers, and
samplers should call the applicable public validator at their boundary, then
validate only algorithm-specific configuration and invariants.

Authoritative serializable artifacts:

- `contracts/schemas/cbqm.v1.schema.json`
- `contracts/schemas/cbqm-result.v1.schema.json`
- `contracts/schemas/mis.v1.schema.json`
- `contracts/schemas/mis-result.v1.schema.json`
- `contracts/schemas/qubo.v1.schema.json`
- `contracts/schemas/qubo-result.v1.schema.json`
- `contracts/schemas/binary-sample-set.v1.schema.json`
- `contracts/examples/`
