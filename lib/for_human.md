# Library map

`lib` implements the path from cleaned fund data to a solver result. Data
reading and EDA remain in the root `eda.py` and are intentionally outside this
package.

```text
clean DataFrame
  -> preprocessing: score, classify, compare
  -> portfolio: encode weights, build constraints and cbqm.v1
  -> compilers: cbqm.v1 -> qubo.v1
  -> solvers or adapters: solve the backend's declared problem contract
  -> qubo-result.v1
```

Module responsibilities:

- `preprocessing/`: model-ready scoring, classification, and similarity features.
- `portfolio/`: portfolio variables, policies, constraints, and CBQM creation.
- `pipeline/`: orchestration of asset and portfolio preparation.
- `contracts/`: generic and representation-specific solver protocols plus
  payload validation.
- `compilers/`: constrained-model to solver-model transformations.
- `adapters/`: integration with external modelling or solver frameworks.
- `solvers/`: concrete algorithms grouped into native `qubo`, `cbqm`, and
  `mis` problem representations.

Most public functions are re-exported from `lib`, for example:

```python
from lib import (
    prepare_qubo_inputs,
    build_portfolio_cbqm,
    compile_qubo,
)
```

Each subdirectory contains its own `for_human.md` with inputs, outputs, usage,
and numerical conventions.
