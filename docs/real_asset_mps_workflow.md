# Real asset pool to Aer MPS QAOA

This workflow reads the complete source workbook, scores and classifies all
assets, selects a diversified 40-asset candidate pool, builds canonical
`cbqm.v1` and `qubo.v1` artifacts, and calls solvers through the repository's
standard `solve(problem, config)` boundary.

## First executable model

- Client profile defaults to `steady` and remains configurable.
- Forty candidates are balanced across the five broad asset classes.
- Ten assets are selected at equal 10% weights, represented as exactly 1,000
  basis points per holding.
- Profile allocation ranges are tightened to deterministic exact class quotas.
- Secondary investment-type quotas enforce the 35% type cap without adding
  inequality slack variables.
- The similarity objective retains the three strongest positive neighbors per
  asset by default.
- Simulated annealing is the classical baseline.
- Qiskit Aer uses `method="matrix_product_state"`, a feasible Dicke initial
  state per partition, and a partitioned XY mixer that preserves every exact
  quota throughout QAOA evolution.

Manager concentration, R5 exposure, investment-type concentration, and the
original profile ranges are recomputed after decoding and written into the run
audit. The 40% transitive high-similarity-group cap is reported separately as a
diagnostic: in the current real pool it conflicts with the steady profile's
minimum fixed-income allocation, so treating both as hard constraints would
make the model infeasible. A solver result is reported as a heuristic feasible
candidate, not as a proof of optimality.

## Run

Install dependencies in an isolated environment, then execute from the
repository root:

```powershell
python -m pip install -r requirements.txt
python -m problem.portfolio_workflow `
  --candidate-count 40 `
  --holding-count 10 `
  --profile steady `
  --output tmp/real-asset-mps-run
```

Useful development controls:

```powershell
# Build contracts and run only the classical baseline.
python -m problem.portfolio_workflow --skip-mps

# One MPS expectation evaluation followed by sampling.
python -m problem.portfolio_workflow --mps-iterations 0 --mps-shots 1024
```

The output directory contains:

```text
case.json
artifacts/cbqm.json
artifacts/qubo.json
candidates.csv
simulated-annealing-result.json
simulated-annealing-allocation.csv
aer-mps-qaoa-result.json
aer-mps-qaoa-allocation.csv
report.json
```

The committed workflow never stores IBM credentials and does not contact IBM
Quantum services. Qiskit Aer runs locally.

## Variable-weight workflow

The equal-weight model above remains a compact QUBO smoke test. The production
comparison keeps binary support variables but assigns continuous, non-negative
weights under the PDF-derived policy in `资产配置约束.v1.json`.

Run all reproducible paths with:

```powershell
python -m problem.variable_weight_workflow `
  --candidate-count 40 `
  --holding-count 10 `
  --minimum-active-weight 0.01 `
  --profile steady `
  --output tmp/real-asset-variable-weight-run
```

The command runs and audits four comparable methods:

1. Full-universe continuous relaxation, deterministic top-k support, and LP
   reoptimization.
2. The same relaxation-round-reoptimize baseline on the 40-asset pool.
3. Joint binary-selection and continuous-weight MILP on the 40-asset pool.
4. Simulated-annealing and Aer MPS-QAOA selection followed by the same LP
   reoptimizer. If a proposed support is infeasible, a MILP minimizes support
   replacements before reoptimizing weights.

The continuous contract is `portfolio-allocation.v1`. It enforces budget,
holding count, minimum active weight, single-fund cap, asset-class ranges, R5,
manager, secondary-type, high-similarity-group, and profile performance proxy
constraints. High-similarity groups use deterministic complete-linkage
clustering at the document's 0.85 cosine threshold; this prevents transitive
chain edges from merging mutually dissimilar assets into one artificial group.

The return, volatility, and drawdown limits use the source document's available
multi-period indicators as linear proxies. They are explicitly labeled as
proxies in the contract and result audit; they are not covariance-based
portfolio volatility or NAV-path maximum drawdown.
