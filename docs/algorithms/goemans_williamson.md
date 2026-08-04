# Goemans–Williamson 与 GW-BnB

本项目提供两个共享 `qubo.v1` MaxCut 边界的 classical solver：

- `GoemansWilliamsonQuboSolver`：standalone SDP approximation；
- `GwBranchAndBoundQuboSolver`：Q-RBnBR 论文中的 classical control。

## Standalone GW

公开入口保持四步结构：

```text
严格识别 non-negative weighted MaxCut QUBO
求解 max trace(LX)/4, diag(X)=1, X PSD
把 X 分解为顶点向量
seeded random hyperplanes -> best observed cut
```

正常完成返回 `feasible`。有限次 hyperplane rounding 即使命中 exact optimum，
也不能自行产生 `optimal` 证明。结果会报告 SDP relaxation、cut/SDP ratio、
backend、solver status、迭代数和 solver time。

经典近似保证针对 non-negative MaxCut 和随机 hyperplane 的期望行为。它不是
某一次有限采样结果的确定性证书，也不适用于 parity elimination 后可能出现的
signed reduced graph。

## GW-informed parity BnB

`GwBranchAndBoundQuboSolver` 与 `QrbnbrQuboSolver` 共享：

- parity Union-Find；
- same/opposite edge-parity branches；
- signed variable elimination 与 compensation；
- `u=0` Laplacian spectral bound；
- R1、R2、R3；
- BFS/DFS；
- exact leaf closure；
- timeout/node-limit 与 canonical result 语义。

唯一替换项是 relaxation provider：

```text
QRR-BnB: p=1 QAOA correlation -> QRR candidate / branching matrix
GW-BnB:  SDP matrix          -> hyperplane candidate / branching matrix
```

因此两条 BnB 的 `optimal` 都来自耗尽 admissibly-pruned parity tree，而不是
QRR 或 GW approximation claim。

## 数值 backend

当前 `.venv` 使用 CVXPY，默认选择 CLARABEL，也支持 SCS。公开研究仓库的 GW
实现显式请求 CVXOPT；本项目没有假装两种 backend 的浮点迭代轨迹完全相同，而是
把 backend、tolerance、status 和 iterations 写入 result。

## 使用

```python
from lib.solvers.qubo import (
    GoemansWilliamsonQuboSolver,
    GwBranchAndBoundQuboSolver,
)


gw_result = GoemansWilliamsonQuboSolver().solve(
    problem,
    {'rounds': 128, 'seed': 7, 'sdp_solver': 'CLARABEL'},
)
gw_bnb_result = GwBranchAndBoundQuboSolver().solve(
    problem,
    {
        'branching_rule': 'r1',
        'brute_force_threshold': 4,
        'rounds': 16,
        'seed': 7,
        'sdp_solver': 'CLARABEL',
    },
)
```

验证与方法对照分别见：

- `notebooks/07_goemans_williamson_validation.ipynb`
- `notebooks/08_qrr_vs_gw_bnb_reproduction.ipynb`
