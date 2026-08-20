# 论文复现问题层

这个目录把“论文中的实验问题”与通用数学 benchmark 分开管理。通用 benchmark
用于长期回归；这里的实例用于回答更具体的问题：某篇论文的 solver 是否在其声明
的实验范围内被复现。

每条复现路线至少包含三层：

```text
problem/reproductions/<route>.py   确定性问题定义与来源 metadata
lib/solvers/<contract>/<route>.py  算法实现与 canonical solver 边界
notebooks/<route>.ipynb            配置、调用、对照与结果展示
```

Notebook 不定义算法、objective、复杂 helper 或问题生成器。小规模真实最优值应由
`tests/oracles` 或独立 exact solver 交叉验证。

## Q-RBnBR 首个验收案例

| 项目 | 当前实现 |
|---|---|
| 论文问题 | S1 风格、少于 20 个节点的随机无权 MaxCut |
| 输入契约 | `qubo.v1`，严格识别 `offset - cut` |
| 输出契约 | `qubo-result.v1` |
| 量子松弛 | 理想 p=1 NumPy statevector QAOA correlation |
| rounding | QRR：逐个 sign-round correlation eigenvector |
| classical baseline | CVXPY GW SDP + seeded random hyperplane rounding |
| 搜索树 | same/opposite edge-parity branches |
| 分支规则 | R1、R2、R3 |
| QRR 分支矩阵 | raw correlation、selective composition |
| GW 分支矩阵 | SDP matrix |
| bound | `u=0` 的 admissible Laplacian 最大特征值 bound |
| leaf closure | 可配置规模的 exhaustive enumeration |
| 最优性 | bound pruning 后 edge-parity tree 穷尽 |

当前可直接对照的 solver 是：

```text
GoemansWilliamsonQuboSolver  standalone classical approximation
QrbnbrQuboSolver             QRR-informed parity BnB
GwBranchAndBoundQuboSolver   GW-informed parity BnB
```

固定来源：

- 论文：`Quantum Relaxation Informed Branch-and-Bound Algorithm — An
  Application to Max-Cut`
- 公开仓库：`https://github.com/HoijanLai/Q-RBnBR`
- 审计 revision：`ec72c202559655dc170f8bdf41f2936107ce94f8`

当前没有声称复现以下实验层内容：

- 原论文使用的具体随机图样本或 BiqMac 数据；
- Qiskit 1024-shot sampling、真实硬件或噪声；
- 原仓库 CVXOPT 与当前 CLARABEL/SCS 的逐迭代数值轨迹一致性；
- 优化后的 diagonal correction；
- empirical/non-admissible bound 与 random-pass 变体；
- S2 的 60–100 节点规模结论。

因此 notebook 验证的是“方法链路和小规模最优性”，不是对论文所有图表数值的
逐点复刻。后续路线应继续使用同一结构，并把“已复现、工程扩展、尚未复现”分别
写进 metadata 和文档。

## SCMF-QAOA 首个验收案例

| 项目 | 当前实现 |
|---|---|
| 论文问题 | 零场、完全图、未缩放 Gaussian SK spin glass |
| 输入契约 | `qubo.v1`，metadata 保留论文 Ising Hamiltonian |
| 输出契约 | `qubo-result.v1`，启发式状态固定为 `feasible` |
| 分解 | seeded balanced random partitions |
| 量子子问题 | 理想 NumPy statevector QAOA |
| 参数 | 所有子问题共享 `2p` 个角度 |
| environment | seeded random-order asynchronous self-consistency |
| 外层优化 | dependency-light Nelder--Mead |
| 零场对称性 | 固定最后一个 spin 为 `+1`，再恢复完整 sample |
| 对照 | Exact、完整 QAOA、environmentless independent subproblems |

固定来源：

- 论文：`Self-consistent mean-field quantum approximate optimization`
- 版本：`arXiv:2603.09838v1`
- 当前未发现作者公开代码；实现依据论文公式独立完成

本阶段复现的是小规模理想模拟器上的算法结构、式 (8) 一体期望值、`K=1`
退化性质、自洽收敛与采样链路。当前没有声称复现论文的完整规模曲线、解析
back-propagation 加速、252-variable molecular docking、classical clique repair、
Rigetti Ankaa-3 硬件结果或噪声缓解。
