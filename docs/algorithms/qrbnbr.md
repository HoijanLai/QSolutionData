# Q-RBnBR MaxCut 复现

`QrbnbrQuboSolver` 是 QSolutionData 第一条“论文 → 问题脚本 → solver 脚本
→ notebook → 独立 oracle”复现路线中的 quantum-relaxation 分支。它消费
`qubo.v1`，返回 `qubo-result.v1`。对应的 classical control 是
`GwBranchAndBoundQuboSolver`。

## 为什么仍然使用 QUBO 契约

论文研究的是非负权 MaxCut。该目标可以无损写为：

```text
cut(x) = Σ w_uv [x_u != x_v]
E(x)   = offset - cut(x)
```

对应的 QUBO 系数为：

```text
Q_uu += -w_uv
Q_vv += -w_uv
Q_uv +=  2 w_uv
```

solver 会检查每个二次系数是否对应严格正权边，以及每个对角系数是否等于负的
weighted degree。一般 QUBO 会被拒绝，避免把 signed/anchored MaxCut 工程扩展
误称为论文原问题。

## 伪代码级入口

`_run` 只保留论文主流程：

```text
读取 MaxCut
初始化 edge-parity tree
while 仍有节点且预算允许:
    取一个节点
    消去 parity 约束
    如果 upper bound 不可能改善 incumbent: prune
    如果子问题足够小: exact close
    否则:
        p=1 QAOA -> correlation
        QRR sign rounding -> candidate
        branching matrix -> variable pair
        创建 same / opposite 两个 parity 分支
返回结果与搜索统计
```

QAOA statevector、correlation、QRR、selective composition、Union-Find parity、
变量消元、bound、分支规则和 result metadata 都封装在 protected 方法中。

## 最优性的来源

QRR 是启发式 incumbent 和分支信息来源，不负责证明最优。若没有达到
`timeout_seconds` 或 `max_nodes`：

1. every branch 固定一对 component 的 same/opposite parity；
2. 每次 union 至少减少一个自由 component；
3. 小叶子由 exhaustive enumeration 精确关闭；
4. 其余节点只由 admissible Laplacian bound 剪枝；
5. frontier 为空后才能返回 `optimal`。

达到时间限制时返回 `timeout`；达到节点限制且已有候选时返回 `feasible`，不会
伪造最优性声明。

## 使用

```python
from lib.solvers.qubo import QrbnbrQuboSolver
from problem.reproductions import build_qrbnbr_s1_instance


problem = build_qrbnbr_s1_instance(
    variable_count=10,
    edge_probability=0.45,
    seed=2025,
)
result = QrbnbrQuboSolver().solve(
    problem,
    {
        'branching_rule': 'r1',
        'branching_matrix': 'correlation',
        'brute_force_threshold': 4,
    },
)
```

完整交叉验证见 `notebooks/06_qrbnbr_reproduction.ipynb`。论文来源、固定源码
revision、当前偏差和未复现范围见 `problem/reproductions/README.md`。
QRR/GW 同问题对照见 `notebooks/08_qrr_vs_gw_bnb_reproduction.ipynb`。
