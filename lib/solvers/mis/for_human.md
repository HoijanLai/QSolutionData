# MIS Solver 开发指南

这个子模块原生处理 Maximum Independent Set（最大独立集）与 Maximum
Weight Independent Set（最大权独立集）。它消费 `mis.v1` 图并返回
`mis-result.v1`，不会先把图编译成 QUBO。

## 契约边界

```text
mis.v1
  -> MisSolver.solve(problem, config=None)
mis-result.v1
```

- 顶点以从 0 开始的连续索引稳定标识，同时保留真实名称。
- 边必须写成 `u < v`，去重并按字典序排列。
- `maximum-cardinality` 使用单位权重，顶点不得携带 `weight`。
- `maximum-weight` 要求每个顶点显式携带有限数值 `weight`。
- 解使用 `vertex-index-set.v1`：严格递增、无重复的顶点索引列表。
- `fixed_values` 是 hard assignments，不是初始化提示。

`BaseMisSolver` 拥有公共不变量：输入验证、防御性复制、计时、候选语义重算、
trace 规范化和最终 result 验证。具体算法的 `_run()` 只能提交候选顶点及声明，
不能自行伪造 objective、cardinality、total weight 或 feasibility。

## 已实现 solver

### ExactMisSolver

公开入口可以概括为：

```text
检查规模上限
-> 准备 fixed domain
-> 安全约简
-> 上界剪枝
-> include/exclude 分支
-> 返回 proof/bounds
```

它使用 `Fraction` 做内部目标比较，支持最大基数与最大权重，采用“目标值优先，
顶点索引集字典序次之”的确定性 tie-break。负权可选点可安全移除；零权点仍然
保留在搜索中，因为它可能改变规范 tie-break 见证。`max_vertices` 和
`timeout_seconds` 是显式安全控制。

Exact 返回的 proof 是 solver 自身声明，`independently_verified` 保持 `false`。
ProblemCase 只有在应用层另行穷举 canonical `mis.v1` 后，才会把 task 的
best-known 提升为 `exact=True`。

### GreedyMisSolver

公开入口保持四段式：

```text
准备 fixed domain
-> 确定性贪心构造
-> 一点/二点局部交换
-> 返回 feasible incumbent
```

贪心分数使用“精确权重 / 当前邻域大小”，再用权重、度数和顶点索引稳定破平局。
局部搜索只接受目标严格改善，避免零权平局循环。`max_local_passes` 控制局部轮数，
`max_pair_evaluations` 限制每轮二点交换扫描；扫描未穷尽时不会误报局部最优。

Greedy 始终返回 `feasible` 而不是 `optimal`。命中 Exact 的同一见证不等于拥有
最优性证明。

## 验证

自动化测试覆盖：

- 与独立暴力 oracle 的 Exact 对照；
- cardinality、正/零/负权重和 fixed values；
- fixed-in 边冲突的不可行证书；
- timeout 与规模保护；
- Greedy 的确定性、局部改善和扫描上限；
- result 契约、输入不可变性与可选依赖隔离；
- ProblemCase 原生执行、best-known 更新和独立 exact promotion。

可交互的小规模演示位于
`notebooks/04_mis_solver_validation.ipynb`。Notebook 不实现 solver、
objective、feasibility 或穷举 helper。

由相关矩阵或业务数据生成 MIS 图属于 problem builder/preprocessing；NetworkX
互转属于 adapter。执行搜索、reduction 或量子子程序的代码才放在这里。
