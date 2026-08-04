# CBQM Solver 开发指南

这个子模块放置直接消费 `cbqm.v1`、在原始可行域上工作的 solver。若算法先把
约束编译成 penalty QUBO，它仍属于 `solvers/qubo/`；CBQM 到 QUBO 的转换必须由
显式 compiler 负责并保存 lineage。

## 已实现边界

原生调用链是：

```text
cbqm.v1
  -> CbqmSolver.solve(problem, config)
cbqm-result.v1
```

`BaseCbqmSolver` 固定以下流程：

```text
validate
  -> defensive copy
  -> resolve config
  -> _run
  -> recompute original objective and feasibility
  -> validate cbqm-result.v1
```

算法 kernel 只返回 `CbqmSolveOutcome`，不能提供受信任的
`best_objective` 或 `feasibility`。Trace 也只允许 kernel 提供 sample 与时间信息，
公开 objective/feasible 字段由 Base 层重算。

## ExactCbqmSolver

`ExactCbqmSolver` 是小规模 correctness oracle：

- 只枚举 `fixed_values` 之外的自由变量；
- 直接检查原始 fixed values 与线性约束；
- 用 `Fraction` 比较原始线性/二次 objective；
- min/max 均支持；
- objective 相同时选择字典序最小 sample；
- 搜索穷尽后返回 `optimal` 或 `infeasible`；
- timeout 只保留已经发现的可行 incumbent，不伪造 proof；
- `max_variables` 默认 24，是防止误跑指数搜索的安全栏。

Exact solver 的 proof 中 `independently_verified` 仍为 `false`，因为那是 solver
自身产生的证据。`ProblemCase` 只有在
`solve_native_problem_task()` 独立枚举 canonical CBQM 后，才会把 task
best-known 升级为持久化 `exact=True`。

## LocalSearchCbqmSolver

`LocalSearchCbqmSolver` 是不依赖 penalty 的原生启发式基线：

```text
prepare fixed values and exact constraints
  -> create canonical/random start
  -> repair constraint violations
  -> improve the original objective through feasible moves
  -> retain the best feasible incumbent
```

- repair 阶段只比较 violated constraint 数量与 exact violation magnitude；
- objective 不参与 repair 排序，避免目标值掩盖约束违反；
- 同时扫描一变量与二变量翻转，因此能在 one-hot 等离散可行域中移动；
- `seed`、`max_restarts`、repair/local step 上限与 pair scan 上限均可配置；
- 找到的候选只声明 `feasible`，即使恰好等于 exact optimum 也不伪造 proof；
- repair 预算耗尽且没有可行候选时返回 `unknown`，绝不据此声称
  `infeasible`；
- timeout 会保留已经找到的可行 incumbent，并只报告 primal bound。

## 新 solver 的文件风格

每个算法的公开 `_run()` 应像伪代码一样只保留主流程。数值计算、候选比较、
deadline、backend 适配与结果片段构造放进 protected helpers，并为下列边界写测试：

- 输入与 config 不可变；
- invalid config 在 backend 执行前失败；
- 原始 objective 与 feasibility 被 Base 层重算；
- timeout、infeasible 与 backend error 不冒充异常输入；
- solver-reported `optimal` 不自动成为 persistent exact；
- 浮点大数不会通过相对容差掩盖真实 objective 差异。

小规模可执行示例见
`notebooks/03_cbqm_native_solver_validation.ipynb`。
