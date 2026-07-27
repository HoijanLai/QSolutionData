# CBQM Solver 开发指南

这个子模块用于直接处理带显式约束的二进制二次模型。目标算法应原生维护
约束，例如 constraint-preserving QAOA、可行域 mixer、约束 oracle 或其他
不依赖 QUBO penalty compilation 的方法。

预期输入是现有 `cbqm.v1`：

- 二进制变量及稳定索引；
- 具有方向和 offset 的线性/二次目标；
- 有上下界的显式线性约束；
- 必须精确满足的 fixed values。

当前这里只是子模块边界，尚未定义 `cbqm-result.v1` 和 `CbqmSolver` 专用
Protocol。因此暂时不要把直接 CBQM solver 声明成 `QuboSolver`，也不要
返回 `qubo-result.v1`：QUBO 的 penalty energy 不等于原始 CBQM objective，
而且不能独立表达约束可行性。

开始实现生产 solver 前应先约定：

1. `cbqm-result.v1` 的 sample 顺序、objective 与 feasibility 语义。
2. 约束违反量及不可行候选的记录方式。
3. `CbqmSolver: cbqm.v1 -> cbqm-result.v1` Protocol。
4. 对应的 schema validator、example 和 contract tests。

如果算法实际上先调用 `compile_qubo`，它属于 `solvers/qubo/`；CBQM 到
QUBO 的转换仍由 `lib/compilers/` 负责，不应隐藏在本子模块内部。

