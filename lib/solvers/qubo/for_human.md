# QUBO Solver 开发指南

这个子模块用于原生接受 `qubo.v1` 并返回 `qubo-result.v1` 的求解算法。

## Protocol 与 ABC

- `lib.contracts.QuboSolver` 是结构化 `Protocol`，用于类型检查和外部实现
  兼容性，不要求继承。
- `BaseQuboSolver` 是项目内推荐的模板方法 ABC，统一完成输入验证、防御性
  拷贝、计时、标准能量复算和 `qubo-result.v1` 构造。
- `QuboSolveOutcome` 是算法 kernel 返回给 ABC 的内部结果；算法不负责报告
  canonical energy，避免 Ising、MaxCut 或其他后端的能量约定泄漏。

新 solver 通常只需声明身份并实现 `_run`：

```python
from lib.solvers.qubo import BaseQuboSolver, QuboSolveOutcome


class MyQuboSolver(BaseQuboSolver):
    SOLVER_NAME = 'my-qubo-solver'
    SOLVER_VERSION = '0.1.0'
    BACKEND = 'local'

    def _run(self, problem, config):
        sample = _run_my_algorithm(problem, config)
        return QuboSolveOutcome(
            status='feasible',
            best_sample=sample,
            termination_reason='completed',
        )


def _run_my_algorithm(problem, config):
    ...
```

如需配置，重写 `_resolve_config`，先调用 `super()._resolve_config(config)`，
再检查允许字段并返回解析后的副本。不要重写公共 `solve`。

你通常不需要修改：

- `lib/portfolio/`：投资组合建模。
- `lib/compilers/`：CBQM 到 QUBO 的编译。
- `lib/adapters/`：Qiskit、Q-RBnBR 等框架适配。
- `contracts/`：稳定的输入输出协议。

## Solver 接口

每个 QUBO solver 应提供：

```python
def solve(problem: dict, config: dict | None = None) -> dict:
    ...
```

其中：

- `problem` 必须符合 `qubo.v1`。
- `config` 由具体 solver 自己定义，不得修改调用者传入的对象。
- 返回值必须符合 `qubo-result.v1`。

Solver 可以使用类，也可以使用函数。使用类时推荐：

```python
class MyQuboSolver:
    def solve(self, problem, config=None):
        ...
```

当前参考实现包括：

- `ExactQuboSolver`：小规模穷举，搜索耗尽后可以声明 `optimal`。
- `GoemansWilliamsonQuboSolver`：CVXPY SDP relaxation 与 seeded random
  hyperplane rounding，正常完成返回 `feasible`。
- `GwBranchAndBoundQuboSolver`：论文的 classical control；使用 GW SDP
  matrix 生成候选和分支信息，与 QRR-BnB 共享 parity tree 和 exact closure。
- `QaoaQuboSolver`：NumPy statevector QAOA，返回启发式 `feasible`。
- `ScmfQaoaSolver`：SCMF-QAOA 论文的独立复现；把完整 QUBO 转为 Ising、
  分成平衡子问题，以共享 QAOA 参数和自洽 mean-field environment 保留跨区
  影响，再对 product state 采样并返回 `feasible`。
- `QrbnbrQuboSolver`：面向非负权 MaxCut QUBO 的 Q-RBnBR 论文复现；p=1
  QAOA/QRR 指导 edge-parity tree，admissible bound 与完整 leaf closure
  决定是否可以声明 `optimal`。
- `SimulatedAnnealingQuboSolver`：NumPy Metropolis 退火，正常完成返回
  `feasible`，deadline 返回 `timeout`，永不自行声明 `optimal`。

## QUBO 能量约定

Solver 要最小化：

```text
E(x) = offset + sum(coefficient * x[i] * x[j])
```

输入中的每个 term 是：

```python
[i, j, coefficient]
```

约定：

- `0 <= i <= j < num_variables`
- `i == j` 表示线性 QUBO 系数
- `i < j` 表示二次系数
- 二次项只计算一次，没有隐含的乘二
- `x[i]` 只能是 `0` 或 `1`
- 返回的 `best_energy` 必须包含 `offset`

Solver 内部可以忽略 offset，因为它不改变最优解，但返回结果时必须加回来。
如果 solver 要返回 `status='optimal'`，内部的最优性比较不能先把所有 JSON
数值无条件转成 `float`。例如很大的整数 offset 与一个 `-1` bias 在 binary64
中可能看起来相等。项目内的 exact solver 使用精确有理数比较，公共 wrapper
也会用同一规则复算最终能量。

## 最小结果格式

```python
result = {
    'schema': 'qubo-result.v1',
    'problem_id': problem['problem_id'],
    'solver': {
        'name': 'my-solver',
        'version': '0.1.0',
        'backend': 'local',
    },
    'status': 'feasible',
    'best_sample': [0, 1, 0],
    'best_energy': -1.25,
    'runtime_seconds': 0.01,
}
```

常用 status：

- `optimal`：已经证明最优。
- `feasible`：找到候选解，但没有最优性证明。
- `timeout`：达到时间限制，可能仍然返回候选解。
- `infeasible`：证明不存在候选解。
- `error`：后端执行失败。
- `unknown`：无法给出更强结论。

这里的 `optimal` 是 solver 对自身算法终止条件的声明。`qubo-result.v1`
校验器能验证 sample 和 energy 一致，却不能从一个字符串推导出全局最优。
当结果进入 `ProblemCase` 时，`solve_problem_task()` 会在配置的变量上限内
独立穷举一次；只有这次复核也通过，才会把 best-known 标成 exact。

## 推荐文件组织

```text
lib/solvers/qubo/
├── for_human.md
├── exact.py
├── goemans_williamson.py
├── gw_bnb.py
├── qaoa.py
├── qrbnbr.py
├── simulated_annealing.py
├── your_solver.py
└── _utils.py
```

公开 wrapper 负责：

- 检查输入格式
- 解析 config
- 计时
- 处理主要逻辑分支
- 生成 `qubo-result.v1`

计算、搜索、分支规则、松弛和 rounding 等算法逻辑放在保护式函数中：

```python
def solve(problem, config=None):
    _validate_input(problem, config)
    started_at = time.perf_counter()
    sample, energy, metadata = _run_algorithm(problem, config or {})
    runtime = time.perf_counter() - started_at
    return _build_result(problem, sample, energy, runtime, metadata)


def _run_algorithm(problem, config):
    ...
```

这与项目现有 EDA 风格一致：检查和主要分支留在 wrapper，元逻辑放在保护式函数中。

## 开发时至少测试

每个 solver 至少需要验证：

1. 空配置可以运行。
2. 不修改 `problem` 和 `config`。
3. `best_sample` 长度等于 `num_variables`。
4. sample 只包含 `0/1`。
5. 重新计算的 QUBO energy 等于 `best_energy`。
6. offset 被正确恢复。
7. 固定随机种子时结果可复现。
8. timeout 和无候选解可以正常返回。
9. 大整数 offset 加很小 bias 时不会丢失最优解次序。

小规模问题应与穷举最优值比较。

## 参考文件

- `lib/contracts/solver_protocol.py`
- `contracts/schemas/qubo.v1.schema.json`
- `contracts/schemas/qubo-result.v1.schema.json`
- `contracts/examples/qubo.v1.example.json`
- `contracts/examples/qubo-result.v1.example.json`

如果算法原生接受 MaxCut，可以通过 `QRBnBRSolverAdapter` 接入；如果算法直接接受 QUBO，则直接实现这里的标准签名，不需要经过 MaxCut。
