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

## 推荐文件组织

```text
lib/solvers/qubo/
├── for_human.md
├── exact_solver.py
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

小规模问题应与穷举最优值比较。

## 参考文件

- `lib/contracts/solver_protocol.py`
- `contracts/schemas/qubo.v1.schema.json`
- `contracts/schemas/qubo-result.v1.schema.json`
- `contracts/examples/qubo.v1.example.json`
- `contracts/examples/qubo-result.v1.example.json`

如果算法原生接受 MaxCut，可以通过 `QRBnBRSolverAdapter` 接入；如果算法直接接受 QUBO，则直接实现这里的标准签名，不需要经过 MaxCut。
