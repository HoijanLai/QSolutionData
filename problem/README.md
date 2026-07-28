# Problem set

这个目录用 `problem-case.v1` 统一管理一个业务问题的不同表示。`cbqm.v1`
和 `qubo.v1` 仍然是各自独立、严格的数学契约；NetworkX 图是同一 case 下的
artifact，而不是一个尚未定义语义的“万能图问题”。

```text
ProblemCase
├── artifacts
│   ├── cbqm.v1
│   ├── qubo.v1
│   ├── networkx.cbqm-factor.v1
│   ├── networkx.qubo-interaction.v1
│   └── networkx.qubo-maxcut.v1
└── tasks
    ├── canonical_artifact_id
    └── best_known
```

这样做有两个直接好处：

- 图可以先入库，之后再决定它承载 MaxCut、MIS 或其他业务任务。
- 同一 case 可以有多个 task，每个 task 独立维护目标方向和 best-known；
  `exact=True` 只锁定对应 task。

## 文件与组件

```text
problem/
├── problem_def.py       # ProblemCase、artifact、task、best-known
├── reader.py            # UTF-8 JSON 读写和原子保存
├── updater.py           # canonical objective/feasibility 与 best-known 更新
├── graph_codec.py       # NetworkX node-link JSON
├── transforms.py        # CBQM/QUBO 的可逆图表示与 signed MaxCut
├── case_operations.py   # 编译、派生、投影和 lineage 校验
├── solving.py           # ProblemCase 到原生 QUBO solver 的执行桥
└── data/
    ├── problem1/
    └── problem2/
```

每个数据目录包含 `case.json` 和独立的 `artifacts/*.json`。更新 task 的
best-known 不需要重写大图。派生 artifact 会记录直接父节点、转换器、版本、
配置、上下文和 payload 哈希。

## 建立 case 与派生表示

```python
from problem import (
    ProblemArtifact,
    ProblemCase,
    TaskDefinition,
    compile_case_qubo,
    derive_cbqm_factor_graph,
    derive_qubo_interaction_graph,
)

case = ProblemCase(
    problem_id='portfolio-001',
    artifacts=(
        ProblemArtifact(
            artifact_id='cbqm',
            representation='cbqm.v1',
            payload=cbqm,
        ),
    ),
    tasks=(
        TaskDefinition(
            task_id='allocation',
            canonical_artifact_id='cbqm',
            task_type='portfolio-allocation',
            sense='minimize',
        ),
    ),
    primary_artifact_id='cbqm',
)

case = derive_cbqm_factor_graph(case, 'cbqm')
case = compile_case_qubo(
    case,
    'cbqm',
    {'default_penalty': 10.0},
    target_artifact_id='qubo-p10',
)
case = derive_qubo_interaction_graph(case, 'qubo-p10')
```

图表示的语义：

- `networkx.cbqm-factor.v1`：变量与 objective/constraint factor 的二部图，
  可还原 `cbqm.v1`。
- `networkx.qubo-interaction.v1`：对角 bias 存在节点，二次 bias 存在边，
  可还原 `qubo.v1`。
- `networkx.qubo-maxcut.v1`：带 anchor 的 signed MaxCut 编码，满足
  `qubo_energy = qubo_offset - cut_value`。

原生图也可以先创建为无 task 的 case：

```python
import networkx as nx

from problem import create_graph_case

graph = nx.Graph()
graph.add_edge('fund-a', 'fund-b', weight=0.8)
case = create_graph_case('fund-relations-001', graph)
assert case.tasks == ()
```

## 运行 solver

Solver 本身继续只消费 `qubo.v1`，不感知 `ProblemCase`。应用层通过
`solve_problem_task()` 显式选择 task 和 QUBO artifact：

```python
from lib.solvers.qubo import ExactQuboSolver
from problem import solve_problem_task

record = solve_problem_task(
    case,
    task_id='allocation',
    artifact_id='qubo-p10',
    solver=ExactQuboSolver(),
    config={'max_variables': 24},
    update_best=True,
    exact_verification_max_variables=24,
)

case = record.case
print(record.canonical_solution)
print(record.canonical_objective_value)
print(record.exact_for_task)
```

执行桥负责：

- 在调用 solver 前严格验证 QUBO；
- 隔离 problem/config，验证返回的 `qubo-result.v1`；
- 保存 artifact hash、solver config、原始结果和 canonical 结果；
- 将直接编译的 QUBO sample 投影回 CBQM，并重新验证可行性和原目标；
- 按请求更新对应 task 的 best-known。

`qubo-result.v1` 的 `status='optimal'` 是 solver 的结论，不是通用结果校验器
能够自行证明的事实。无论直接 QUBO 还是编译后的 CBQM，执行桥都要用精确数值
重新枚举所选 QUBO，确认没有更优 assignment 后，才允许把结果写成 task 的
`exact=True`。`exact_verification_max_variables` 是这次独立复核的规模上限；
超过上限时仍可保留候选解，但会保守地保持非 exact。

CBQM 的 exact 传播还需要额外的编译证明。只有以下条件全部成立才会设置
`exact_for_task=True`：

1. solver 返回 `status='optimal'`，且执行桥独立穷举所选 QUBO 后确认全局最优；
2. 编译器的格点、浮点聚合和零项过滤证书均为 exact；
3. 使用记录的配置重新运行 canonical compiler，QUBO 和上下文完全一致；
4. 当前 sample 的所有 penalty 方程（包括 slack）残差为零；
5. QUBO energy 与原 CBQM objective 的方向映射以精确数值算术成立。

因此，penalty 太小、约束舍入、丢弃微小项、伪造 context 或错误 slack 都不会
把一个候选解升级成原业务问题的 exact solution。通过 `compiler=` 注入的其他
编译器仍可用于求解，但在没有对应可信重验证器前会保守地保持非 exact。

## Best-known

```python
from problem import BestKnownSolution, update_best_known

candidate = BestKnownSolution(
    solution=[1, 0],
    objective_value=-2.0,
    exact=True,
    source='external-certified-optimum',
)
update = update_best_known(case, 'allocation', candidate)
case = update.case
```

`cbqm.v1` 与 `qubo.v1` 使用内置 evaluator 先验证完整数学契约，再以精确
JSON-number 算术重新计算 objective；CBQM 还会验证 fixed values 和全部约束。
二进制向量只接受真正的 Python 整数 `0/1`，不会把 `False` 或 `1.0` 静默
归一化。其他 task 必须显式传入 `evaluator=`。

已有 exact incumbent 时，普通更新返回 `reason='exact_solution_locked'`。
更换 canonical artifact 的 payload 前必须先清除关联 task 的 incumbent，
避免旧 exact 证据锁住新模型。

## 读取与保存

```python
from problem import load_problem_case, load_problem_set, save_problem_case

case = load_problem_case('problem/data/problem2')
all_cases = load_problem_set('problem/data')
save_problem_case(case, 'problem/data/problem2')
```

JSON 使用 UTF-8，禁止 NaN/Infinity；manifest 最后以原子替换方式写入。
Artifact path 必须保持在所属 case 目录内。
