# Mathematical benchmark catalog

这里存放完全脱离业务语义的、可重复生成的 solver 微型基准。每个生成器直接
返回项目的 canonical contract：

```text
QUBO family -> qubo.v1
CBQM family -> cbqm.v1
MIS family  -> mis.v1
```

它们不读取 EDA、资产数据或本地文件，也不需要业务 adapter。固定 seed 只属于
数学生成过程；每次调用都会返回新的 JSON-native 文档，调用方可以安全修改。

## 当前问题族

| Name | Contract | Mathematical model | Size | Exact objective | Audited wall time |
|---|---|---|---:|---:|---:|
| `qubo.ea-grid-4x4` | `qubo.v1` | Edwards–Anderson ±J grid | 16 variables | -20 | 2.252 s |
| `qubo.sk-15` | `qubo.v1` | Sherrington–Kirkpatrick ±J | 15 variables | -35 | 3.087 s |
| `qubo.planted-consistency-18` | `qubo.v1` | planted unique optimum | 18 variables | 0 | 12.415 s |
| `qubo.ferromagnetic-cycle-18` | `qubo.v1` | degenerate ferromagnetic cycle | 18 variables | 0 | 11.677 s |
| `qubo.frustrated-cycle-17` | `qubo.v1` | frustrated odd antiferromagnet | 17 variables | 1 | 5.647 s |
| `cbqm.tsp-4` | `cbqm.v1` | cyclic travelling salesperson | 16 variables, 1 fixed | 14 | 1.011 s |
| `cbqm.knapsack-16` | `cbqm.v1` | 0/1 knapsack | 16 variables | 115 | 5.326 s |
| `cbqm.exact-cover-8x12` | `cbqm.v1` | exact cover | 12 variables | 2 | 0.145 s |
| `cbqm.graph-coloring-5x3` | `cbqm.v1` | one-hot graph colouring | 15 variables, 1 fixed | 4 | 0.463 s |
| `mis.path-32` | `mis.v1` | path | 32 vertices | 16 | 0.457 s |
| `mis.cycle-31` | `mis.v1` | odd cycle | 31 vertices | 15 | 0.358 s |
| `mis.clique-36` | `mis.v1` | complete graph | 36 vertices | 1 | 0.013 s |
| `mis.complete-bipartite-12x12` | `mis.v1` | complete bipartite graph | 24 vertices | 12 | 0.003 s |
| `mis.grid-5x6` | `mis.v1` | rectangular grid | 30 vertices | 15 | 0.069 s |
| `mis.weighted-path-30` | `mis.v1` | weighted path with hard values | 30 vertices | 69 | 0.006 s |
| `mis.erdos-renyi-28-p022` | `mis.v1` | fixed-seed G(n,p) | 28 vertices | 12 | 0.045 s |

Wall times were measured on the development machine with the repository
`.venv` on 2026-07-31. They are not portable performance promises. The sizing
policy is deliberately conservative: each instance carries an intended
per-instance Exact budget of 1800 seconds, while the complete audited catalog
finished in about 43 seconds on that machine.

## 使用

读取全部 canonical problems：

```python
from problem.benchmarks import build_mathematical_benchmark_suite

suite = build_mathematical_benchmark_suite()
tsp = suite['cbqm.tsp-4']
spin_glass = suite['qubo.ea-grid-4x4']
```

读取已核验的规范最优见证：

```python
from problem.benchmarks import build_mathematical_benchmark_references

references = build_mathematical_benchmark_references()
known_energy = references['qubo.ea-grid-4x4']['objective_value']
known_sample = references['qubo.ea-grid-4x4']['solution']
```

直接获得带 exact best-known 的 `ProblemCase`：

```python
from problem.benchmarks import build_mathematical_benchmark_cases

cases = build_mathematical_benchmark_cases()
case = cases['mis.grid-5x6']
assert case.get_task('optimize').best_known.exact
```

在当前机器重新运行 representation-native Exact 审计：

```powershell
python -m problem.benchmarks
python -m problem.benchmarks qubo.sk-15 cbqm.tsp-4 --json
```

CLI 的 `--timeout-seconds` 是每个实例的上限。QUBO 与 CBQM 使用完整二进制
枚举；MIS 使用原生 branch-and-reduce。任何 timeout 都会使 CLI 返回非零。

## 数学约定

- Spin glass 使用
  `H(s) = -Σ J_ij s_i s_j - Σ h_i s_i`，再以 `s_i = 2x_i - 1`
  精确转换为 `qubo.v1`。原始 couplings 与 fields 保留在 metadata。
- TSP 使用城市×位置 one-hot 编码，并固定一个起点去掉旋转对称。
- Graph colouring 使用 vertex×colour one-hot 变量和线性冲突约束。
- Planted QUBO 是若干非负局部 penalty 之和；目标样本让每项同时为零。
- MIS 直接使用规范顶点索引和边表，不把图先转换成 QUBO。

## 扩展规则

新增基准时：

1. 只表达数学模型，不添加业务字段。
2. 输出已公开的 canonical contract；只有 native solver 真正需要新语义时才
   增加新 contract。
3. 随机实例必须有固定 seed、生成器版本和完整参数。
4. 保持在对应 Exact solver 的显式安全栏内，并运行完整 exact audit。
5. 将规范最优见证加入 `references.py`，再由 strict ProblemCase evaluator
   重算目标与可行性。
6. 大规模性能数据另建 suite；不要让日常正确性测试变成半小时 CI。
