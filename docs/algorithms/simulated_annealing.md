# Simulated Annealing QUBO solver

`SimulatedAnnealingQuboSolver` 是一个纯本地、基于 NumPy 的参考实现。它直接消费
`qubo.v1`，并通过 `BaseQuboSolver` 返回经过统一校验的 `qubo-result.v1`：

```text
qubo.v1
  -> validate and copy
  -> Metropolis simulated annealing
  -> select an observed incumbent
  -> recompute canonical QUBO energy
  -> qubo-result.v1
```

它是后续 QAOA、量子退火和混合求解器的经典启发式基线，不是最优性证明器。

## 算法

对于二进制样本 \(x\)，仓库的标准 QUBO 能量为

\[
E(x)=\mathrm{offset}+\sum_{(i,j,q)\in\mathrm{terms}}q\,x_i x_j.
\]

每个 read 从随机或用户指定的二进制样本开始。一次 sweep 按顺序或随机排列访问全部
变量，并为每个变量提出一次单比特翻转。若翻转带来的能量变化为
\(\Delta E\)，Metropolis 规则为：

- \(\Delta E\leq 0\)：接受；
- \(\Delta E>0\)：以 \(\exp(-\beta\Delta E)\) 的概率接受。

\(\beta\) 是逆温度，随 sweep 单调增加。较小的 \(\beta\) 允许搜索跨过局部能垒，较大的
\(\beta\) 更倾向于保留低能量状态。实现预先建立稀疏邻接结构，并用局部
`delta` 更新判断翻转；QUBO 的常量 offset 在能量差中抵消，不参与每次翻转计算。
所有 read 中实际观察到的最好样本成为 incumbent。求解器不会构造、修补或返回未曾
观察到的样本。

内部 Metropolis 计算可以使用浮点数，但公共结果中的 `best_energy` 不信任内部累计值。
`BaseQuboSolver` 会针对原始 `qubo.v1` 重新计算标准能量，并保留大整数 offset 等契约
边界上的精确语义。

## 使用

```python
from lib.solvers.qubo import SimulatedAnnealingQuboSolver

result = SimulatedAnnealingQuboSolver().solve(
    problem,
    {
        "num_reads": 32,
        "sweeps": 1000,
        "seed": 1729,
    },
)
```

公共入口只有 `solve(problem, config=None)`。输入 `problem` 和嵌套的 `config` 都不会被
修改。未知配置字段、错误类型、非法样本或不合法的 schedule 会在算法运行前抛出
`TypeError` 或 `ValueError`。

## 配置

| 字段 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `num_reads` | `int` | `32` | 相互独立的退火运行次数，必须大于零。 |
| `sweeps` | `int` | `1000` | 每个 read 的 sweep 数，必须大于零。 |
| `beta_schedule_type` | `str` | `"geometric"` | `"linear"`、`"geometric"` 或 `"custom"`。 |
| `beta_start` | `float \| None` | `None` | 自动 schedule 的起始逆温度；`None` 时由问题系数确定性估计。 |
| `beta_end` | `float \| None` | `None` | 自动 schedule 的结束逆温度；`None` 时由问题系数确定性估计。 |
| `beta_schedule` | `list[float] \| None` | `None` | custom schedule；长度必须等于 `sweeps`，元素非负且单调不减。 |
| `seed` | `int` | `0` | NumPy 随机数生成器的非负种子。相同输入与配置产生相同算法结果。 |
| `initial_samples` | `list[list[int]] \| None` | `None` | 可选二进制初态；不足 `num_reads` 时循环复用。 |
| `update_order` | `str` | `"random"` | 每个 sweep 使用 `"random"` 排列或 `"sequential"` 顺序。 |
| `timeout_seconds` | `float \| None` | `None` | 正数 deadline；检查到超时时保留已经验证的 incumbent。 |
| `trace_interval` | `int \| None` | `None` | 正整数时，按指定 sweep 间隔记录 incumbent trace。 |

`linear` 和 `geometric` schedule 使用 `beta_start`、`beta_end` 与 `sweeps`。`custom`
模式必须显式提供 `beta_schedule`，并且不能用不一致的端点参数代替它。自动估计只依赖
问题系数，因此不会引入额外随机性；结果元数据记录 schedule 类型和实际端点，完整
custom 序列仍由调用配置或 `CaseSolveRecord` 保存。

## 结果与证明语义

- 正常完成返回 `feasible`。
- deadline 到达返回 `timeout`，并保留已知合法候选。
- 该求解器**永远不返回 `optimal`**。即使命中了精确最优解，命中本身也不是证明。
- 无约束 QUBO 总有二进制 assignment，因此本地 SA 不使用 `infeasible` 表示搜索失败。
- `best_sample` 是索引顺序下的 0/1 列表；真实变量名仍由输入
  `variable_names` 跟踪。
- `best_energy` 由公共 wrapper 根据原问题重新计算，而不是直接相信退火 kernel。
- 固定 seed 保证算法拥有的字段可重复；`runtime_seconds` 来自墙钟计时，不属于逐位相同
  的可重复字段。

若需要证明最优性，应独立运行 `ExactQuboSolver` 或其他具有完整证明语义的求解器。
“SA energy 等于 exact energy”只证明本次候选碰巧最优，不会把 SA 的状态升级为
`optimal`。

## 验证

测试与 `notebooks/05_simulated_annealing_validation.ipynb` 使用
`problem.benchmarks.build_annealing_validation_suite()` 提供的小规模确定性 fixtures，
其中包含零变量、正负单变量 bias、退化最优解、受挫三角形、稀疏 QUBO、大整数 offset
与 custom schedule 案例。

验证分为相互独立的几层：

1. `tests.oracles` 使用 `Fraction` 穷举小规模 QUBO，不调用 production energy helper；
2. SA 返回的 sample 用独立 oracle 再次计算能量；
3. `ExactQuboSolver` 提供仓库 solver 协议下的精确对照；
4. 合约验证器检查 `qubo-result.v1`、sample 长度、能量和 problem ID；
5. 固定 seed 运行两次，比较除墙钟 runtime 外的算法字段；
6. 单元测试逐个 bit 比较局部 `delta` 与完整能量差，并通过可控 deadline seam 测试
   timeout，不在 notebook 中依赖脆弱的真实时间。

SA 不保证在有限预算内命中最优，因此正确断言是
`sa_energy >= exact_energy`，而不是强制 gap 为零。

## 依赖与限制

- 必需依赖只有 NumPy；不需要量子 SDK、云端凭证或付费硬件。
- 本实现没有把 `dwave-samplers` 作为依赖，也不是其 adapter 或代码副本。
- 固定 seed 只承诺本实现中相同版本、相同输入和相同配置的重复性，不承诺与第三方
  sampler 产生相同轨迹。
- 邻接 delta 避免每次提案都重算完整 QUBO，但稠密问题每个 sweep 仍可能达到二次成本。
- 极端尺度的浮点系数会影响 Metropolis 接受概率；最终公共能量虽然会重新计算，但这不
  能修复已经受到数值尺度影响的搜索轨迹。
- SA 是启发式基线，不提供近似比、下界、上界或最优性证书。

配置命名与 schedule 语义参考了 D-Wave 官方
[`SimulatedAnnealingSampler.sample`](https://docs.dwavequantum.com/en/latest/ocean/api_ref_samplers/generated/dwave.samplers.SimulatedAnnealingSampler.sample.html)
接口，尤其是 reads、sweeps、逆温度 schedule、initial states 和 update order 的概念。
本仓库仍是独立的本地实现，参数默认值、随机轨迹和返回协议以本项目文档与测试为准。
