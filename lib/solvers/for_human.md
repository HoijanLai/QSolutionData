# Solver 子模块

这个目录按照 solver 原生接受的问题表示组织算法实现：

```text
lib/solvers/
├── qubo/   # qubo.v1
├── cbqm/   # 直接处理带约束二次模型
└── mis/    # 直接处理最大独立集图
```

三个子模块共享调用形状：

```python
def solve(problem, config=None):
    ...
```

`lib.contracts.Solver[ProblemT, ResultT]` 只约定这一调用形状。每个子模块
仍须使用自己的版本化输入、结果协议；相同的方法名不代表不同问题表示
可以混用。

当前只有 QUBO 路径拥有完整的输入和结果协议，即
`qubo.v1 -> qubo-result.v1`。CBQM 和 MIS 子模块先建立清晰的代码归属，
在各自结果 schema 与专用 Protocol 确定前不应放入生产 solver。

公共 wrapper 负责数据检查、配置解析、计时、主要逻辑分支和结果构造；
计算、搜索、分支、松弛与 rounding 等元逻辑放在保护式命名函数中。这与
项目 EDA 的代码风格保持一致。

选择目录时以 solver 的原生输入为准：

- 接受 `qubo.v1`：放入 `qubo/`。
- 直接接受 `cbqm.v1` 并在算法内部维护约束：放入 `cbqm/`。
- 直接接受图并求最大独立集：放入 `mis/`。
- 只是做表示转换而不负责求解：放入 `lib/adapters/` 或
  `lib/compilers/`，不要放入 `solvers/`。

