# MIS Solver 开发指南

这个子模块用于原生处理 Maximum Independent Set（最大独立集）图问题，
例如 qReduMIS 或直接面向 Rydberg/QAOA 的 MIS 算法。它解决的是节点选择，
不直接解决当前 CBQM 中的离散资产权重配置。

当前项目还没有 `mis.v1`、`mis-result.v1` 或 `MisSolver` 专用 Protocol。
因此这里先作为明确的代码边界，不应复用 `qubo-result.v1` 或把任意 QUBO
误标为 MIS。

开始实现生产 solver 前应先约定：

1. `mis.v1` 的稳定 vertex 顺序、名称和规范化边表示。
2. 问题是 maximum-cardinality MIS 还是 maximum-weight MIS。
3. `mis-result.v1` 的 sample、cardinality/weight 与 feasibility 语义。
4. `MisSolver` Protocol、validator、example 和 contract tests。

由相关矩阵阈值化生成市场图属于 problem builder/preprocessing；图表示适配
属于 `lib/adapters/`。只有执行 MIS 搜索、reduction 或量子子程序的代码才
放在这里。现有 `QRBnBRSolverAdapter` 是 QUBO-to-MaxCut 路径，不是 MIS
solver。

