# Solver validation notebooks

这组 notebook 用于小规模、可重复的 solver 交叉验证。它们是自包含的，彼此不
共享内存状态，也不会读写 `problem/data`。

## 推荐顺序

1. `01_qubo_solver_validation.ipynb`
   - 手工构造并验证 `qubo.v1`
   - 调用 `tests/oracles/` 中独立维护的 `Fraction` 穷举器
   - 对比 `ExactQuboSolver` 与 `QaoaQuboSolver`
   - 验证 `qubo-result.v1`、输入不可变性和固定 seed 重复性
   - 演示坏 QUBO 与被篡改 result 如何在契约边界被拒绝

2. `02_cbqm_solver_validation.ipynb`
   - 手工构造并验证 `cbqm.v1`
   - 调用 `tests/oracles/` 独立枚举原始可行域和 objective
   - 建立 `ProblemCase` 并编译出 `qubo.v1`
   - 独立枚举派生 QUBO，确认本例 penalty 足够
   - 运行 Exact/QAOA，再投影回 CBQM 检查可行性与 canonical objective
   - 检查 `exact_for_task` 证据，并用弱 penalty 展示负例
   - 用额外 fixed-variable 微型案例验证非恒等投影与变量恢复

3. `03_cbqm_native_solver_validation.ipynb`
   - 读取 canonical `cbqm.v1` 契约示例
   - 用 `tests/oracles` 独立枚举原始可行域和 objective
   - 对比 `ExactCbqmSolver` 与 `LocalSearchCbqmSolver`
   - 校验 exact proof 与 heuristic `feasible` 语义
   - 检查 feasibility、bounds、optimality/infeasibility proof
   - 通过 `solve_native_problem_task()` 验证独立 exact promotion
   - notebook 不定义 solver、objective、feasibility 或穷举 helper

4. `04_mis_solver_validation.ipynb`
   - 读取 canonical `mis.v1` 契约示例
   - 用 `tests.oracles` 独立枚举最大独立集
   - 对比 `ExactMisSolver` 与 `GreedyMisSolver`
   - 验证 maximum-cardinality、maximum-weight 与 fixed-value 不可行证书
   - 通过 `solve_native_problem_task()` 验证独立 exact promotion
   - notebook 不定义 solver、objective、feasibility 或穷举 helper

5. `05_simulated_annealing_validation.ipynb`
   - 从 `problem.benchmarks` 加载 deterministic QUBO fixtures
   - 用 `tests/oracles` 与 `ExactQuboSolver` 建立独立对照
   - 验证 SA 的 `feasible` 语义、canonical energy 与固定 seed
   - 演示显式 custom beta schedule，不在 notebook 中实现退火逻辑

6. `06_qrbnbr_reproduction.ipynb`
   - 生成论文 S1 风格的确定性随机无权 MaxCut `qubo.v1`
   - 保留 NetworkX node-link 源图、论文 URL 与源码 revision
   - 用独立 oracle 和 `ExactQuboSolver` 建立真实最优值
   - 运行 R1 correlation 与 R2 selective-composition 两条 Q-RBnBR 路线
   - 验证 edge-parity tree 穷尽、admissible bound 与 canonical result

7. `07_goemans_williamson_validation.ipynb`
   - 验证 CVXPY SDP、PSD factorization 与 seeded hyperplane rounding
   - 用独立 oracle 检查 cut、SDP relaxation 和 canonical energy 方向
   - 验证 CLARABEL，并在可用时交叉运行 SCS
   - 即使命中真实最优值也保持 approximation solver 的 `feasible`

8. `08_qrr_vs_gw_bnb_reproduction.ipynb`
   - 在同一个 S1 风格 MaxCut 上比较 QRR-BnB 与 GW-BnB
   - 对照 R1 correlation/SDP 与 R2 selective/SDP
   - 验证两条路线共享相同的 parity-tree optimality certificate
   - 展示 nodes、pruned、relaxations 与 exact leaves，不实现算法逻辑

## 运行

在 VS Code 中打开 notebook，选择项目 `.venv` 对应的 Python 3.11 kernel，
然后点击 **Run All**。第一个 code cell 会自动向上寻找同时包含 `lib/` 与
`problem/` 的仓库根目录，不需要硬编码本机路径。

所有 notebook 都保持无输出状态提交，避免 runtime、随机实验输出和本地路径
污染 Git diff。当前版本已使用项目 Python 3.11 `.venv` kernel 逐 cell 执行
通过；kernelspec 的显示名称不作为验证依据。

独立 oracle 的实现统一放在 `tests/oracles/`。Notebook 不定义 objective、
feasibility、穷举或算法 helper，只负责导入、调用、展示与简单断言。

## 使用边界

- Exact 与 statevector QAOA 都按指数增长，只适合这里的小规模验证。
- 固定 QAOA seed 用于复现实验路径，但不要把启发式命中最优解等同于最优性
  证明。
- 编译器证书证明代数转换属性，不自动保证任意 penalty 都足够。第二本
  notebook 同时给出强 penalty 正例和弱 penalty 反例。
- 替换示例模型时，应同步更新独立 oracle 的预期断言；不要只删除断言让
  notebook “运行成功”。
