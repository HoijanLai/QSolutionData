# Preparation pipeline

This module orchestrates the asset and portfolio preparation steps. It does
not read files, choose objective coefficients, compile penalties, or run a
solver.

## `prepare_qubo_inputs`

Inputs:

- `df`: the numeric asset frame returned by the existing formatting flow.
- `manager_df`: a frame with unique `code` and non-missing `fund_manager`.
- profile, score, similarity, weight-level, and constraint strategy options.

Output: a portfolio payload containing aligned assets, scores, classifications,
similarity results, variable mapping, structured constraints, fixed variables,
client profile, and metadata.

```python
from lib import prepare_qubo_inputs

payload = prepare_qubo_inputs(
    df,
    manager_df=managers,
    profile='steady',
    high_threshold=0.85,
    low_threshold=0.30,
)
```

The expected upstream flow is:

```python
raw_df, meta = my_read('真实资产池.xlsx')
meta_en = meta_translate(meta)
managers, manager_mapping = manager_formatting(
    meta_en[['code', 'fund_manager']]
)
df, category_dict = basic_formatting(raw_df)
```

`manager_mapping` and `category_dict` remain external provenance artifacts;
keep them with the experiment if their encodings must later be decoded.

The payload is an intermediate Python object with DataFrames. Pass it to
`build_portfolio_cbqm` to obtain the serializable neutral model.

## Human-expert audit before model construction

The source technical document determines the overall preparation workflow, but
it does not normatively fix every modelling and engineering choice below. A
domain expert should review these payload fields before the payload is converted
to `cbqm.v1`. Defaults are executable starting points, not approved investment
policy.

| Payload field | Current choice that requires review | Expert decision |
|---|---|---|
| `assets.return_score`, `assets.risk_score`, `assets.stability_score`; `score_weights` | Fixed lookback columns, cross-sectional z-score standardization, median imputation, absolute drawdown magnitude, and equal within-group weights by default | Confirm the metric universe, missing-data policy, scaling method, lookback importance, and score direction |
| `assets.asset_class`, `assets.risk_level` | Rule-table mapping from secondary investment type; some risk labels are intervals such as `R3-R4` and `R4-R5` | Approve mappings and decide how interval/unknown labels participate in risk constraints |
| `assets.fund_manager` | Uses the upstream manager encoding; a manager team may currently be represented as one category | Decide whether a team is atomic, split among individuals, multi-hot, or represented through an exposure matrix; audit aliases and missing values |
| `similarity_matrix` | Cosine similarity over a fixed feature list after implementation-defined cleaning and standardization | Confirm features, preprocessing, distance/similarity measure, signs, and treatment of missing or constant features |
| `high_similarity_pairs`, `low_similarity_pairs`, `high_similarity_groups`, `style_clusters`; `metadata.*_similarity_threshold` | Default thresholds are `0.85` and `0.30`; comparisons are inclusive; high-similarity groups use transitive connected components | Calibrate thresholds and approve whether transitive grouping and singleton clusters match the investment interpretation |
| `weight_levels`, `variable_mapping`, `weight_lookup` | Default discrete grid is `(0, .02, .04, .06, .08, .10, .12, .15)` | Confirm granularity, maximum representable holding, feasibility, and resulting binary-variable count |
| `constraints` with family `manager_cap` | Default aggregate cap is `0.25`; constraints are emitted only for repeated manager categories | Approve the cap and manager exposure model, especially for teams and multi-manager funds |
| `constraints` with family `investment_type_cap` | Default aggregate cap is `0.35`; constraints are emitted only for repeated secondary types | Approve the taxonomy and cap, including whether singleton types should still receive an explicit constraint |
| `constraints` with family `similarity_group_cap` | Default aggregate cap is `0.40`, applied to connected high-similarity groups | Approve the cap and decide whether overlapping pairwise, cluster, or factor-exposure constraints are preferable |
| `client_profile` and profile-derived `constraints` | Template targets, holding counts, single-asset caps, asset-class ranges, and `r5_cap` are policy assumptions unless supplied by an authoritative client mandate | Validate every numerical bound and check feasibility against the available asset pool and weight grid |

Three payload fields need an explicit modelling decision even though they are
present in `client_profile`:

- `target_return` is recorded but is not currently converted into a constraint.
- `volatility_cap` is recorded but is not currently converted into a
  constraint.
- `drawdown_cap` is recorded but is not currently converted into a constraint.

The expert must choose whether these remain reporting targets, become hard
constraints, or enter the objective as soft penalties. The implementation must
not infer that choice from the stored values.

Also audit the generated `constraints` rather than relying only on
`constraint_summary`: the summary contains counts, not the actual coefficients
or bounds. `fixed_variables` should be checked against the approved
`single_asset_cap`, because it is derived from that cap and the chosen weight
grid.

For reproducibility, record the code revision together with the payload.
`metadata` currently records dimensions, profile name, and similarity
thresholds, but not the full scoring, classification, similarity, manager, or
constraint-policy implementation versions. Objective coefficients and QUBO
penalty strengths are outside this payload and require separate expert review.

For the complete field-by-field contract, see `payload_reference.md` in this
directory.
