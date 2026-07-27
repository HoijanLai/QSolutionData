# Portfolio payload field reference

`prepare_qubo_inputs` returns a Python intermediate payload. It connects data
preparation to `build_portfolio_cbqm`.

The payload is not `cbqm.v1` or `qubo.v1` and must not be passed directly to a
solver. It contains pandas objects and is therefore not guaranteed to be JSON
serializable.

## Top-level fields

| Field | Python type | Meaning | Downstream role |
|---|---|---|---|
| `assets` | `pandas.DataFrame` | Aligned asset metadata, classifications, and derived scores | Used to build CBQM variable metadata and linear objective terms |
| `score_weights` | `dict[str, dict[str, float]]` | Normalized metric weights actually used by `asset_scoring` | Provenance and experiment reproducibility |
| `similarity_matrix` | `pandas.DataFrame` | Symmetric code-by-code cosine-similarity matrix | Used to build cross-asset quadratic objective terms |
| `high_similarity_pairs` | `pandas.DataFrame` | Asset pairs whose similarity is at least the high threshold | Diagnostics and group construction provenance |
| `low_similarity_pairs` | `pandas.DataFrame` | Asset pairs whose similarity is at most the low threshold | Diagnostics and diversification analysis |
| `high_similarity_groups` | `list[list[str]]` | Connected components formed from high-similarity pairs | Source for similarity-group concentration constraints |
| `style_clusters` | `pandas.Series` | Integer style-cluster ID indexed by asset code | Diagnostics and variable metadata |
| `client_profile` | `dict` | Resolved allocation and risk policy | Records the policy used to build constraints |
| `weight_levels` | `tuple[float, ...]` | Allowed discrete allocation weights | Defines the one-hot decision levels per asset |
| `variable_mapping` | `pandas.DataFrame` | Stable mapping from binary variables to assets and levels | Defines CBQM variables and translates constraints to indices |
| `weight_lookup` | `dict[str, dict]` | Name-keyed form of the variable mapping | Convenient inspection and later decoding |
| `constraints` | `list[dict]` | Structured bounded linear constraints | Converted directly into CBQM constraints |
| `fixed_variables` | `list[str]` | Variables that must be fixed to binary zero | Converted into CBQM `fixed_values` |
| `constraint_summary` | `dict` | Constraint and fixed-variable counts | Diagnostics and logging |
| `metadata` | `dict` | Payload-level dimensions and selected thresholds | Alignment checks and experiment provenance |

## `assets`

Rows follow the original asset-code order after manager alignment.

| Column | Type | Meaning |
|---|---|---|
| `code` | string | Unique stable asset identifier |
| `security_name` | string, optional | Human-readable fund/security name |
| `fund_manager` | string or encoded category | Manager or normalized manager-team category |
| `investment_type_secondary` | categorical/string | Formatted secondary investment type |
| `asset_class` | categorical | Broad English class such as `equity` or `fixed_income` |
| `risk_level` | categorical | Rule-derived level such as `R1`, `R3`, or `R4-R5` |
| `return_score` | float | Standardized weighted return feature; higher means stronger return metrics |
| `risk_score` | float | Standardized weighted risk magnitude; higher means more risk |
| `stability_score` | float | Standardized weighted Sharpe/Sortino/Calmar feature |
| `style_cluster` | integer | Similarity-connected style cluster |

`return_score`, `risk_score`, and `stability_score` are features. Their signs
and importance in the optimization objective are chosen later through
`objective_config`.

## `score_weights`

Structure:

```python
{
    'return': {'6m_return': ..., '1y_return': ..., ...},
    'risk': {'1y_return_std': ..., '1y_max_drawdown': ..., ...},
    'stability': {'1y_sharpe': ..., '1y_sortino': ..., ...},
}
```

Weights within each group are normalized to sum to one. This field records how
the three score columns were calculated; it is not the CBQM objective weight.

## Similarity fields

### `similarity_matrix`

- Index and columns are both the asset codes in `assets` order.
- Shape is `(asset_count, asset_count)`.
- The matrix is symmetric.
- The diagonal is `1.0`.
- Values are clipped to `[-1, 1]`.

### Pair DataFrames

Both `high_similarity_pairs` and `low_similarity_pairs` have columns:

| Column | Meaning |
|---|---|
| `code_a` | First asset code |
| `code_b` | Second asset code |
| `similarity` | Pairwise cosine similarity |

Only one row per unordered asset pair is emitted.

### `high_similarity_groups`

Each entry is a sorted list of codes in one connected component. Connectivity
is transitive: if A is paired with B and B with C, the group is `[A, B, C]`
even when A and C are not directly above the threshold.

### `style_clusters`

This Series maps every code to an integer cluster. Assets outside all
high-similarity groups receive their own singleton cluster.

## `client_profile`

```python
{
    'name': 'steady',
    'target_return': 0.055,
    'volatility_cap': 0.08,
    'drawdown_cap': 0.12,
    'holding_count': (8, 15),
    'single_asset_cap': 0.15,
    'asset_class_ranges': {
        'cash_money': (0.05, 0.20),
        # ...
    },
    'r5_cap': 0.10,
}
```

The current constraint builder directly uses `holding_count`,
`single_asset_cap`, `asset_class_ranges`, and `r5_cap`. Return, volatility, and
drawdown targets remain recorded policy fields and are not silently converted
into additional constraints by the current implementation.

## Weight-variable fields

### `weight_levels`

Sorted, unique values in `[0, 1]`; the first level is always zero. Every asset
receives exactly one binary variable for every level.

### `variable_mapping`

| Column | Meaning |
|---|---|
| `variable_index` | Zero-based stable index used by CBQM/QUBO |
| `variable_name` | Stable name such as `x_3_2` |
| `asset_index` | Zero-based row position in `assets` |
| `level_index` | Position in `weight_levels` |
| `code` | Asset code |
| `weight_level` | Portfolio weight represented when the variable equals one |

Variable order is asset-major, then level-major:

```text
variable_index = asset_index * number_of_levels + level_index
```

### `weight_lookup`

Equivalent name-keyed representation:

```python
{
    'x_3_2': {
        'variable_index': 26,
        'code': 'asset-code',
        'weight_level': 0.04,
    }
}
```

## `constraints`

Each item has the form:

```python
{
    'name': 'budget',
    'family': 'budget',
    'type': 'equality',
    'terms': {
        'x_0_1': 0.02,
        'x_0_2': 0.04,
    },
    'lower_bound': 1.0,
    'upper_bound': 1.0,
}
```

Its mathematical meaning is:

```text
lower_bound <= sum(terms[name] * variable[name]) <= upper_bound
```

`None` means that side is unbounded. Equal lower and upper bounds represent an
equality.

Current families:

| Family | Meaning |
|---|---|
| `budget` | Total selected weight equals one |
| `one_weight_level` | Exactly one weight-level variable is active per asset |
| `holding_count` | Number of assets assigned positive weight stays in range |
| `single_asset_cap` | Weight of each individual asset does not exceed its cap |
| `asset_class` | Combined weight per broad asset class stays in profile range |
| `r5_cap` | Combined weight of assets explicitly classified `R5` stays below cap |
| `manager_cap` | Combined weight of repeated manager category stays below limit |
| `investment_type_cap` | Combined weight of repeated secondary type stays below limit |
| `similarity_group_cap` | Combined weight of a high-similarity group stays below limit |

`type` is a descriptive field derived from the available bounds:
`equality`, `range`, `upper_bound`, or `lower_bound`.

## `fixed_variables`

This list contains variable names whose represented weight exceeds the
single-asset cap. Their required value is always zero. `build_portfolio_cbqm`
converts them to:

```python
{'index': variable_index, 'value': 0}
```

The QUBO compiler later eliminates these variables exactly.

## `constraint_summary`

```python
{
    'constraint_count': 25,
    'fixed_variable_count': 12,
    'constraints_by_family': {
        'budget': 1,
        'one_weight_level': 6,
        # ...
    },
}
```

This field is diagnostic only and does not independently affect the model.

## `metadata`

| Key | Meaning |
|---|---|
| `asset_count` | Number of rows in `assets` |
| `variable_count` | Number of rows in `variable_mapping` |
| `profile` | Resolved canonical client-profile name |
| `high_similarity_threshold` | Threshold used for high pairs/groups/clusters |
| `low_similarity_threshold` | Threshold used for low pairs |

## Fields used by `build_portfolio_cbqm`

The CBQM builder consumes:

- `assets`
- `similarity_matrix`
- `variable_mapping`
- `constraints`
- `fixed_variables`
- selected values from `metadata`

Other payload fields remain valuable for diagnostics, auditability, experiment
reproduction, and later human-readable decoding.

## Quick inspection

```python
print(payload.keys())
print(payload['assets'].head())
print(payload['variable_mapping'].head(16))
print(payload['constraint_summary'])

for constraint in payload['constraints'][:5]:
    print(
        constraint['name'],
        constraint['lower_bound'],
        constraint['upper_bound'],
        len(constraint['terms']),
    )
```
