# Portfolio model construction

This module converts asset information into portfolio decision variables,
structured constraints, and finally `cbqm.v1`.

## `client_profile`

Input: a profile name (`conservative`, `steady`, `balanced`, or `aggressive`)
and optional field overrides.

Output: an independent profile dictionary containing return/risk targets,
holding-count limits, per-asset caps, asset-class ranges, and the R5 cap.

```python
from lib.portfolio import client_profile

profile = client_profile(
    'steady',
    overrides={'single_asset_cap': 0.12},
)
```

## `weight_encoding`

Input: assets with unique `code` values and an ordered collection of allowed
weight levels whose first value is zero.

Output: the normalized levels, a DataFrame mapping each binary variable to an
asset and level, a lookup dictionary, and the total variable count.

```python
from lib.portfolio import weight_encoding

encoding = weight_encoding(df, levels=(0.0, 0.05, 0.10, 0.15))
```

## `portfolio_constraints`

Input: classified assets, a resolved client profile, the variable-mapping
DataFrame, and optional high-similarity groups and concentration limits.

Output: structured linear constraints, variables fixed to zero, and a summary
by constraint family. No QUBO penalties are added here.

```python
from lib.portfolio import portfolio_constraints

constraint_data = portfolio_constraints(
    classified_df,
    profile,
    encoding['variable_mapping'],
    similarity_groups=high_similarity_groups,
)
```

## `build_portfolio_cbqm`

`build_portfolio_cbqm` converts the output of `prepare_qubo_inputs` into the
neutral `cbqm.v1` contract. It does not compile constraints into penalties.

The objective configuration is explicit:

```python
objective_config = {
    'strategy': 'score_similarity',
    'sense': 'minimize',
    'offset': 0.0,
    'score_coefficients': {
        'return_score': -1.0,
        'risk_score': 1.0,
        'stability_score': -1.0,
    },
    'similarity_coefficient': 1.0,
    'similarity_transform': 'raw',
}

cbqm = build_portfolio_cbqm(
    payload,
    objective_config,
    problem_id='portfolio-demo',
)
```

For a variable representing asset `i` at weight level `w`, its linear
coefficient is

```text
w * sum(score_coefficients[k] * asset_score[i, k])
```

For variables representing distinct assets `i` and `j` at levels `w_i` and
`w_j`, the quadratic coefficient is

```text
similarity_coefficient * transform(similarity[i, j]) * w_i * w_j
```

Available similarity transforms are:

- `raw`: preserve signed similarity.
- `positive_only`: ignore negative similarity.
- `absolute`: penalize similarity magnitude regardless of sign.

The signs above are examples, not defaults. In a minimization model, negative
return/stability coefficients reward those scores while a positive risk
coefficient penalizes risk. The caller must provide the coefficients, so the
research choice remains visible and versionable.
