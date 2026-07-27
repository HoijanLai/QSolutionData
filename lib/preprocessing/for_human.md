# Data preprocessing

This module derives model-ready information for individual assets. Its public
functions accept a pandas `DataFrame`, return a new object, and do not mutate
the caller's frame.

## `asset_scoring`

Input: formatted asset metrics, including the return, risk, Sharpe, Sortino,
Calmar, volatility, downside-volatility, and drawdown columns listed in
`asset_scoring.py`.

Output: `(scored_df, resolved_weights)`. The frame gains `return_score`,
`risk_score`, and `stability_score`; the second value records the normalized
metric weights actually used.

```python
from lib.preprocessing import asset_scoring

scored_df, weights = asset_scoring(df, weights=None)
```

Scores are weighted standardized metrics. They are engineering features, not
portfolio weights or final objective coefficients.

## `asset_classification`

Input: a frame containing `investment_type_secondary`.

Output: a copied frame with English `asset_class` and `risk_level` columns.

```python
from lib.preprocessing import asset_classification

classified_df = asset_classification(
    scored_df,
    strategy='rules',
    unknown_policy='raise',
)
```

Use `unknown_policy='keep'` when new investment types should be preserved as
an explicit unknown category instead of stopping the pipeline.

## `asset_similarity`

Input: classified/scored assets with unique `code` values and all similarity
feature columns.

Output: a dictionary containing the aligned similarity matrix, high/low pairs,
high-similarity connected groups, and a style-cluster label per asset.

```python
from lib.preprocessing import asset_similarity

similarity = asset_similarity(
    classified_df,
    strategy='cosine',
    high_threshold=0.85,
    low_threshold=0.30,
)
```

The matrix uses standardized features and cosine similarity. Thresholds and
clustering are modelling choices and should be recorded in experiments.
