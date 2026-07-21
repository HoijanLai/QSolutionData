import numpy as np
import pandas as pd


RETURN_COLUMNS = (
    '6m_return',
    '1y_return',
    '2y_return',
    '3y_annualized_return',
)

RISK_COLUMNS = (
    '1y_return_std',
    '3y_return_std',
    '1y_downside_std',
    '3y_downside_std',
    '1y_max_drawdown',
    '3y_max_drawdown',
)

STABILITY_COLUMNS = (
    '1y_sharpe',
    '3y_sharpe',
    '1y_sortino',
    '3y_sortino',
    '1y_calmar',
    '3y_calmar',
)


def asset_scoring(df, weights=None, strategy='weighted'):
    _validate_scoring_data(df)

    if strategy == 'weighted':
        score_weights = _resolve_score_weights(weights)
        result = df.copy()
        result['return_score'] = _calculate_return_score(result, score_weights['return'])
        result['risk_score'] = _calculate_risk_score(result, score_weights['risk'])
        result['stability_score'] = _calculate_stability_score(
            result,
            score_weights['stability'],
        )
        return result, score_weights

    raise NotImplementedError(f"Strategy '{strategy}' is not implemented.")


def _validate_scoring_data(df):
    if not isinstance(df, pd.DataFrame):
        raise TypeError('df must be a pandas DataFrame.')
    if df.empty:
        raise ValueError('df must contain at least one asset.')

    required = set(RETURN_COLUMNS + RISK_COLUMNS + STABILITY_COLUMNS)
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f'Missing scoring columns: {missing}')

    for column in required:
        values = pd.to_numeric(df[column], errors='coerce')
        if values.notna().sum() == 0:
            raise ValueError(f"Column '{column}' contains no numeric values.")


def _resolve_score_weights(weights):
    groups = {
        'return': RETURN_COLUMNS,
        'risk': RISK_COLUMNS,
        'stability': STABILITY_COLUMNS,
    }
    resolved = {
        group: {column: 1 / len(columns) for column in columns}
        for group, columns in groups.items()
    }

    if weights is None:
        return resolved
    if not isinstance(weights, dict):
        raise TypeError('weights must be a dictionary grouped by score name.')

    unknown_groups = set(weights).difference(groups)
    if unknown_groups:
        raise ValueError(f'Unknown score groups: {sorted(unknown_groups)}')

    for group, custom_weights in weights.items():
        if not isinstance(custom_weights, dict) or not custom_weights:
            raise ValueError(f"Weights for '{group}' must be a non-empty dictionary.")

        unknown_columns = set(custom_weights).difference(groups[group])
        if unknown_columns:
            raise ValueError(
                f"Unknown columns in '{group}' weights: {sorted(unknown_columns)}"
            )

        merged = resolved[group].copy()
        merged.update(custom_weights)
        numeric = {column: float(value) for column, value in merged.items()}
        if any(not np.isfinite(value) or value < 0 for value in numeric.values()):
            raise ValueError(f"Weights for '{group}' must be finite and non-negative.")

        total = sum(numeric.values())
        if total <= 0:
            raise ValueError(f"Weights for '{group}' must have a positive sum.")
        resolved[group] = {column: value / total for column, value in numeric.items()}

    return resolved


def _calculate_weighted_score(df, columns, weights, absolute_columns=()):
    standardized = pd.DataFrame(index=df.index)

    for column in columns:
        values = pd.to_numeric(df[column], errors='coerce').replace(
            [np.inf, -np.inf],
            np.nan,
        )
        if column in absolute_columns:
            values = values.abs()

        values = values.fillna(values.median())
        standard_deviation = values.std(ddof=0)
        if standard_deviation == 0 or pd.isna(standard_deviation):
            standardized[column] = 0.0
        else:
            standardized[column] = (values - values.mean()) / standard_deviation

    return sum(standardized[column] * weights[column] for column in columns)


def _calculate_return_score(df, weights):
    return _calculate_weighted_score(df, RETURN_COLUMNS, weights)


def _calculate_risk_score(df, weights):
    return _calculate_weighted_score(
        df,
        RISK_COLUMNS,
        weights,
        absolute_columns=('1y_max_drawdown', '3y_max_drawdown'),
    )


def _calculate_stability_score(df, weights):
    return _calculate_weighted_score(df, STABILITY_COLUMNS, weights)
