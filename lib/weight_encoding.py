import numpy as np
import pandas as pd


DEFAULT_WEIGHT_LEVELS = (0.00, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.15)


def weight_encoding(df, levels=DEFAULT_WEIGHT_LEVELS):
    _validate_weight_data(df)
    normalized_levels = _validate_weight_levels(levels)
    variable_mapping = _build_binary_variable_mapping(df, normalized_levels)

    return {
        'weight_levels': normalized_levels,
        'variable_mapping': variable_mapping,
        'weight_lookup': _build_weight_lookup(variable_mapping),
        'variable_count': _calculate_variable_count(df, normalized_levels),
    }


def _validate_weight_data(df):
    if not isinstance(df, pd.DataFrame):
        raise TypeError('df must be a pandas DataFrame.')
    if df.empty:
        raise ValueError('df must contain at least one asset.')
    if 'code' not in df.columns:
        raise ValueError("df must contain a 'code' column.")

    codes = df['code'].astype('string').str.strip()
    if codes.isna().any() or codes.eq('').any():
        raise ValueError("Column 'code' must not contain missing or blank values.")
    if codes.duplicated().any():
        raise ValueError("Column 'code' must contain unique values.")


def _validate_weight_levels(levels):
    if isinstance(levels, (str, bytes)):
        raise TypeError('levels must be an iterable of numeric values.')
    try:
        normalized = tuple(float(level) for level in levels)
    except (TypeError, ValueError) as error:
        raise TypeError('levels must be an iterable of numeric values.') from error

    if not normalized:
        raise ValueError('levels must not be empty.')
    if any(not np.isfinite(level) or not 0 <= level <= 1 for level in normalized):
        raise ValueError('Each weight level must be finite and between 0 and 1.')
    if len(set(normalized)) != len(normalized):
        raise ValueError('Weight levels must be unique.')
    if tuple(sorted(normalized)) != normalized:
        raise ValueError('Weight levels must be sorted in ascending order.')
    if normalized[0] != 0:
        raise ValueError('Weight levels must include 0 as the first level.')
    return normalized


def _build_binary_variable_mapping(df, levels):
    rows = []
    codes = df['code'].astype('string').str.strip().tolist()

    for asset_index, code in enumerate(codes):
        for level_index, weight_level in enumerate(levels):
            variable_index = asset_index * len(levels) + level_index
            rows.append({
                'variable_index': variable_index,
                'variable_name': f'x_{asset_index}_{level_index}',
                'asset_index': asset_index,
                'level_index': level_index,
                'code': code,
                'weight_level': weight_level,
            })

    return pd.DataFrame(rows)


def _build_weight_lookup(mapping):
    return {
        row.variable_name: {
            'variable_index': row.variable_index,
            'code': row.code,
            'weight_level': row.weight_level,
        }
        for row in mapping.itertuples(index=False)
    }


def _calculate_variable_count(df, levels):
    return len(df) * len(levels)
