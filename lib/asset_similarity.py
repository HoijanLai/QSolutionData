import numpy as np
import pandas as pd


SIMILARITY_COLUMNS = (
    '6m_return',
    '1y_return',
    '2y_return',
    '3y_return',
    '3y_annualized_return',
    '1y_return_std',
    '3y_return_std',
    '1y_downside_std',
    '3y_downside_std',
    '1y_max_drawdown',
    '3y_max_drawdown',
    '1y_sharpe',
    '3y_sharpe',
    '1y_sortino',
    '3y_sortino',
)


def asset_similarity(
    df,
    strategy='cosine',
    high_threshold=0.85,
    low_threshold=0.30,
):
    _validate_similarity_data(df, high_threshold, low_threshold)

    if strategy == 'cosine':
        features = _select_similarity_features(df)
        standardized = _standardize_features(features)
        similarity_matrix = _calculate_cosine_similarity(
            standardized,
            labels=df['code'].astype('string').tolist(),
        )
        high_pairs = _extract_similarity_pairs(
            similarity_matrix,
            high_threshold,
            mode='high',
        )
        low_pairs = _extract_similarity_pairs(
            similarity_matrix,
            low_threshold,
            mode='low',
        )
        high_groups = _build_similarity_groups(high_pairs)
        style_clusters = _cluster_asset_styles(similarity_matrix, high_threshold)

        return {
            'similarity_matrix': similarity_matrix,
            'high_similarity_pairs': high_pairs,
            'low_similarity_pairs': low_pairs,
            'high_similarity_groups': high_groups,
            'style_clusters': style_clusters,
        }

    raise NotImplementedError(f"Strategy '{strategy}' is not implemented.")


def _validate_similarity_data(df, high_threshold, low_threshold):
    if not isinstance(df, pd.DataFrame):
        raise TypeError('df must be a pandas DataFrame.')
    if len(df) < 2:
        raise ValueError('df must contain at least two assets.')
    if 'code' not in df.columns:
        raise ValueError("df must contain a 'code' column.")
    if df['code'].isna().any() or df['code'].astype('string').str.strip().eq('').any():
        raise ValueError("Column 'code' must not contain missing or blank values.")
    if df['code'].astype('string').duplicated().any():
        raise ValueError("Column 'code' must contain unique values.")

    missing = sorted(set(SIMILARITY_COLUMNS).difference(df.columns))
    if missing:
        raise ValueError(f'Missing similarity columns: {missing}')

    if not -1 <= low_threshold <= 1 or not -1 <= high_threshold <= 1:
        raise ValueError('Similarity thresholds must be between -1 and 1.')
    if low_threshold >= high_threshold:
        raise ValueError('low_threshold must be smaller than high_threshold.')

    for column in SIMILARITY_COLUMNS:
        if pd.to_numeric(df[column], errors='coerce').notna().sum() == 0:
            raise ValueError(f"Column '{column}' contains no numeric values.")


def _select_similarity_features(df):
    return df.loc[:, SIMILARITY_COLUMNS].apply(pd.to_numeric, errors='coerce')


def _standardize_features(features):
    standardized = pd.DataFrame(index=features.index)

    for column in features.columns:
        values = features[column].replace([np.inf, -np.inf], np.nan)
        values = values.fillna(values.median())
        standard_deviation = values.std(ddof=0)
        if standard_deviation == 0 or pd.isna(standard_deviation):
            standardized[column] = 0.0
        else:
            standardized[column] = (values - values.mean()) / standard_deviation

    return standardized


def _calculate_cosine_similarity(matrix, labels=None):
    values = np.asarray(matrix, dtype=float)
    norms = np.linalg.norm(values, axis=1)
    denominator = np.outer(norms, norms)
    similarity = np.divide(
        values @ values.T,
        denominator,
        out=np.zeros((len(values), len(values)), dtype=float),
        where=denominator != 0,
    )
    similarity = np.clip(similarity, -1.0, 1.0)
    np.fill_diagonal(similarity, 1.0)

    if labels is None:
        labels = list(range(len(values)))
    return pd.DataFrame(similarity, index=labels, columns=labels)


def _extract_similarity_pairs(matrix, threshold, mode):
    if mode not in {'high', 'low'}:
        raise ValueError("mode must be either 'high' or 'low'.")

    pairs = []
    labels = matrix.index.tolist()
    for left_index, left_code in enumerate(labels):
        for right_index in range(left_index + 1, len(labels)):
            similarity = float(matrix.iat[left_index, right_index])
            selected = similarity >= threshold if mode == 'high' else similarity <= threshold
            if selected:
                pairs.append({
                    'code_a': left_code,
                    'code_b': labels[right_index],
                    'similarity': similarity,
                })

    return pd.DataFrame(pairs, columns=['code_a', 'code_b', 'similarity'])


def _build_similarity_groups(pairs):
    if pairs.empty:
        return []

    graph = {}
    for row in pairs.itertuples(index=False):
        graph.setdefault(row.code_a, set()).add(row.code_b)
        graph.setdefault(row.code_b, set()).add(row.code_a)

    groups = []
    visited = set()
    for code in sorted(graph, key=str):
        if code in visited:
            continue

        stack = [code]
        group = set()
        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)
            group.add(current)
            stack.extend(graph.get(current, set()).difference(visited))
        groups.append(sorted(group, key=str))

    return groups


def _cluster_asset_styles(matrix, threshold=0.85):
    pairs = _extract_similarity_pairs(matrix, threshold, mode='high')
    groups = _build_similarity_groups(pairs)
    cluster_by_code = {}

    for cluster_id, group in enumerate(groups):
        for code in group:
            cluster_by_code[code] = cluster_id

    next_cluster = len(groups)
    for code in matrix.index:
        if code not in cluster_by_code:
            cluster_by_code[code] = next_cluster
            next_cluster += 1

    return pd.Series(cluster_by_code, name='style_cluster').reindex(matrix.index)
