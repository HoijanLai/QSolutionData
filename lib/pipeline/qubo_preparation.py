import pandas as pd

from ..preprocessing.asset_classification import asset_classification
from ..preprocessing.asset_scoring import asset_scoring
from ..preprocessing.asset_similarity import asset_similarity
from ..portfolio.client_profile import client_profile
from ..portfolio.portfolio_constraints import portfolio_constraints
from ..portfolio.weight_encoding import DEFAULT_WEIGHT_LEVELS, weight_encoding


def prepare_qubo_inputs(
    df,
    manager_df=None,
    profile='steady',
    score_weights=None,
    similarity_strategy='cosine',
    high_threshold=0.85,
    low_threshold=0.30,
    weight_levels=None,
    constraint_strategy='document_rules',
):
    """Orchestrate asset analytics and portfolio-constraint preparation.

    Args:
        df: Formatted asset DataFrame. It must contain unique ``code`` values,
            ``investment_type_secondary``, and all metrics required by scoring
            and similarity modules.
        manager_df: Optional metadata frame with unique ``code`` and non-missing
            ``fund_manager``. It is required when ``df`` has no manager column.
        profile: Client-profile name or alias passed to ``client_profile``.
        score_weights: Optional nested metric-weight overrides for scoring.
        similarity_strategy: Asset-similarity strategy.
        high_threshold: Threshold for high-similarity pairs and groups.
        low_threshold: Threshold for low-similarity pairs.
        weight_levels: Optional allowed allocation levels. Defaults to
            ``DEFAULT_WEIGHT_LEVELS``.
        constraint_strategy: Portfolio-constraint policy.

    Returns:
        A Python payload containing aligned asset metadata, derived scores,
        similarity outputs, client profile, weight-variable mapping, structured
        constraints, fixed variables, summaries, and provenance metadata.

    Raises:
        TypeError: If an input container has the wrong type.
        ValueError: If required data is missing, duplicated, or misaligned.
        NotImplementedError: If a selected downstream strategy is unavailable.

    Notes:
        The function never mutates ``df`` or ``manager_df``. The returned payload
        still contains pandas objects and becomes a neutral serializable model
        only after ``build_portfolio_cbqm``.
    """
    _validate_qubo_input(df, manager_df)
    aligned_df = _align_manager_data(df, manager_df)
    _validate_asset_alignment(aligned_df)

    scored_df, resolved_weights = asset_scoring(
        aligned_df,
        weights=score_weights,
    )
    classified_df = asset_classification(scored_df)
    similarity_result = asset_similarity(
        classified_df,
        strategy=similarity_strategy,
        high_threshold=high_threshold,
        low_threshold=low_threshold,
    )
    selected_profile = client_profile(profile)
    levels = DEFAULT_WEIGHT_LEVELS if weight_levels is None else weight_levels
    encoding_result = weight_encoding(classified_df, levels=levels)
    constraint_result = portfolio_constraints(
        classified_df,
        selected_profile,
        encoding_result['variable_mapping'],
        similarity_groups=similarity_result['high_similarity_groups'],
        strategy=constraint_strategy,
    )

    payload = _build_qubo_payload(
        classified_df,
        resolved_weights,
        similarity_result,
        selected_profile,
        encoding_result,
        constraint_result,
        high_threshold,
        low_threshold,
    )
    _validate_qubo_payload(payload)
    return payload


def _validate_qubo_input(df, manager_df):
    """Validate qubo input."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError('df must be a pandas DataFrame.')
    if df.empty:
        raise ValueError('df must contain at least one asset.')

    required = {'code', 'investment_type_secondary'}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f'Missing QUBO input columns: {missing}')

    codes = df['code'].astype('string').str.strip()
    if codes.isna().any() or codes.eq('').any():
        raise ValueError("Column 'code' must not contain missing or blank values.")
    if codes.duplicated().any():
        raise ValueError("Column 'code' must contain unique values.")

    if manager_df is None:
        if 'fund_manager' not in df.columns:
            raise ValueError(
                "Provide manager_df when df does not contain 'fund_manager'."
            )
        if df['fund_manager'].isna().any():
            raise ValueError("Column 'fund_manager' must not contain missing values.")
        return

    if not isinstance(manager_df, pd.DataFrame):
        raise TypeError('manager_df must be a pandas DataFrame.')
    manager_required = {'code', 'fund_manager'}
    manager_missing = sorted(manager_required.difference(manager_df.columns))
    if manager_missing:
        raise ValueError(f'Missing manager columns: {manager_missing}')


def _align_manager_data(df, manager_df):
    """Align manager data."""
    result = df.copy()
    result['code'] = result['code'].astype('string').str.strip()

    if manager_df is None:
        return result.reset_index(drop=True)

    managers = manager_df.copy()
    managers['code'] = managers['code'].astype('string').str.strip()
    if managers['code'].isna().any() or managers['code'].eq('').any():
        raise ValueError("manager_df column 'code' contains missing or blank values.")
    if managers['code'].duplicated().any():
        raise ValueError("manager_df column 'code' must contain unique values.")
    if managers['fund_manager'].isna().any():
        raise ValueError("manager_df column 'fund_manager' must not contain missing values.")

    asset_codes = set(result['code'])
    manager_codes = set(managers['code'])
    if asset_codes != manager_codes:
        missing_codes = sorted(asset_codes.difference(manager_codes))
        extra_codes = sorted(manager_codes.difference(asset_codes))
        raise ValueError(
            'manager_df codes do not match assets. '
            f'Missing: {missing_codes}; extra: {extra_codes}'
        )

    metadata_columns = ['code', 'fund_manager']
    if 'security_name' in managers.columns:
        metadata_columns.append('security_name')

    result = result.drop(
        columns=[
            column for column in metadata_columns
            if column != 'code' and column in result.columns
        ]
    )
    result = result.merge(
        managers[metadata_columns],
        on='code',
        how='left',
        validate='one_to_one',
        sort=False,
    )
    return result.reset_index(drop=True)


def _validate_asset_alignment(df):
    """Validate asset alignment."""
    if df['fund_manager'].isna().any():
        raise ValueError('Manager alignment produced missing fund managers.')
    if df['code'].duplicated().any():
        raise ValueError('Manager alignment produced duplicate asset codes.')


def _build_asset_output(df, style_clusters):
    """Build asset output."""
    preferred_columns = [
        'code',
        'security_name',
        'fund_manager',
        'investment_type_secondary',
        'asset_class',
        'risk_level',
        'return_score',
        'risk_score',
        'stability_score',
    ]
    columns = [column for column in preferred_columns if column in df.columns]
    assets = df.loc[:, columns].copy()
    assets['style_cluster'] = assets['code'].map(style_clusters)
    return assets


def _build_qubo_payload(
    df,
    score_weights,
    similarity_result,
    selected_profile,
    encoding_result,
    constraint_result,
    high_threshold,
    low_threshold,
):
    """Build qubo payload."""
    assets = _build_asset_output(df, similarity_result['style_clusters'])
    return {
        'assets': assets,
        'score_weights': score_weights,
        'similarity_matrix': similarity_result['similarity_matrix'],
        'high_similarity_pairs': similarity_result['high_similarity_pairs'],
        'low_similarity_pairs': similarity_result['low_similarity_pairs'],
        'high_similarity_groups': similarity_result['high_similarity_groups'],
        'style_clusters': similarity_result['style_clusters'],
        'client_profile': selected_profile,
        'weight_levels': encoding_result['weight_levels'],
        'variable_mapping': encoding_result['variable_mapping'],
        'weight_lookup': encoding_result['weight_lookup'],
        'constraints': constraint_result['constraints'],
        'fixed_variables': constraint_result['fixed_variables'],
        'constraint_summary': constraint_result['summary'],
        'metadata': {
            'asset_count': len(assets),
            'variable_count': encoding_result['variable_count'],
            'profile': selected_profile['name'],
            'high_similarity_threshold': high_threshold,
            'low_similarity_threshold': low_threshold,
        },
    }


def _validate_qubo_payload(payload):
    """Validate qubo payload."""
    asset_codes = payload['assets']['code'].astype('string').tolist()
    asset_count = len(asset_codes)
    if len(set(asset_codes)) != asset_count:
        raise ValueError('QUBO payload assets must have unique codes.')

    matrix = payload['similarity_matrix']
    if matrix.shape != (asset_count, asset_count):
        raise ValueError('Similarity matrix shape does not match asset count.')
    if matrix.index.astype('string').tolist() != asset_codes:
        raise ValueError('Similarity matrix index is not aligned with assets.')
    if matrix.columns.astype('string').tolist() != asset_codes:
        raise ValueError('Similarity matrix columns are not aligned with assets.')

    mapping = payload['variable_mapping']
    expected_variables = asset_count * len(payload['weight_levels'])
    if len(mapping) != expected_variables:
        raise ValueError('Variable mapping size does not match assets and levels.')
    if payload['metadata']['variable_count'] != expected_variables:
        raise ValueError('QUBO metadata variable count is inconsistent.')

    valid_variables = set(mapping['variable_name'])
    for constraint in payload['constraints']:
        unknown = set(constraint['terms']).difference(valid_variables)
        if unknown:
            raise ValueError(
                f"Constraint '{constraint['name']}' references unknown variables."
            )
    if set(payload['fixed_variables']).difference(valid_variables):
        raise ValueError('QUBO payload contains unknown fixed variables.')
