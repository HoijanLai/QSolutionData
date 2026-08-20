"""Real-asset loading and deterministic candidate-pool selection."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from pathlib import Path

import pandas as pd

from eda import basic_formatting, manager_formatting, meta_translate, my_read

from ..preprocessing.asset_classification import asset_classification
from ..preprocessing.asset_scoring import asset_scoring
from ..preprocessing.asset_similarity import asset_similarity

DEFAULT_ASSET_CLASS_ORDER = (
    'cash_money',
    'fixed_income',
    'mixed',
    'equity',
    'alternative',
)

DEFAULT_SELECTION_SCORE_COEFFICIENTS = {
    'return_score': -1.0,
    'risk_score': 1.0,
    'stability_score': -1.0,
}


def load_real_asset_pool(path):
    """Load one source workbook through the repository's formatting flow.

    The returned frame contains numeric metrics plus aligned security and
    manager metadata. Manager teams use the stable integer encoding produced by
    :func:`eda.manager_formatting`; the reverse mapping is returned as
    provenance rather than embedded in every row.
    """
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f'Asset-pool file not found: {source}')
    if source.suffix.lower() not in {'.xlsx', '.xls'}:
        raise ValueError('The real-asset loader currently accepts Excel workbooks.')

    raw_df, meta_df = my_read(source)
    assets, category_mapping = basic_formatting(raw_df)
    metadata = meta_translate(meta_df)
    managers, manager_mapping = manager_formatting(
        metadata.loc[:, ['code', 'fund_manager']]
    )

    assets = assets.copy()
    assets['code'] = assets['code'].astype('string').str.strip()
    manager_columns = managers.loc[:, ['code', 'fund_manager']].copy()
    manager_columns['code'] = manager_columns['code'].astype('string').str.strip()
    assets = assets.merge(
        manager_columns,
        on='code',
        how='left',
        validate='one_to_one',
        sort=False,
    )

    if 'security_name' in metadata.columns:
        names = metadata.loc[:, ['code', 'security_name']].copy()
        names['code'] = names['code'].astype('string').str.strip()
        assets = assets.merge(
            names,
            on='code',
            how='left',
            validate='one_to_one',
            sort=False,
        )

    _validate_loaded_assets(assets)
    return assets.reset_index(drop=True), {
        'source_path': str(source.resolve()),
        'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
        'source_size_bytes': source.stat().st_size,
        'source_row_count': len(assets),
        'manager_mapping': dict(manager_mapping),
        'category_mapping': category_mapping,
    }


def select_asset_candidates(
    assets,
    *,
    candidate_count=40,
    score_weights=None,
    score_coefficients=None,
    class_order=DEFAULT_ASSET_CLASS_ORDER,
    high_threshold=0.85,
    low_threshold=0.30,
):
    """Score the full universe and select a balanced deterministic candidate pool.

    Candidate slots are distributed as evenly as possible across the five broad
    asset classes. Inside each class, the lowest selection cost is preferred,
    with asset code as a stable tie-breaker. Similarity is computed only after
    selection, keeping the downstream quantum model bounded by
    ``candidate_count``.
    """
    _validate_candidate_request(assets, candidate_count, class_order)
    coefficients = _resolve_score_coefficients(score_coefficients)

    scored, resolved_weights = asset_scoring(assets, weights=score_weights)
    classified = asset_classification(scored)
    classified = classified.copy()
    classified['selection_cost'] = sum(
        coefficients[column] * classified[column].astype(float)
        for column in coefficients
    )

    quotas = _balanced_candidate_quotas(
        classified,
        candidate_count,
        class_order,
    )
    selected_parts = []
    for asset_class in class_order:
        count = quotas[asset_class]
        group = classified.loc[
            classified['asset_class'].astype('string').eq(asset_class)
        ].copy()
        selected_parts.append(_select_diversified_class(group, count))

    selected = pd.concat(selected_parts, ignore_index=True)
    if len(selected) != candidate_count:
        raise RuntimeError('Candidate selection did not produce the requested size.')
    if selected['code'].astype('string').duplicated().any():
        raise RuntimeError('Candidate selection produced duplicate asset codes.')

    similarity = asset_similarity(
        selected,
        high_threshold=high_threshold,
        low_threshold=low_threshold,
    )
    return {
        'assets': selected.reset_index(drop=True),
        'similarity': similarity,
        'score_weights': resolved_weights,
        'score_coefficients': coefficients,
        'candidate_quotas': quotas,
        'metadata': {
            'source_asset_count': len(assets),
            'candidate_count': int(candidate_count),
            'selection_strategy': 'balanced_asset_class_score',
            'class_order': list(class_order),
            'high_similarity_threshold': float(high_threshold),
            'low_similarity_threshold': float(low_threshold),
        },
    }


def _validate_loaded_assets(assets):
    required = {'code', 'fund_manager', 'investment_type_secondary'}
    missing = sorted(required.difference(assets.columns))
    if missing:
        raise ValueError(f'Loaded asset pool is missing columns: {missing}')
    if assets.empty:
        raise ValueError('Loaded asset pool must contain at least one asset.')
    codes = assets['code'].astype('string').str.strip()
    if codes.isna().any() or codes.eq('').any() or codes.duplicated().any():
        raise ValueError('Loaded asset codes must be unique non-empty strings.')
    if assets['fund_manager'].isna().any():
        raise ValueError('Loaded asset pool contains missing fund managers.')


def _validate_candidate_request(assets, candidate_count, class_order):
    if not isinstance(assets, pd.DataFrame):
        raise TypeError('assets must be a pandas DataFrame.')
    if not isinstance(candidate_count, int) or isinstance(candidate_count, bool):
        raise TypeError('candidate_count must be an integer.')
    if candidate_count <= 0 or candidate_count > len(assets):
        raise ValueError('candidate_count must be between one and asset count.')
    if isinstance(class_order, (str, bytes)) or not isinstance(class_order, Sequence):
        raise TypeError('class_order must be a sequence of asset-class names.')
    if tuple(class_order) != DEFAULT_ASSET_CLASS_ORDER:
        unknown = sorted(set(class_order).difference(DEFAULT_ASSET_CLASS_ORDER))
        if unknown or len(set(class_order)) != len(DEFAULT_ASSET_CLASS_ORDER):
            raise ValueError('class_order must contain each supported asset class once.')


def _resolve_score_coefficients(score_coefficients):
    if score_coefficients is None:
        return dict(DEFAULT_SELECTION_SCORE_COEFFICIENTS)
    if not isinstance(score_coefficients, Mapping):
        raise TypeError('score_coefficients must be a mapping or None.')
    unknown = sorted(
        set(score_coefficients).difference(DEFAULT_SELECTION_SCORE_COEFFICIENTS)
    )
    if unknown:
        raise ValueError(f'Unknown score coefficients: {unknown}')
    resolved = dict(DEFAULT_SELECTION_SCORE_COEFFICIENTS)
    resolved.update(score_coefficients)
    for name, value in resolved.items():
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise TypeError(f"Score coefficient '{name}' must be numeric.")
    return {name: float(value) for name, value in resolved.items()}


def _balanced_candidate_quotas(assets, candidate_count, class_order):
    availability = {
        asset_class: int(
            assets['asset_class'].astype('string').eq(asset_class).sum()
        )
        for asset_class in class_order
    }
    if any(value == 0 for value in availability.values()):
        missing = [name for name, value in availability.items() if value == 0]
        raise ValueError(f'Asset universe has no candidates for classes: {missing}')

    base, remainder = divmod(candidate_count, len(class_order))
    requested = {
        asset_class: base + (position < remainder)
        for position, asset_class in enumerate(class_order)
    }
    quotas = {
        asset_class: min(requested[asset_class], availability[asset_class])
        for asset_class in class_order
    }
    remaining = candidate_count - sum(quotas.values())
    while remaining:
        progressed = False
        for asset_class in class_order:
            if quotas[asset_class] < availability[asset_class]:
                quotas[asset_class] += 1
                remaining -= 1
                progressed = True
                if remaining == 0:
                    break
        if not progressed:
            raise ValueError('Asset classes cannot supply the requested candidate count.')
    return quotas


def _select_diversified_class(group, count):
    """Balance secondary types and prefer distinct managers within each type."""
    type_names = sorted(
        group['investment_type_secondary'].astype('string').unique().tolist()
    )
    availability = {
        name: int(
            group['investment_type_secondary'].astype('string').eq(name).sum()
        )
        for name in type_names
    }
    quotas = {name: 0 for name in type_names}
    remaining = count
    while remaining:
        progressed = False
        for name in type_names:
            if quotas[name] < availability[name]:
                quotas[name] += 1
                remaining -= 1
                progressed = True
                if not remaining:
                    break
        if not progressed:
            raise ValueError('Investment types cannot supply the class candidate quota.')

    parts = []
    for name in type_names:
        candidates = group.loc[
            group['investment_type_secondary'].astype('string').eq(name)
        ].sort_values(['selection_cost', 'code'], kind='mergesort')
        distinct = candidates.drop_duplicates('fund_manager', keep='first')
        selected = distinct.head(quotas[name])
        if len(selected) < quotas[name]:
            remainder = candidates.loc[~candidates.index.isin(selected.index)]
            selected = pd.concat(
                [selected, remainder.head(quotas[name] - len(selected))]
            )
        parts.append(selected)
    return pd.concat(parts).sort_values(
        ['investment_type_secondary', 'selection_cost', 'code'],
        kind='mergesort',
    )


__all__ = [
    'DEFAULT_ASSET_CLASS_ORDER',
    'DEFAULT_SELECTION_SCORE_COEFFICIENTS',
    'load_real_asset_pool',
    'select_asset_candidates',
]
