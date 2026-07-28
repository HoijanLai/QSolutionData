import pandas as pd


def asset_classification(df, strategy='rules', unknown_policy='raise'):
    """Classify each asset into a broad asset class and risk level.

    Args:
        df: Non-empty pandas DataFrame containing
            ``investment_type_secondary``. The input is never mutated.
        strategy: Classification implementation. Version 1 supports ``rules``.
        unknown_policy: ``raise`` rejects an unmapped investment type; ``keep``
            emits the explicit ``unknown`` category.

    Returns:
        A copied DataFrame with categorical ``asset_class`` and ``risk_level``
        columns aligned to the original rows.

    Raises:
        TypeError: If ``df`` is not a DataFrame.
        ValueError: If required data is empty, missing, or invalid.
        NotImplementedError: If the requested strategy is unavailable.
    """
    _validate_classification_data(df, unknown_policy)

    if strategy == 'rules':
        asset_rules = _get_asset_class_rules()
        risk_rules = _get_risk_level_rules()
        result = df.copy()
        result['asset_class'] = result['investment_type_secondary'].map(
            lambda value: _classify_asset_type(value, asset_rules, unknown_policy)
        )
        result['risk_level'] = result['investment_type_secondary'].map(
            lambda value: _classify_risk_level(value, risk_rules, unknown_policy)
        )
        result['asset_class'] = result['asset_class'].astype('category')
        result['risk_level'] = result['risk_level'].astype('category')
        return result

    raise NotImplementedError(f"Strategy '{strategy}' is not implemented.")


def _validate_classification_data(df, unknown_policy):
    """Validate classification data."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError('df must be a pandas DataFrame.')
    if df.empty:
        raise ValueError('df must contain at least one asset.')
    if 'investment_type_secondary' not in df.columns:
        raise ValueError("df must contain an 'investment_type_secondary' column.")
    if df['investment_type_secondary'].isna().any():
        raise ValueError("Column 'investment_type_secondary' must not contain missing values.")
    if unknown_policy not in {'raise', 'keep'}:
        raise ValueError("unknown_policy must be either 'raise' or 'keep'.")


def _get_asset_class_rules():
    """Return asset class rules."""
    return {
        '货币市场型基金': 'cash_money',
        'money_market': 'cash_money',
        '短期纯债型基金': 'fixed_income',
        '中长期纯债型基金': 'fixed_income',
        '指数债券型基金': 'fixed_income',
        'short_term_bond': 'fixed_income',
        'medium_long_term_bond': 'fixed_income',
        'bond_index': 'fixed_income',
        '混合债券型一级基金': 'fixed_income',
        '混合债券型二级基金': 'fixed_income',
        'hybrid_bond_primary': 'fixed_income',
        'hybrid_bond_secondary': 'fixed_income',
        '股票多空': 'alternative',
        'equity_long_short': 'alternative',
        '偏债混合型基金': 'mixed',
        '灵活配置型基金': 'mixed',
        '平衡混合型基金': 'mixed',
        '偏股混合型基金': 'mixed',
        'bond_hybrid': 'mixed',
        'flexible_allocation': 'mixed',
        'balanced_hybrid': 'mixed',
        'equity_hybrid': 'mixed',
        '普通股票型基金': 'equity',
        '被动指数型基金': 'equity',
        '增强指数型基金': 'equity',
        'equity': 'equity',
        'passive_index': 'equity',
        'enhanced_index': 'equity',
        '商品型基金': 'alternative',
        '可转换债券型基金': 'alternative',
        'commodity': 'alternative',
        'convertible_bond': 'alternative',
    }


def _get_risk_level_rules():
    """Return risk level rules."""
    asset_rules = _get_asset_class_rules()
    risk_by_asset_class = {
        'cash_money': 'R1',
        'fixed_income': 'R2',
        'mixed': 'R3-R4',
        'equity': 'R4-R5',
        'alternative': 'R5',
    }
    rules = {
        investment_type: risk_by_asset_class[asset_class]
        for investment_type, asset_class in asset_rules.items()
    }

    for investment_type in (
        '混合债券型一级基金',
        '混合债券型二级基金',
        'hybrid_bond_primary',
        'hybrid_bond_secondary',
        '股票多空',
        'equity_long_short',
    ):
        rules[investment_type] = 'R3'
    return rules


def _classify_asset_type(investment_type, rules, unknown_policy):
    """Classify asset type."""
    key = str(investment_type).strip()
    if key in rules:
        return rules[key]
    if unknown_policy == 'keep':
        return 'unknown'
    raise ValueError(f"Unknown investment type: '{investment_type}'")


def _classify_risk_level(investment_type, rules, unknown_policy):
    """Classify risk level."""
    key = str(investment_type).strip()
    if key in rules:
        return rules[key]
    if unknown_policy == 'keep':
        return 'unknown'
    raise ValueError(f"Unknown investment type: '{investment_type}'")
