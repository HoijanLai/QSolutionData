import math

import pandas as pd


def portfolio_constraints(
    df,
    client_template,
    variable_mapping,
    similarity_groups=None,
    strategy='document_rules',
    manager_limit=0.25,
    investment_type_limit=0.35,
    similarity_group_limit=0.40,
):
    """Build structured linear portfolio constraints before QUBO compilation.

    Args:
        df: Classified asset DataFrame containing codes, managers, investment
            types, asset classes, and risk levels.
        client_template: Validated profile dictionary, normally returned by
            :func:`client_profile`.
        variable_mapping: DataFrame returned by :func:`weight_encoding`.
        similarity_groups: Optional iterable of high-similarity asset-code
            groups.
        strategy: Constraint policy. Version 1 supports ``document_rules``.
        manager_limit: Maximum combined portfolio weight per repeated manager.
        investment_type_limit: Maximum combined weight per repeated secondary
            investment type.
        similarity_group_limit: Maximum combined weight per similarity group.

    Returns:
        A dictionary with ``constraints`` in bounded sparse-linear form,
        ``fixed_variables`` that must equal zero, and a family-count summary.
        No penalties or slack variables are introduced at this stage.

    Raises:
        TypeError: If a required container has the wrong type.
        ValueError: If inputs are missing, misaligned, duplicated, or outside
            supported bounds.
        NotImplementedError: If the requested strategy is unavailable.
    """
    _validate_constraint_data(
        df,
        client_template,
        variable_mapping,
        similarity_groups,
        manager_limit,
        investment_type_limit,
        similarity_group_limit,
    )

    if strategy == 'document_rules':
        constraints = [_build_budget_constraint(variable_mapping)]
        constraints.extend(_build_single_weight_level_constraints(variable_mapping))
        constraints.append(
            _build_holding_count_constraint(
                variable_mapping,
                client_template['holding_count'],
            )
        )

        asset_cap_constraints, fixed_variables = _build_single_asset_cap_constraints(
            variable_mapping,
            client_template['single_asset_cap'],
        )
        constraints.extend(asset_cap_constraints)
        constraints.extend(
            _build_asset_class_constraints(df, client_template, variable_mapping)
        )
        constraints.append(
            _build_r5_constraint(df, client_template, variable_mapping)
        )
        constraints.extend(
            _build_manager_constraints(df, variable_mapping, manager_limit)
        )
        constraints.extend(
            _build_investment_type_constraints(
                df,
                variable_mapping,
                investment_type_limit,
            )
        )
        constraints.extend(
            _build_similarity_group_constraints(
                variable_mapping,
                similarity_groups or [],
                similarity_group_limit,
            )
        )

        _validate_generated_constraints(
            constraints,
            fixed_variables,
            variable_mapping,
        )
        return {
            'constraints': constraints,
            'fixed_variables': fixed_variables,
            'summary': _build_constraint_summary(constraints, fixed_variables),
        }

    raise NotImplementedError(f"Strategy '{strategy}' is not implemented.")


def _validate_constraint_data(
    df,
    client_template,
    variable_mapping,
    similarity_groups,
    manager_limit,
    investment_type_limit,
    similarity_group_limit,
):
    """Validate constraint data."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError('df must be a pandas DataFrame.')
    if df.empty:
        raise ValueError('df must contain at least one asset.')

    required_columns = {
        'code',
        'fund_manager',
        'investment_type_secondary',
        'asset_class',
        'risk_level',
    }
    missing = sorted(required_columns.difference(df.columns))
    if missing:
        raise ValueError(f'Missing constraint columns: {missing}')

    codes = df['code'].astype('string').str.strip()
    if codes.isna().any() or codes.eq('').any():
        raise ValueError("Column 'code' must not contain missing or blank values.")
    if codes.duplicated().any():
        raise ValueError("Column 'code' must contain unique values.")

    for column in required_columns.difference({'code'}):
        if df[column].isna().any():
            raise ValueError(f"Column '{column}' must not contain missing values.")

    required_template_fields = {
        'holding_count',
        'single_asset_cap',
        'asset_class_ranges',
        'r5_cap',
    }
    if not isinstance(client_template, dict):
        raise TypeError('client_template must be a dictionary.')
    missing_template_fields = sorted(
        required_template_fields.difference(client_template)
    )
    if missing_template_fields:
        raise ValueError(
            f'Missing client template fields: {missing_template_fields}'
        )

    _validate_variable_mapping(variable_mapping, codes.tolist())
    _validate_similarity_groups(similarity_groups or [], set(codes.tolist()))

    for name, limit in (
        ('manager_limit', manager_limit),
        ('investment_type_limit', investment_type_limit),
        ('similarity_group_limit', similarity_group_limit),
    ):
        if not isinstance(limit, (int, float)) or not math.isfinite(limit):
            raise TypeError(f'{name} must be a finite number.')
        if not 0 <= limit <= 1:
            raise ValueError(f'{name} must be between 0 and 1.')


def _validate_variable_mapping(variable_mapping, asset_codes):
    """Validate variable mapping."""
    if not isinstance(variable_mapping, pd.DataFrame):
        raise TypeError('variable_mapping must be a pandas DataFrame.')
    if variable_mapping.empty:
        raise ValueError('variable_mapping must not be empty.')

    required = {'variable_name', 'code', 'weight_level'}
    missing = sorted(required.difference(variable_mapping.columns))
    if missing:
        raise ValueError(f'Missing variable mapping columns: {missing}')
    if variable_mapping['variable_name'].isna().any():
        raise ValueError('Variable names must not contain missing values.')
    if variable_mapping['variable_name'].duplicated().any():
        raise ValueError('Variable names must be unique.')

    mapping = variable_mapping.copy()
    mapping['code'] = mapping['code'].astype('string').str.strip()
    mapping['weight_level'] = pd.to_numeric(mapping['weight_level'], errors='coerce')
    if mapping[['code', 'weight_level']].isna().any().any():
        raise ValueError('Variable mapping contains missing codes or weight levels.')
    if (mapping['weight_level'] < 0).any() or (mapping['weight_level'] > 1).any():
        raise ValueError('Variable mapping weight levels must be between 0 and 1.')

    mapped_codes = set(mapping['code'])
    expected_codes = set(asset_codes)
    if mapped_codes != expected_codes:
        missing_codes = sorted(expected_codes.difference(mapped_codes))
        extra_codes = sorted(mapped_codes.difference(expected_codes))
        raise ValueError(
            'Variable mapping codes do not match assets. '
            f'Missing: {missing_codes}; extra: {extra_codes}'
        )

    expected_levels = None
    for code, group in mapping.groupby('code', observed=True):
        levels = tuple(sorted(group['weight_level'].tolist()))
        if len(levels) != len(set(levels)):
            raise ValueError(f"Asset '{code}' has duplicate weight levels.")
        if not levels or levels[0] != 0:
            raise ValueError(f"Asset '{code}' must have a zero weight level.")
        if expected_levels is None:
            expected_levels = levels
        elif levels != expected_levels:
            raise ValueError('All assets must use the same weight levels.')


def _validate_similarity_groups(similarity_groups, asset_codes):
    """Validate similarity groups."""
    if not isinstance(similarity_groups, (list, tuple)):
        raise TypeError('similarity_groups must be a list or tuple of groups.')

    for group_index, group in enumerate(similarity_groups):
        if not isinstance(group, (list, tuple, set)):
            raise TypeError(f'Similarity group {group_index} must be an iterable of codes.')
        codes = [str(code).strip() for code in group]
        if len(codes) != len(set(codes)):
            raise ValueError(f'Similarity group {group_index} contains duplicate codes.')
        unknown = sorted(set(codes).difference(asset_codes))
        if unknown:
            raise ValueError(
                f'Similarity group {group_index} contains unknown codes: {unknown}'
            )


def _build_budget_constraint(variable_mapping):
    """Build budget constraint."""
    return _make_constraint(
        name='budget',
        family='budget',
        terms=_build_weight_terms(variable_mapping),
        lower_bound=1.0,
        upper_bound=1.0,
    )


def _build_single_weight_level_constraints(variable_mapping):
    """Build single weight level constraints."""
    constraints = []
    for code, group in variable_mapping.groupby('code', sort=False, observed=True):
        terms = {name: 1.0 for name in group['variable_name']}
        constraints.append(
            _make_constraint(
                name=f'one_weight_level::{code}',
                family='one_weight_level',
                terms=terms,
                lower_bound=1.0,
                upper_bound=1.0,
            )
        )
    return constraints


def _build_holding_count_constraint(variable_mapping, holding_count):
    """Build holding count constraint."""
    if (
        not isinstance(holding_count, (tuple, list))
        or len(holding_count) != 2
        or not all(isinstance(value, int) for value in holding_count)
    ):
        raise ValueError('holding_count must be an integer (minimum, maximum) pair.')

    minimum, maximum = holding_count
    asset_count = variable_mapping['code'].nunique()
    if minimum <= 0 or minimum > maximum:
        raise ValueError('holding_count must satisfy 0 < minimum <= maximum.')
    if minimum > asset_count:
        raise ValueError(
            f'holding_count minimum {minimum} exceeds asset count {asset_count}.'
        )

    selected = variable_mapping.loc[variable_mapping['weight_level'] > 0]
    terms = {name: 1.0 for name in selected['variable_name']}
    return _make_constraint(
        name='holding_count',
        family='holding_count',
        terms=terms,
        lower_bound=float(minimum),
        upper_bound=float(min(maximum, asset_count)),
    )


def _build_single_asset_cap_constraints(variable_mapping, cap):
    """Build single asset cap constraints."""
    if not isinstance(cap, (int, float)) or not 0 <= cap <= 1:
        raise ValueError('single_asset_cap must be between 0 and 1.')

    constraints = []
    fixed_variables = []
    for code, group in variable_mapping.groupby('code', sort=False, observed=True):
        constraints.append(
            _make_constraint(
                name=f'single_asset_cap::{code}',
                family='single_asset_cap',
                terms=_build_weight_terms(group),
                lower_bound=None,
                upper_bound=float(cap),
            )
        )
        fixed_variables.extend(
            group.loc[group['weight_level'] > cap, 'variable_name'].tolist()
        )
    return constraints, sorted(set(fixed_variables))


def _build_asset_class_constraints(df, client_template, variable_mapping):
    """Build asset class constraints."""
    constraints = []
    ranges = client_template['asset_class_ranges']
    if not isinstance(ranges, dict):
        raise TypeError('asset_class_ranges must be a dictionary.')

    for asset_class, bounds in ranges.items():
        if (
            not isinstance(bounds, (tuple, list))
            or len(bounds) != 2
            or not 0 <= bounds[0] <= bounds[1] <= 1
        ):
            raise ValueError(
                f"Asset class range '{asset_class}' must satisfy 0 <= min <= max <= 1."
            )

        codes = df.loc[df['asset_class'].astype('string') == asset_class, 'code']
        terms = _build_group_weight_terms(variable_mapping, codes)
        constraints.append(
            _make_constraint(
                name=f'asset_class::{asset_class}',
                family='asset_class',
                terms=terms,
                lower_bound=float(bounds[0]),
                upper_bound=float(bounds[1]),
            )
        )
    return constraints


def _build_r5_constraint(df, client_template, variable_mapping):
    """Build r5 constraint."""
    cap = client_template['r5_cap']
    if not isinstance(cap, (int, float)) or not 0 <= cap <= 1:
        raise ValueError('r5_cap must be between 0 and 1.')

    # 按技术文档的窄口径，只统计风险等级明确为 R5 的资产。
    r5_codes = df.loc[df['risk_level'].astype('string') == 'R5', 'code']
    return _make_constraint(
        name='r5_cap',
        family='r5_cap',
        terms=_build_group_weight_terms(variable_mapping, r5_codes),
        lower_bound=None,
        upper_bound=float(cap),
    )


def _build_manager_constraints(df, variable_mapping, limit):
    """Build manager constraints."""
    return _build_group_cap_constraints(
        df,
        variable_mapping,
        group_column='fund_manager',
        limit=limit,
        family='manager_cap',
    )


def _build_investment_type_constraints(df, variable_mapping, limit):
    """Build investment type constraints."""
    return _build_group_cap_constraints(
        df,
        variable_mapping,
        group_column='investment_type_secondary',
        limit=limit,
        family='investment_type_cap',
    )


def _build_group_cap_constraints(
    df,
    variable_mapping,
    group_column,
    limit,
    family,
):
    """Build group cap constraints."""
    constraints = []
    for group_name, group in df.groupby(group_column, sort=False, observed=True):
        if len(group) < 2:
            continue
        terms = _build_group_weight_terms(variable_mapping, group['code'])
        constraints.append(
            _make_constraint(
                name=f'{family}::{group_name}',
                family=family,
                terms=terms,
                lower_bound=None,
                upper_bound=float(limit),
            )
        )
    return constraints


def _build_similarity_group_constraints(variable_mapping, groups, limit):
    """Build similarity group constraints."""
    constraints = []
    for group_index, group in enumerate(groups):
        codes = list(dict.fromkeys(str(code).strip() for code in group))
        if len(codes) < 2:
            continue
        constraints.append(
            _make_constraint(
                name=f'similarity_group_cap::{group_index}',
                family='similarity_group_cap',
                terms=_build_group_weight_terms(variable_mapping, codes),
                lower_bound=None,
                upper_bound=float(limit),
            )
        )
    return constraints


def _build_weight_terms(variable_mapping):
    """Build weight terms."""
    return {
        row.variable_name: float(row.weight_level)
        for row in variable_mapping.itertuples(index=False)
        if float(row.weight_level) != 0
    }


def _build_group_weight_terms(variable_mapping, codes):
    """Build group weight terms."""
    code_set = {str(code).strip() for code in codes}
    selected = variable_mapping.loc[
        variable_mapping['code'].astype('string').isin(code_set)
    ]
    return _build_weight_terms(selected)


def _make_constraint(
    name,
    family,
    terms,
    lower_bound=None,
    upper_bound=None,
):
    """Create constraint."""
    if lower_bound is not None and upper_bound is not None:
        constraint_type = 'equality' if lower_bound == upper_bound else 'range'
    elif upper_bound is not None:
        constraint_type = 'upper_bound'
    else:
        constraint_type = 'lower_bound'

    return {
        'name': name,
        'family': family,
        'type': constraint_type,
        'terms': terms,
        'lower_bound': lower_bound,
        'upper_bound': upper_bound,
    }


def _validate_generated_constraints(
    constraints,
    fixed_variables,
    variable_mapping,
):
    """Validate generated constraints."""
    valid_variables = set(variable_mapping['variable_name'])
    names = [constraint['name'] for constraint in constraints]
    if len(names) != len(set(names)):
        raise ValueError('Generated constraint names must be unique.')

    for constraint in constraints:
        lower_bound = constraint['lower_bound']
        upper_bound = constraint['upper_bound']
        if (
            lower_bound is not None
            and upper_bound is not None
            and lower_bound > upper_bound
        ):
            raise ValueError(
                f"Constraint '{constraint['name']}' has invalid bounds."
            )

        unknown_variables = set(constraint['terms']).difference(valid_variables)
        if unknown_variables:
            raise ValueError(
                f"Constraint '{constraint['name']}' references unknown variables: "
                f'{sorted(unknown_variables)}'
            )
        if any(not math.isfinite(value) for value in constraint['terms'].values()):
            raise ValueError(
                f"Constraint '{constraint['name']}' has non-finite coefficients."
            )
        if not constraint['terms'] and lower_bound is not None and lower_bound > 0:
            raise ValueError(
                f"Constraint '{constraint['name']}' requires unavailable assets."
            )

    unknown_fixed = set(fixed_variables).difference(valid_variables)
    if unknown_fixed:
        raise ValueError(f'Unknown fixed variables: {sorted(unknown_fixed)}')


def _build_constraint_summary(constraints, fixed_variables):
    """Build constraint summary."""
    family_counts = {}
    for constraint in constraints:
        family = constraint['family']
        family_counts[family] = family_counts.get(family, 0) + 1
    return {
        'constraint_count': len(constraints),
        'fixed_variable_count': len(fixed_variables),
        'constraints_by_family': family_counts,
    }
