from copy import deepcopy


PROFILE_ALIASES = {
    'conservative': 'conservative',
    '保守型': 'conservative',
    'steady': 'steady',
    'moderate': 'steady',
    '稳健型': 'steady',
    'balanced': 'balanced',
    '平衡型': 'balanced',
    'aggressive': 'aggressive',
    '进取型': 'aggressive',
}


def client_profile(profile='steady', overrides=None):
    """Return an independent, validated client-allocation profile.

    Args:
        profile: Canonical profile name or supported alias. Canonical values are
            ``conservative``, ``steady``, ``balanced``, and ``aggressive``.
        overrides: Optional mapping of profile fields to replace. Nested
            ``asset_class_ranges`` overrides are merged by asset class.

    Returns:
        A deep-copied dictionary containing targets, holding-count bounds,
        concentration caps, asset-class ranges, and the R5 allocation cap.

    Raises:
        TypeError: If profile or override containers have invalid types.
        ValueError: If a name, field, bound, or range is unsupported.

    Notes:
        Templates are never returned by reference, so callers may safely modify
        the resulting dictionary without changing future calls.
    """
    profile_name = _validate_client_profile(profile)
    templates = _get_client_profile_templates()
    selected = deepcopy(templates[profile_name])

    if overrides is not None:
        selected = _apply_profile_overrides(selected, overrides)

    _validate_profile_values(selected)
    return selected


def _get_client_profile_templates():
    """Return client profile templates."""
    return {
        'conservative': {
            'name': 'conservative',
            'target_return': 0.035,
            'volatility_cap': 0.045,
            'drawdown_cap': 0.08,
            'holding_count': (5, 10),
            'single_asset_cap': 0.20,
            'asset_class_ranges': {
                'cash_money': (0.15, 0.35),
                'fixed_income': (0.60, 0.85),
                'mixed': (0.00, 0.10),
                'equity': (0.00, 0.10),
                'alternative': (0.00, 0.05),
            },
            'r5_cap': 0.00,
        },
        'steady': {
            'name': 'steady',
            'target_return': 0.055,
            'volatility_cap': 0.08,
            'drawdown_cap': 0.12,
            'holding_count': (8, 15),
            'single_asset_cap': 0.15,
            'asset_class_ranges': {
                'cash_money': (0.05, 0.20),
                'fixed_income': (0.45, 0.75),
                'mixed': (0.00, 0.25),
                'equity': (0.00, 0.30),
                'alternative': (0.00, 0.10),
            },
            'r5_cap': 0.10,
        },
        'balanced': {
            'name': 'balanced',
            'target_return': 0.075,
            'volatility_cap': 0.13,
            'drawdown_cap': 0.20,
            'holding_count': (10, 18),
            'single_asset_cap': 0.12,
            'asset_class_ranges': {
                'cash_money': (0.03, 0.15),
                'fixed_income': (0.25, 0.55),
                'mixed': (0.00, 0.30),
                'equity': (0.20, 0.50),
                'alternative': (0.00, 0.15),
            },
            'r5_cap': 0.25,
        },
        'aggressive': {
            'name': 'aggressive',
            'target_return': 0.10,
            'volatility_cap': 0.20,
            'drawdown_cap': 0.30,
            'holding_count': (12, 25),
            'single_asset_cap': 0.10,
            'asset_class_ranges': {
                'cash_money': (0.00, 0.10),
                'fixed_income': (0.10, 0.35),
                'mixed': (0.00, 0.30),
                'equity': (0.40, 0.75),
                'alternative': (0.00, 0.20),
            },
            'r5_cap': 0.50,
        },
    }


def _validate_client_profile(profile):
    """Validate client profile."""
    if not isinstance(profile, str):
        raise TypeError('profile must be a string.')

    normalized = profile.strip()
    if normalized not in PROFILE_ALIASES:
        raise ValueError(
            f"Unknown client profile '{profile}'. "
            f"Supported profiles: {sorted(set(PROFILE_ALIASES.values()))}"
        )
    return PROFILE_ALIASES[normalized]


def _apply_profile_overrides(profile, overrides):
    """Apply profile overrides."""
    if not isinstance(overrides, dict):
        raise TypeError('overrides must be a dictionary.')

    allowed = set(profile)
    unknown = set(overrides).difference(allowed)
    if unknown:
        raise ValueError(f'Unknown profile fields: {sorted(unknown)}')

    result = deepcopy(profile)
    for key, value in overrides.items():
        if key == 'asset_class_ranges':
            if not isinstance(value, dict):
                raise TypeError('asset_class_ranges override must be a dictionary.')
            unknown_classes = set(value).difference(result[key])
            if unknown_classes:
                raise ValueError(f'Unknown asset classes: {sorted(unknown_classes)}')
            result[key].update(value)
        else:
            result[key] = value
    return result


def _validate_profile_values(profile):
    """Validate profile values."""
    for field in ('target_return', 'volatility_cap', 'drawdown_cap', 'single_asset_cap', 'r5_cap'):
        value = profile[field]
        if not isinstance(value, (int, float)) or not 0 <= value <= 1:
            raise ValueError(f"Profile field '{field}' must be between 0 and 1.")

    holding_count = profile['holding_count']
    if (
        not isinstance(holding_count, (tuple, list))
        or len(holding_count) != 2
        or not all(isinstance(value, int) for value in holding_count)
        or holding_count[0] <= 0
        or holding_count[0] > holding_count[1]
    ):
        raise ValueError('holding_count must be a positive (minimum, maximum) pair.')

    for asset_class, bounds in profile['asset_class_ranges'].items():
        if (
            not isinstance(bounds, (tuple, list))
            or len(bounds) != 2
            or not all(isinstance(value, (int, float)) for value in bounds)
            or not 0 <= bounds[0] <= bounds[1] <= 1
        ):
            raise ValueError(
                f"Asset class range '{asset_class}' must satisfy 0 <= min <= max <= 1."
            )
