def my_read(filename):
    import pandas as pd

    meta_out_cols = ['序号', '证券简称', '基金经理']
    meta_cols = ['序号', '代码', '证券简称', '基金经理', '投资类型(二级)']

    df = pd.read_excel(filename)

    # 工作簿末尾包含空行及“数据来源：Wind”，应在拆分数据前统一删除，
    # 否则 raw_df 和 meta 经过各自清洗后会出现行数/索引不一致。
    if '代码' not in df.columns:
        raise ValueError("The input file must contain a '代码' column.")
    code = df['代码'].astype('string').str.strip()
    df = df.loc[code.notna() & code.ne('')].reset_index(drop=True)

    meat_df = df.drop(columns=meta_out_cols, errors='ignore')
    meta_df = df[[col for col in meta_cols if col in df.columns]]

    return meat_df, meta_df

def basic_formatting(df):
    import pandas as pd

    def repair_text(value):
        """修复本文件中 GBK 文本被当作 Latin-1 解码造成的乱码。"""
        if not isinstance(value, str):
            return value
        value = value.strip()
        try:
            repaired = value.encode('latin1').decode('gbk')
        except (UnicodeEncodeError, UnicodeDecodeError):
            return value
        # 只在修复结果确实包含中文时替换，避免误伤正常英文。
        return repaired if any('\u4e00' <= char <= '\u9fff' for char in repaired) else value

    periods = {
        '近3月': '3m',
        '近6月': '6m',
        '近1年': '1y',
        '近2年': '2y',
        '近3年': '3y',
    }

    metrics = {
        'Sharpe': 'sharpe',
        'Sortino': 'sortino',
        'Calmar': 'calmar',
        '收益标准差': 'return_std',
        '下行标准差': 'downside_std',
        '最大回撤': 'max_drawdown',
        '收益率': 'return',
    }

    column_names = {
        '序号': 'id',
        '代码': 'code',
        '证券简称': 'security_name',
        '基金经理': 'fund_manager',
        '投资类型(二级)': 'investment_type_secondary',
        '近3年年化收益率': '3y_annualized_return',
    }

    column_names.update({
        period_cn + metric_cn: f'{period_en}_{metric_en}'
        for period_cn, period_en in periods.items()
        for metric_cn, metric_en in metrics.items()
    })

    new_df = df.copy()
    new_df.columns = [repair_text(column) for column in new_df.columns]
    new_df = new_df.rename(columns=column_names)

    duplicates = new_df.columns[new_df.columns.duplicated()].tolist()
    if duplicates:
        raise ValueError(f'格式化后存在重复列名: {duplicates}')

    metric_columns = [
        column for column in new_df.columns
        if column.endswith((
            '_sharpe', '_sortino', '_calmar', '_return_std',
            '_downside_std', '_max_drawdown', '_return',
            '_annualized_return',
        ))
    ]
    if not metric_columns:
        raise ValueError('未找到可识别的指标列，请检查输入表头')

    # 修复文本内容、空白和统一缺失值。
    text_columns = [
        'code', 'security_name', 'fund_manager',
        'investment_type_secondary',
    ]
    missing_markers = {'': pd.NA, '-': pd.NA, '--': pd.NA, 'N/A': pd.NA}
    for column in text_columns:
        if column in new_df:
            new_df[column] = (
                new_df[column]
                .map(repair_text, na_action='ignore')
                .astype('string')
                .str.strip()
                .replace(missing_markers)
            )

    # 原文件最后两行是空行和“数据来源：Wind”，不属于资产记录。
    if 'code' in new_df:
        new_df = new_df.loc[new_df['code'].notna()].copy()

    # 标识列
    if 'id' in new_df:
        new_df['id'] = pd.to_numeric(new_df['id'], errors='coerce').astype('Int64')

    # 百分比列
    percentage_suffixes = (
        '_return',
        '_annualized_return',
        '_return_std',
        '_downside_std',
        '_max_drawdown',
    )

    for column in new_df.columns:
        if column.endswith(percentage_suffixes):
            values = (
                new_df[column].astype('string').str.strip()
                .replace(missing_markers)
            )
            has_percent = values.str.contains('%', na=False)

            numeric = pd.to_numeric(
                values.str.replace('%', '', regex=False)
                      .str.replace(',', '', regex=False),
                errors='coerce',
            )

            new_df[column] = numeric.where(~has_percent, numeric / 100)

    # 比率列
    ratio_suffixes = ('_sharpe', '_sortino', '_calmar')

    for column in new_df.columns:
        if column.endswith(ratio_suffixes):
            new_df[column] = pd.to_numeric(
                new_df[column].replace(missing_markers), errors='coerce'
            )

    # 分类列及其中英文映射
    category_dict = {
        'investment_type_secondary': {
            '普通股票型基金': 'equity',
            '偏股混合型基金': 'equity_hybrid',
            '灵活配置型基金': 'flexible_allocation',
            '被动指数型基金': 'passive_index',
            '增强指数型基金': 'enhanced_index',
            '可转换债券型基金': 'convertible_bond',
            '商品型基金': 'commodity',
            '混合债券型一级基金': 'hybrid_bond_primary',
            '混合债券型二级基金': 'hybrid_bond_secondary',
            '股票多空': 'equity_long_short',
            '货币市场型基金': 'money_market',
        }
    }

    for column, mapping in category_dict.items():
        if column in new_df:
            # replace 会保留未来新增的未知类别，避免 map 静默转为缺失值。
            new_df[column] = (
                new_df[column]
                .replace(mapping)
                .astype('category')
            )

    return new_df.reset_index(drop=True), category_dict
 

def meta_translate(meta_df):
    translation_dict = {
        '序号': 'id',
        '代码': 'code',
        '证券简称': 'security_name',
        '基金经理': 'fund_manager',
        '投资类型(二级)': 'investment_type_secondary',
    }

    translated_df = meta_df.rename(columns=translation_dict)
    return translated_df

def manager_formatting(manager_df, strategy='low_dim'):
    if 'code' in manager_df.columns:
        en = True
    elif '代码' in manager_df.columns:
        en = False
    else:
        raise ValueError("The input DataFrame must contain a 'code' column.")

    if strategy == 'low_dim':
        return _manager_low_dim(manager_df, en=en)

    raise NotImplementedError(f"Strategy '{strategy}' is not implemented.")



def _manager_low_dim(manager_df, en=True):
    import pandas as pd

    if not en:
        M_COL = '基金经理'
        CODE_COL = '代码'
    else:
        M_COL = 'fund_manager'
        CODE_COL = 'code'

    missing = [column for column in (CODE_COL, M_COL) if column not in manager_df]
    if missing:
        raise ValueError(f'The input DataFrame is missing columns: {missing}')

    result = manager_df.copy()
    result[CODE_COL] = result[CODE_COL].astype('string').str.strip()
    result = result.loc[result[CODE_COL].notna() & result[CODE_COL].ne('')].copy()

    def normalize_manager_team(value):
        """将多人共管名称规范为与顺序、分隔符无关的组合。"""
        if pd.isna(value):
            return pd.NA

        text = str(value).strip()
        if not text or text in {'-', '--', 'N/A'}:
            return pd.NA

        for separator in ('，', '、', ';', '；', '/', '／'):
            text = text.replace(separator, ',')

        managers = sorted({name.strip() for name in text.split(',') if name.strip()})
        return ','.join(managers) if managers else pd.NA

    normalized = result[M_COL].map(normalize_manager_team)

    # 排序后编号，使映射不依赖 DataFrame 当前行顺序，可稳定复现。
    categories = sorted(normalized.dropna().unique().tolist())
    manager_mapping = {manager: code for code, manager in enumerate(categories)}

    result[M_COL] = (
        normalized.map(manager_mapping)
        .fillna(-1)
        .astype('Int64')
    )

    return result.reset_index(drop=True), manager_mapping
