def _corr(df):
    """
    Calculate the correlation matrix of a DataFrame.

    Parameters:
    df (pd.DataFrame): The input DataFrame.

    Returns:
    pd.DataFrame: The correlation matrix.
    """
    return df.corr()



def get_corr_mat(df, name_col, drop_cols=[]):
    
    clean_df = df.drop(columns=drop_cols, errors='ignore')
    corr_mat = _corr(clean_df)
    return corr_mat



