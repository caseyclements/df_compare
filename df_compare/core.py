"""Core Comparison of two dataframes"""
import pandas as pd


def describe_row_diffs(df_obs, df_exp):
    return f'df_obs has {len(df_obs)}. df_exp has {len(df_exp)}'


def describe_column_diffs(cols_obs, cols_exp):
    cols_not_in_exp = sorted(cols_obs - cols_exp)
    cols_not_in_obs = sorted(cols_exp - cols_obs)
    return f'cols_not_in_obs: {cols_not_in_obs}. cols_not_in_exp: {cols_not_in_exp}.'


def describe_dtype_diffs(df_dtypes_diff, n_rows=5):
    return f'first {n_rows} rows of dtype diffs:\n {repr(df_dtypes_diff.iloc[:n_rows])}'


def describe_index_diffs(index_obs, index_exp):
    idx_not_in_exp = index_obs - index_exp
    idx_not_in_obs = index_exp - index_obs
    return f'idx_not_in_obs: {idx_not_in_obs}. idx_not_in_exp: {idx_not_in_exp}.'


def df_compare(df_obs, df_exp, n_show=5, *args, **kwargs):
    """
    TODO Docstring

    :param df_obs: (DataFrame) 1st to compare
    :param df_exp: (DataFrame) 2nd to compare
    :param n_show: number of examples used to preview differences (typically rows)
    :return: (dict) dictionary describing differences.

    TODO IMPLEMENT
      5. Integers
      6. Floats
      7. Datetimes
      8. Objects (typically strings)
      9. NaNs

    As we proceed with tests, in order for further tests to make sense,
    we take the intersections across axes. Of columns first. Then optionally, of rows.
    """

    diffs = {}
    assert isinstance(df_obs, pd.DataFrame)
    assert isinstance(df_exp, pd.DataFrame)

    # 1. Number of Rows
    n_rows_obs = len(df_obs)
    n_rows_exp = len(df_exp)
    if n_rows_obs != n_rows_exp:
        diffs['rows'] = describe_row_diffs(df_obs, df_exp)

    # 2. Columns
    cols_obs = set(df_obs.columns)
    cols_exp = set(df_exp.columns)
    if cols_obs != cols_exp:
        diffs['columns'] = describe_column_diffs(cols_obs, cols_exp)

    #  Continue with Intersection of columns
    if cols_obs != cols_obs:
        cols_common = cols_obs.intersection(cols_exp)
        dfx_obs = df_obs[cols_common]
        dfx_exp = df_exp[cols_common]
    else:
        dfx_obs = df_obs
        dfx_exp = df_exp

    # 3. dtypes
    df_dtypes = pd.concat([dfx_obs.dtypes, dfx_exp.dtypes], axis=1)
    df_dtypes.columns = ['obs', 'exp']
    df_dtypes_diff = df_dtypes.loc[df_dtypes.obs != df_dtypes.exp]
    if len(df_dtypes_diff) > 0:
        diffs['dtypes'] = describe_dtype_diffs(df_dtypes_diff, n_rows=n_show)

    # 4. Index
    index_obs = set(df_obs.index)
    index_exp = set(df_exp.index)
    if ('rows' in diffs) or (index_obs != index_exp):
        diffs['index'] = describe_index_diffs(index_obs, index_exp)
        #  Continue with Intersection of columns
        index_common = index_obs.intersection(index_exp)
        dfx_obs = dfx_obs.loc[index_common]
        dfx_exp = dfx_exp.loc[index_common]

    return diffs

