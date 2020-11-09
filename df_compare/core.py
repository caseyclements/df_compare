"""Core Comparison of two dataframes"""
import pandas as pd


def describe_row_diffs(df_obs, df_exp):
    return f'df_obs has {len(df_obs)}. df_exp has {len(df_exp)}'


def describe_column_diffs(cols_obs, cols_exp):
    cols_not_in_exp = cols_obs - cols_exp
    cols_not_in_obs = cols_exp - cols_obs
    return f'cols_not_in_obs: {cols_not_in_obs}. cols_not_in_exp: {cols_not_in_exp}.'


def df_compare(df_obs, df_exp, *args, **kwargs):
    """
    TODO Docstring
    """

    # TODO IMPLEMENT
    #  1. Number of rows
    #  2. Column names
    #  3. dtypes
    #  4. Index
    #  5. Integers
    #  6. Floats
    #  7. Datetimes
    #  8. Objects (typically strings)
    #  9. NaNs

    diffs = {}
    assert isinstance(df_obs, pd.DataFrame)
    assert isinstance(df_exp, pd.DataFrame)

    # 1. Number of Rows
    n_rows_obs = len(df_obs)
    n_rows_exp = len(df_exp)
    if n_rows_obs != n_rows_exp:
        diffs['rows'] = describe_row_diffs(df_obs, df_exp)

    # 2. Column Names
    cols_obs = set(df_obs.columns)
    cols_exp = set(df_exp.columns)
    if cols_obs != cols_obs:
        diffs['columns'] = describe_column_diffs(cols_obs, cols_exp)

    return diffs


