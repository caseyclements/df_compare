"""Core Comparison of two dataframes"""
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger('df_compare')


def describe_row_diffs(df_obs, df_exp):
    return f'rows differ: df_obs has {len(df_obs)}. df_exp has {len(df_exp)}'


def describe_column_diffs(cols_obs, cols_exp):
    cols_not_in_exp = sorted(cols_obs - cols_exp)
    cols_not_in_obs = sorted(cols_exp - cols_obs)
    return f'columns differ: cols_not_in_obs: {cols_not_in_obs}. cols_not_in_exp: {cols_not_in_exp}.'


def describe_dtype_diffs(df_dtypes_diff, n_rows=5):
    return f'dtypes differ: first few rows of dtype diffs:\n{repr(df_dtypes_diff.iloc[:n_rows])}'


def describe_index_diffs(index_obs, index_exp):
    idx_not_in_exp = index_obs - index_exp
    idx_not_in_obs = index_exp - index_obs
    return f'indexes differ: idx_not_in_obs: {idx_not_in_obs}. idx_not_in_exp: {idx_not_in_exp}.'


def describe_int_diffs(dfx_int_obs, dfx_int_exp, mask_int, n_rows=5):
    df_prev_obs = dfx_int_obs.loc[mask_int].iloc[:n_rows]
    df_prev_exp = dfx_int_exp.loc[mask_int].iloc[:n_rows]
    return '\n'.join([f'ints differ: first few rows of int diffs:',
                      'observed', repr(df_prev_obs), 'expected', repr(df_prev_exp)])


def describe_bool_diffs(dfx_bool_obs, dfx_bool_exp, mask_bool, n_rows=5):
    df_prev_obs = dfx_bool_obs.loc[mask_bool].iloc[:n_rows]
    df_prev_exp = dfx_bool_exp.loc[mask_bool].iloc[:n_rows]
    return '\n'.join([f'bools differ: first few rows of bool diffs:',
                      'observed', repr(df_prev_obs), 'expected', repr(df_prev_exp)])


def df_compare(df_obs, df_exp, n_show=5, *args, **kwargs):
    """ Descriptive comparison of two dataframes along a number of dimensions.

    For each difference, provides a description to aid debugging.
    Results are packaged in a dictionary.

    When columns differ, comparison continues along the ones common to both.
    When indices differ, comparison continues along the rows common to both.
    When types are compared, (e.g. int, object) include any row with a difference

    Where equality is possible, we use ==. For floats, we provide a tolerance.

    Compares the following features of the two DataFrames
      1. number of rows ('rows'): Simple comparison of length.
      2. columns ('columns'): Comparison of column names. Indifferent to order.
      3. dtypes ('dtypes'): Types of common columns
      4. indexes ('index'): Equality. Continue with intersection or stop here.
      5. integers ('int'): Select int columns and compare by equality.
      6. booleans ('bool'): Select bool columns and compare by equality.
      7. floats ('float'): Select float columns and compare with numpy.is_close
      8. datetimes: ('datetime'): Select date-like columns and compare by equality.
      9. objects / strings ('object'): Select object columns and compare by equality.
      10. NaNs ('nan'): Compare location of NaNs

    :param df_obs: (DataFrame) 1st to compare
    :param df_exp: (DataFrame) 2nd to compare
    :param n_show: number of examples used to preview differences (typically rows)
    :return: (dict) dictionary describing differences.

    TODO IMPLEMENT
      6. Floats (need tolerance)
      7. Datetimes
      8. Objects (typically strings)
      9. NaNs
      ..
      10. sort_by
      11. ignore_index
      12. option to stop after index


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
        logger.warning(diffs['rows'])

    # 2. Columns
    cols_obs = set(df_obs.columns)
    cols_exp = set(df_exp.columns)
    if cols_obs != cols_exp:
        diffs['columns'] = describe_column_diffs(cols_obs, cols_exp)
        logger.warning(diffs['columns'])

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
    mask_dtypes = df_dtypes.obs != df_dtypes.exp
    df_dtypes_diff = df_dtypes.loc[mask_dtypes]
    if len(df_dtypes_diff) > 0:
        diffs['dtypes'] = describe_dtype_diffs(df_dtypes_diff, n_rows=n_show)
        logger.warning(diffs['dtypes'])
        #  Continue with Intersection
        dfx_obs = dfx_obs.loc[:, ~mask_dtypes]
        dfx_exp = dfx_exp.loc[:, ~mask_dtypes]

    # 4. Index
    index_obs = set(df_obs.index)
    index_exp = set(df_exp.index)
    if ('rows' in diffs) or (index_obs != index_exp):
        diffs['index'] = describe_index_diffs(index_obs, index_exp)
        logger.warning(diffs['index'])
        #  Continue with Intersection of columns
        index_common = index_obs.intersection(index_exp)
        dfx_obs = dfx_obs.loc[index_common]
        dfx_exp = dfx_exp.loc[index_common]
        if dfx_obs.shape != dfx_exp.shape:
            logger.error('Length of DataFrames differ after index intersection. To continue, index values must be unique')
            diffs['complete'] = False
            return diffs

    # 5. Integers
    dfx_int_obs = dfx_obs.select_dtypes(include=['int'])
    dfx_int_exp = dfx_exp.select_dtypes(include=['int'])
    mask_int = (dfx_int_obs != dfx_int_exp).any(axis=1)
    if np.any(mask_int):
        diffs['int'] = describe_int_diffs(dfx_int_obs, dfx_int_exp, mask_int)
        logger.warning(diffs['int'])

    # 6. Booleans
    dfx_bool_obs = dfx_obs.select_dtypes(include=['bool'])
    dfx_bool_exp = dfx_exp.select_dtypes(include=['bool'])
    mask_bool = (dfx_bool_obs != dfx_bool_exp).any(axis=1)
    if np.any(mask_bool):
        diffs['bool'] = describe_bool_diffs(dfx_bool_obs, dfx_bool_exp, mask_bool)
        logger.warning(diffs['bool'])


    diffs['complete'] = True
    return diffs
