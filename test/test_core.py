from df_compare import df_compare
import pandas as pd
import numpy as np
import pytest


@pytest.fixture(scope='session')
def base_dict():
    """Simple dict to make DataFrames from. Has 4 dtypes"""
    d = {
        "i": [0, 1, 2],
        "f": [0.0, np.nan, 2.0],
        "d": ["2018-01-01", "2019-01-01", "2020-01-01"],
        "s": ["0", "1", "2"]
    }
    return d


@pytest.fixture(scope='session')
def base_df(base_dict):
    """Simple DataFrame to make tests from. Has 4 dtypes"""
    return pd.DataFrame(base_dict)


def test_nrows(base_df):
    """ Test number of rows when same, different, and its description"""
    df_obs = base_df.copy()
    diffs = df_compare(df_obs=df_obs, df_exp=base_df)
    assert 'rows' not in diffs

    df_obs2 = pd.concat([df_obs, base_df])
    diffs2 = df_compare(df_obs=df_obs2, df_exp=base_df)
    desc = diffs2.get('rows')
    assert desc == f'df_obs has {len(df_obs2)}. df_exp has {len(base_df)}'


def test_columns(base_df):
    df_obs = base_df.copy()
    df_obs['s_copy'] = df_obs['s']  # Add a column

    diffs = df_compare(df_obs=df_obs, df_exp=base_df)
    assert 'rows' not in diffs
    assert diffs.get('columns') is not None
    assert diffs['columns'] == f"cols_not_in_obs: []. cols_not_in_exp: ['s_copy']."


def test_dtypes(base_df):
    df_obs = base_df.copy()
    df_obs['s'] = df_obs['s'].astype(int)  # Switch type

    diffs = df_compare(df_obs=df_obs, df_exp=base_df)
    assert 'rows' not in diffs
    assert 'columns' not in diffs
    assert diffs.get('dtypes') is not None
