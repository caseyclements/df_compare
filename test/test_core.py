from df_compare import df_compare
import pandas as pd
import numpy as np
import pytest
import warnings
import logging
logging.basicConfig(level=logging.WARNING)

warnings.filterwarnings(action='ignore', category=pd.core.common.SettingWithCopyWarning)

@pytest.fixture(scope='session')
def base_dict():
    """Simple dict to make DataFrames from. Has 4 dtypes"""
    d = {
        'i': [0, 1, 2],
        'f': [0.0, np.nan, 2.0],
        'd': pd.Series(['2018-01-01', '2019-01-01', '2020-01-01'], dtype='datetime64[ns]'),
        's': ['0', '1', '2'],
        'b': [False, True, True],
    }
    return d


@pytest.fixture(scope='session')
def base_df(base_dict):
    """Simple DataFrame to make tests from. Has 4 dtypes"""
    return pd.DataFrame(base_dict)


def test_nrows(base_df):
    """ Test number of rows when same, different, and its description"""
    # Match
    df_obs = base_df.copy()
    diffs = df_compare(df_obs=df_obs, df_exp=base_df)
    assert 'rows' not in diffs

    # Duplicate values in index
    df_obs2 = pd.concat([df_obs, base_df])
    diffs2 = df_compare(df_obs=df_obs2, df_exp=base_df)
    assert 'rows' in diffs2
    assert 'index' in diffs2
    assert not diffs2['complete']
    assert diffs2['rows'] == f'rows differ: df_obs has {len(df_obs2)}. df_exp has {len(base_df)}'

    # Extra rows, but some match index values
    df_obs3 = df_obs2.reset_index(drop=True)
    diffs3 = df_compare(df_obs=df_obs3, df_exp=base_df)
    assert 'rows' in diffs3
    assert 'index' in diffs3
    assert 'int' not in diffs3
    assert diffs3['complete']


def test_columns(base_df):
    df_obs = base_df.copy()
    df_obs['s_copy'] = df_obs['s']  # Add a column

    diffs = df_compare(df_obs=df_obs, df_exp=base_df)
    assert 'rows' not in diffs
    assert diffs['columns'] == f"columns are different: 1 total. (first few) not_in_obs: []. not_in_exp: ['s_copy']."


def test_dtypes(base_df):
    df_obs = base_df.copy()
    df_obs['s'] = df_obs['s'].astype(int)  # Switch type

    diffs = df_compare(df_obs=df_obs, df_exp=base_df)
    assert 'rows' not in diffs
    assert 'columns' not in diffs
    assert diffs.get('dtypes') is not None


def test_index(base_df):
    df_exp = base_df.copy()
    df_exp = df_exp.set_index('s')

    df_obs = base_df.copy()
    df_obs['s'].iloc[1] = '3'
    df_obs = df_obs.set_index('s')

    diffs = df_compare(df_obs=df_obs, df_exp=df_exp)
    assert 'rows' not in diffs
    assert 'columns' not in diffs
    assert diffs.get('index') is not None
    assert '3' in diffs['index']


def test_ints(base_df):
    """Test comparison of integers."""
    df_obs = base_df.copy()
    df_obs['i'].iloc[:2] = [3, 2]  # Change a couple values
    diffs = df_compare(df_obs=df_obs, df_exp=base_df)
    assert all([k not in diffs for k in ['rows', 'columns', 'index']])
    assert diffs.get('int') is not None
    assert isinstance(diffs['int'], str)


def test_bools(base_df):
    df_obs = base_df.copy()
    df_obs['b'].iloc[:2] = [True, True]  # Change a couple values
    diffs = df_compare(df_obs=df_obs, df_exp=base_df)
    assert all([k not in diffs for k in ['rows', 'columns', 'index', 'int']])
    assert diffs.get('bool') is not None
    assert isinstance(diffs['bool'], str)


def test_floats(base_df):
    df_obs = base_df.copy()
    df_obs['f'].iloc[2] = 2.5
    diffs = df_compare(df_obs=df_obs, df_exp=base_df)
    assert all([k not in diffs for k in ['rows', 'columns', 'index', 'int', 'bool']])
    assert 'floats are different:' in diffs.get('float')


def test_nan(base_df):
    # No difference when values are different, but nan is same place
    df_obs = base_df.copy()
    df_obs['f'] = [10, np.nan, 10.0]
    diffs = df_compare(df_obs=df_obs, df_exp=base_df)
    assert 'nan' not in diffs

    # Differences, plus NaT, and optionally Nullable Integer
    df_exp = base_df.copy()
    if pd.__version__ >= '1.0':
        df_exp['Int64'] = pd.array([0, 1, None])
        df_obs['Int64'] = pd.array([0, 1, 2], dtype=pd.Int64Dtype())
    df_obs.loc[0, 'd'] = pd.NaT

    diffs = df_compare(df_obs=df_obs, df_exp=df_exp)
    assert all([k not in diffs for k in ['rows', 'columns', 'index', 'int', 'bool']])
    assert 'float' in diffs
    assert 'nan' in diffs
    assert 'nans are different:' in diffs.get('nan')
