from df_compare import df_compare
import pandas as pd
import numpy as np
import datetime
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


@pytest.fixture
def character_set():
    """ Characters of comparison. All of the keys that may show up in df_compare."""
    return {'rows', 'columns', 'dtypes', 'index', 'int', 'bool', 'float',
            'datetime', 'object', 'nan',
            }


def test_nrows(base_df, character_set):
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
    assert all([key not in diffs for key in character_set - {'rows', 'index'}])
    assert diffs3['complete']


def test_columns(base_df, character_set):
    df_obs = base_df.copy()
    df_obs['s_copy'] = df_obs['s']  # Add a column

    diffs = df_compare(df_obs=df_obs, df_exp=base_df)
    assert all([key not in diffs for key in character_set - {'columns'}])
    assert diffs['columns'] == f"columns are different: 1 total. (first few) not_in_obs: []. not_in_exp: ['s_copy']."


def test_dtypes(base_df, character_set):
    df_obs = base_df.copy()
    df_obs['s'] = df_obs['s'].astype(int)  # Switch type

    diffs = df_compare(df_obs=df_obs, df_exp=base_df)
    assert all([key not in diffs for key in character_set - {'dtypes'}])
    assert diffs.get('dtypes') is not None


def test_index(base_df, character_set):
    df_exp = base_df.copy()
    df_exp = df_exp.set_index('s')

    df_obs = base_df.copy()
    df_obs['s'].iloc[1] = '3'
    df_obs = df_obs.set_index('s')

    diffs = df_compare(df_obs=df_obs, df_exp=df_exp)
    assert all([key not in diffs for key in character_set - {'index'}])
    assert diffs.get('index') is not None
    assert '3' in diffs['index']


def test_ints(base_df, character_set):
    """Test comparison of integers."""
    df_obs = base_df.copy()
    df_obs['i'].iloc[:2] = [3, 2]  # Change a couple values
    diffs = df_compare(df_obs=df_obs, df_exp=base_df)
    assert all([key not in diffs for key in character_set - {'int'}])
    assert diffs.get('int') is not None
    assert isinstance(diffs['int'], str)


def test_bools(base_df, character_set):
    df_obs = base_df.copy()
    df_obs['b'].iloc[:2] = [True, True]  # Change a couple values
    diffs = df_compare(df_obs=df_obs, df_exp=base_df)
    assert all([key not in diffs for key in character_set - {'bool'}])
    assert diffs.get('bool') is not None
    assert isinstance(diffs['bool'], str)


def test_floats(base_df, character_set):
    df_obs = base_df.copy()
    df_obs['f'].iloc[2] = 2.5
    diffs = df_compare(df_obs=df_obs, df_exp=base_df)
    assert all([key not in diffs for key in character_set - {'float'}])
    assert 'floats are different:' in diffs.get('float')


def test_strings(base_df, character_set):
    """TODO This naive test does not test non-string objects, nor does it handle StringDtype of pandas 1.0"""
    df_obs = base_df.copy()
    df_obs['s'].iloc[:2] = ['foo', 'bar']  # Change a couple values
    diffs = df_compare(df_obs=df_obs, df_exp=base_df)
    assert all([key not in diffs for key in character_set - {'object'}])
    assert diffs.get('object') is not None
    assert isinstance(diffs['object'], str)


def test_dates(base_df, character_set):
    """Test calendar dates. Time and zone are tossed away"""
    df_obs = base_df.copy()
    df_obs['d'] = pd.Series(['2017-12-31', '2019-01-01', '2020-01-01'], dtype='datetime64[ns]')
    diffs = df_compare(df_obs=df_obs, df_exp=base_df)
    assert all([key not in diffs for key in character_set - {'datetime'}])
    assert 'datetime' in diffs

    df_obs.loc[0, 'd'] = pd.NaT
    df_exp = base_df.copy()
    df_exp.loc[0, 'd'] = pd.NaT

    diffs = df_compare(df_obs=df_obs, df_exp=df_exp)
    assert 'datetime' not in diffs


def test_nan(base_df, character_set):
    # No difference when values are different, but nan is same place
    df_obs = base_df.copy()
    df_obs['f'] = [10, np.nan, 10.0]
    diffs = df_compare(df_obs=df_obs, df_exp=base_df)
    assert diffs['complete'] is True
    assert all([key not in diffs for key in character_set - {'float'}])
    assert 'nan' not in diffs

    # Differences, plus NaT, and optionally Nullable Integer
    df_exp = base_df.copy()
    if pd.__version__ >= '1.0':
        df_exp['Int64'] = pd.array([0, 1, None])
        df_obs['Int64'] = pd.array([0, 1, 2], dtype=pd.Int64Dtype())
    df_obs.loc[0, 'd'] = pd.NaT

    diffs = df_compare(df_obs=df_obs, df_exp=df_exp)
    assert all([key not in diffs for key in character_set - {'nan', 'float', 'datetime'}])
    assert 'float' in diffs
    assert 'nan' in diffs
    assert 'datetime' in diffs
    assert 'nans are different:' in diffs.get('nan')
