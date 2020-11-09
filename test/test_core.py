from df_compare import df_compare
import pandas as pd
import numpy as np
import pytest


@pytest.fixture(scope='session')
def basic_dict():
    d = {
        "i": [0, 1, 2],
        "f": [0.0, np.nan, 2.0],
        "d": ["2018-01-01", "2019-01-01", "2020-01-01"],
        "s": ["0", "1", "2"]
    }
    return d


def test_nrows(basic_dict):
    """ Test number of rows when same, different, and its description"""
    df_obs = pd.DataFrame(basic_dict)
    df_exp = pd.DataFrame(basic_dict)

    diffs = df_compare(df_obs=df_obs, df_exp=df_exp)

    assert diffs.get('rows') is None

    df_obs2 = pd.concat([df_exp, df_obs])

    diffs2 = df_compare(df_obs=df_obs2, df_exp=df_exp)
    desc = diffs2.get('rows')
    assert desc == f'df_obs has {len(df_obs2)}. df_exp has {len(df_exp)}'
