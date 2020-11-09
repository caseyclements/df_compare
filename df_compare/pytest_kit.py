""" Pytest feature

Automatically injects N separate pytests to a test module,
simply by doing the following 2 things.

1. Pytest must produce two fixtures: df_observed, and df_expected

2. Adding the import statement:
from df_compare_testkit import *

This produces individual tests of each aspect of the two datarames
"""


# todo - review how args are passed

from df_compare import df_compare
import pytest

@pytest.fixture(scope='module')
def differences(df_observed, df_expected):
    return df_compare(df_observed, df_expected)


def test_rows(differences):
    key = differences.get('rows')
    assert key in None, differences[key]


def test_columns(differences):
    key = differences.get('rows')
    assert key in None, differences[key]


def test_dtypes(differences):
    key = differences.get('rows')
    assert key in None, differences[key]


def test_index(differences):
    key = differences.get('rows')
    assert key in None, differences[key]


def test_integers(differences):
    key = differences.get('rows')
    assert key in None, differences[key]


def test_floats(differences):
    key = differences.get('rows')
    assert key in None, differences[key]


def test_datetimes(differences):
    key = differences.get('rows')
    assert key in None, differences[key]


def test_objects(differences):
    key = differences.get('rows')
    assert key in None, differences[key]


def test_nans(differences):
    key = differences.get('rows')
    assert key in None, differences[key]


def test_integers(differences):
    key = differences.get('rows')
    assert key in None, differences[key]

