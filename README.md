# Informative Comparison of DataFrame Differences

This is a tool that is used to compare and test Pandas DataFrames. It has three parts.

### Core function
At it's core is a function that compares a number of aspects of two dataframes.

1. Number of rows
2. Column names
3. dtypes
4. Index
5. Integers
6. Floats
7. Datetimes
8. Objects (typically strings)
9. NaNs
10. Booleans

For each dimension, if differences are found, a key is included in the diffs dictionary that is returned.
The value of that key (e.g. 'nrows') would be a string description (e.g. 'observed contains 5 rows. expected 10.)

### PyTest Integration

If one is testing functionality that produces a dataframe, and one can create one to compare it to... 

`pytest_kit.py` automatically injects N separate pytests into a test module (test_\<name\>.py)
simply by doing the following 2 things.

1. Create two pytest fixtures: `df_observed`, and `df_expected`

2. Adding the import statement `from df_compare.pytest_kit import *`

That's it, when you run `$ pytest test_<name>.py`, it will discover and run N separate tests!


### Shell script

This comparison can be used to compare two files, or even a directory of files containing tables.

###### Details to be added




 
