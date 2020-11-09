import pandas
import argparse
import sys
import os


def main(options=None):
    """ Shell script to compare two files, or even two directories of files, containing tables

    Takes two paths (file or dir) and variables. Logs descriptive statistics to console.
    Directory would include file names that don’t match. Arguments include, filetypes
    """
    # 1. Parse arguments
    options = options or parse_args()  # nb. argparse.parse_args will default to sys.argv[1:] if None is provided

    # TODO IMPLEMENT
    #  - Addl arguments such as: filetypes, tolerances, ignore_index, sort_by, ...
    #  - 1) Logic to handle single files.
    #  - 2) Logic to handle directories.
    #       - look for files of the same name
    #       - hdf5's look for same table names
    #       - spreadsheets (one day)


def parse_args(args=None):
    """ Defines and parses arguments for a run. """

    class StoreDict(argparse.Action):
        """argparse action type that allows one to build a dictionary.

            Each instance of the argument adds a key:value pair to the dictionary.
            The format must be key:value ==> No space between colon and value!
            For example:
            parser.add_argument('-e', '--env', action=StoreDict)
            parser.parse_args('-e my_int:1 -e my_str:"1" my_bool:True'.split())
        """
        def __init__(self, option_strings, dest, default=dict(),
                     required=False, help=None, metavar=None):
            super().__init__(option_strings=option_strings, dest=dest, default=default,
                             required=required, help=help, metavar=metavar)

        def __call__(self, parser, namespace, values, option_string=None):
            dico = getattr(namespace, self.dest)
            dico.update(yaml.load(values.replace(':', ': ', 1)))
            setattr(namespace, self.dest, dico)

    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument('-o', '--obs', type=str, required=True,
                        help='path to file or directory containing so-called observed values')
    parser.add_argument('-e', '--exp', type=str, required=True,
                        help='path to file or directory containing so-called expected values')
    return vars(parser.parse_args(args))


if __name__ == '__main__':
    sys.exit(main())
