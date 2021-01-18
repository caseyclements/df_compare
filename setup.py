#!/usr/bin/env python

from setuptools import setup, find_packages
from os.path import join, dirname, abspath
import sys
import os
import datetime
from importlib import import_module
import subprocess

if sys.version_info[:2] < (3, 6):
    raise RuntimeError("Requires Python version >= 3.6.")

PKG_NAME = 'df_compare'
BUILD_META = 'build-meta'
VERSION_PY_FILE = os.path.join(PKG_NAME, 'version.py')
MAJOR = 0
MINOR = 0
MICRO = 1
BETA  = 0
PKG_VERSION = '{}.{}.{}'.format(MAJOR, MINOR, MICRO)
FULL_VERSION = PKG_VERSION
if BETA:
    FULL_VERSION = PKG_VERSION + 'b{}'.format(BETA)


CLASSIFIERS = """\
Development Status :: 3 - Alpha
Programming Language :: Python
Intended Audience :: Developers
Topic :: Scientific/Engineering
Operating System :: OS Independent
"""


def package_data(directory):
    paths = []
    for (path, directories, filenames) in os.walk(directory):
        for filename in filenames:
            if filename.rsplit('.',maxsplit=1)[-1] in ['csv', 'yml', 'yaml', 'txt']:
                paths.append(join('..', path, filename))
    return paths


def get_git_revision():
    """Determine git revision for this build

    :return: git revision (SHA-1 for commit)
    """
    git_revision = 'Unknown'
    if os.path.exists('.git'):
        build_host_git = ''
        try:
            # Git is not in a standard location in our Build
            # environment, so we have to try finding it via token.
            util = import_module('blkcore.util')
            build_host_git = util.get_token('SRPT.GIT_BIN')
            # get_token will return None rather than raise error on failure.
            if build_host_git is None:
                build_host_git = ''
        except ImportError:
            pass
        if os.access(build_host_git, os.X_OK):
            git = build_host_git
        else:
            git = 'git'
        cmd = [git, 'rev-parse', 'HEAD']
        try:
            git_revision = subprocess.check_output(cmd, universal_newlines=True).rstrip('\n')
        except Exception as e:
            # A number of things can go wrong trying to obtain the
            # git revision, so we need to be lenient.
            print('Could not determine git revision: {} - {!s}'.
                  format(e.__class__.__name__, e), file=sys.stderr, flush=True
                  )

    return git_revision


def write_version_py():
    """Write build version so it can be accessed at runtime.
    """
    content = """\"\"\"
THIS FILE IS GENERATED AT BUILD TIME

(c)  {} BlackRock.  All rights reserved.
\"\"\"

version = '{}'
git_revision = '{}'
"""
    now = datetime.datetime.now()
    with open(VERSION_PY_FILE, 'w') as f:
        f.write(content.format(now.year, FULL_VERSION, get_git_revision()))


pkgs = find_packages()
pkg_data = {'': package_data(join(dirname(abspath(__file__)), PKG_NAME))}
write_version_py()

setup(
    name=PKG_NAME,
    version=FULL_VERSION,
    packages=pkgs,
    package_data=pkg_data,
    include_package_data=True,
    description='APF Engine: Distributed Python Workflow configuration and execution platform',
    long_description=open('README.md').read(),
    install_requires=open('requirements.txt').read().splitlines(),
    entry_points={'console_scripts': ['df_compare = df_compare.compare_files:main']},
    maintainer='AGT',
    maintainer_email='casey.clements@gmail.com',
    license="BSD",
    classifiers=[_f for _f in CLASSIFIERS.split('\n') if _f],
    platforms=["MacOS", "Linux", "Windows"],
)
