#!/usr/bin/env python

import importlib.util
import os
from pathlib import Path

from setuptools import find_packages, setup


def read_datafold_version():
    """This reads the version from datafold/version.py without importing parts of
    datafold (which would require some of the dependencies already installed)."""
    # code parts were taken from here https://stackoverflow.com/a/67692

    path2setup = os.path.dirname(__file__)
    version_file = os.path.abspath(os.path.join(path2setup, "datafold", "version.py"))

    spec = importlib.util.spec_from_file_location("version", version_file)
    version = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(version)
    return version.Version.v_short


# see documentation
# https://packaging.python.org/guides/distributing-packages-using-setuptools/

AUTHOR = "datafold development team"
# TODO: in future, if appropriate, can set up a mailing list
EMAIL = "daniel.lehmberg@hm.edu"

long_description = (
    "datafold is a Python package that provides data-driven models for point clouds to "
    "find an explicit manifold parametrization and to identify non-linear dynamical "
    "systems on these manifolds."
)


path_to_pkg_requirements = os.path.join(
    Path(__file__).absolute().parent, "requirements.txt"
)

with open(path_to_pkg_requirements, "r") as f:
    install_requires = f.readlines()
install_requires = [req.replace("\n", "") for req in install_requires]

setup(
    name="datafold",
    author=AUTHOR,
    version=read_datafold_version(),
    description="datafold is Python software for data-driven algorithms with "
    "manifold assumption",
    long_description=long_description,
    license="MIT",
    url="https://datafold-dev.gitlab.io/datafold",
    keywords=[
        "machine learning, dynamical system, data-driven, time series, time series "
        "regression, time series forecasting, manifold learning"
    ],
    author_email=EMAIL,
    packages=find_packages(),
    package_dir={"datafold": "datafold"},
    package_data={"": ["LICENSE"]},
    python_requires=">=3.6",  # uses f-strings
    install_requires=install_requires,
    test_suite="nose.collector",
    tests_require=["nose>=1.3.7,<1.4"],
    extras_require={"pydmd": ["pydmd==0.2.1"]},
    # taken from list: https://pypi.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3 :: Only",
        "Topic :: Scientific/Engineering",
    ],
)
