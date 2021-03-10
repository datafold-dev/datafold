#!/usr/bin/env python

import importlib.util
import os
from pathlib import Path

from setuptools import find_packages, setup


def read_datafold_version():
    """This reads the version from datafold/version.py without importing parts of
    datafold (which would require some of the dependencies already installed)."""
    # code parts taken from here https://stackoverflow.com/a/67692

    path2setup = os.path.dirname(__file__)
    version_file = os.path.join(path2setup, "datafold", "_version.py")
    version_file = os.path.abspath(version_file)

    spec = importlib.util.spec_from_file_location("version", version_file)
    version = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(version)
    return version.Version.v_short


# see documentation
# https://packaging.python.org/guides/distributing-packages-using-setuptools/

author = "datafold development team"
# TODO: maybe there is a more general email to use?
email = "daniel.lehmberg@hm.edu"

path_to_pkg_requirements = os.path.join(
    Path(__file__).absolute().parent, "requirements.txt"
)
path_to_pkg_requirements = os.path.abspath(path_to_pkg_requirements)

with open(path_to_pkg_requirements, "r") as f:
    install_requires = f.readlines()

install_requires = [req.replace("\n", "") for req in install_requires]

short_description = (
    "A package providing manifold parametrization in the Diffusion Maps framework "
    "and identification of dynamical systems in the Koopman operator view with the "
    "Extended Dynamic Mode Decomposition."
)

long_description = """
The package provides:

* (Extended-) Dynamic Mode Decomposition (EDMD) to approximate the Koopman operator for 
  system identification. 
* Diffusion Maps to find meaningful geometric descriptions in point clouds, such as the 
  eigenfunctions of the Laplace-Beltrami operator. 
* Data structure for time series collections (TSCDataFrame) and dedicated 
  transformations, such as time-delay embeddings (TSCTakensEmbedding). The data 
  structures operate with both EDMD and DMAP.  
"""

setup(
    name="datafold",
    author=author,
    version=read_datafold_version(),
    description=short_description,
    long_description=long_description,
    license="MIT",
    url="https://datafold-dev.gitlab.io/datafold",
    keywords=[
        "machine learning, dynamical system, data-driven, time series, time series "
        "regression, time series forecasting, manifold learning, koopman operator"
    ],
    author_email=email,
    packages=find_packages(),
    package_dir={"datafold": "datafold"},
    package_data={"": ["LICENSE"]},
    python_requires=">=3.7",
    install_requires=install_requires,
    test_suite="nose.collector",
    tests_require=["nose>=1.3.7,<1.4"],
    # taken from list: https://pypi.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3 :: Only",
        "Topic :: Scientific/Engineering",
    ],
)
