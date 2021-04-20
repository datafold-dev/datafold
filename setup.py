#!/usr/bin/env python

import importlib.util
import os
from pathlib import Path

from setuptools import find_packages, setup


def read_datafold_version():
    """This reads the version from datafold/version.py without importing parts of
    datafold (which would require some of the dependencies already installed)."""
    # code parts taken from https://stackoverflow.com/a/67692

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
email = "daniel.lehmberg@hm.edu"

path_to_pkg_requirements = os.path.join(
    Path(__file__).absolute().parent, "requirements.txt"
)
path_to_pkg_requirements = os.path.abspath(path_to_pkg_requirements)

with open(path_to_pkg_requirements, "r") as f:
    install_requires = f.readlines()

install_requires = [req.replace("\n", "") for req in install_requires]

short_description = """The package contains operator-theoretic models that can
identify dynamical systems from time series data and infer geometrical structures from
point clouds."""

long_description = """Main models in datafold:

* (Extended-) Dynamic Mode Decomposition (E-DMD) to approximate the Koopman
  operator from time series data or collections thereof.
* Diffusion Map (DMAP) to find meaningful geometric descriptions in point clouds,
  such as the eigenfunctions of the Laplace-Beltrami operator.
* Out-of-sample extensions to interpolate functions on point cloud manifolds, such as
  Geometric Harmonics interpolator and (auto-tuned) Laplacian Pyramids.
* Data structure for time series collections (TSCDataFrame) and data
  transformations, such as time-delay embeddings (TSCTakensEmbedding). The data
  structures operates with both E-DMD and DMAP (internally or as input).

"""

setup(
    name="datafold",
    author=author,
    version=read_datafold_version(),
    description=short_description,
    long_description_content_type="text/x-rst",
    long_description=long_description,
    license="MIT",
    url="https://datafold-dev.gitlab.io/datafold",
    keywords=[
        "mathematics, machine learning, dynamical system, data-driven, time series, "
        "regression, forecasting, manifold learning, diffusion map, koopman operator, "
        "nonlinear"
    ],
    author_email=email,
    packages=find_packages(),
    package_dir={"datafold": "datafold"},
    # package_data={"": ["LICENSE"]},
    python_requires=">=3.7",
    install_requires=install_requires,
    # taken from list: https://pypi.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3 :: Only",
        "Topic :: Scientific/Engineering",
    ],
    # the requirements.txt is needed during setup.py and must also be copied to
    # source distributions (setup.py install sdist)
    package_data={".": ["requirements.txt", "LICENSE", "LICENSES_bundled"]},
)
