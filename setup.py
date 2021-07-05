#!/usr/bin/env python

import importlib.util
import io
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

setuppy_filepath = Path(__file__).absolute().parent

path_to_pkg_requirements = os.path.join(setuppy_filepath, "requirements.txt")
path_to_pkg_requirements = os.path.abspath(path_to_pkg_requirements)


def get_install_requirements():
    with open(path_to_pkg_requirements, "r", newline="\n") as f:
        install_requires = [req.replace("\n", "") for req in f.readlines()]
    return install_requires


def get_short_description():
    short_description = (
        "Operator-theoretic models to identify dynamical systems and parametrize point "
        "cloud geometry"
    )

    if "\n" in short_description:
        # check and raise this because "twine check dist/*" gives unsuitable error message
        # if this is the case
        raise RuntimeError("The short description must not contain newline '\\n'")


def get_long_description():
    # use README.rst for text in PyPI:
    with open(
        os.path.join(setuppy_filepath, "README.rst"), "r", newline="\n"
    ) as readme_file:
        long_description = readme_file.read()
    return long_description


setup(
    name="datafold",
    author=author,
    version=read_datafold_version(),
    description=get_short_description(),
    long_description_content_type="text/x-rst",
    long_description=get_long_description(),
    license="MIT",
    url="https://datafold-dev.gitlab.io/datafold",
    download_url="https://pypi.org/project/datafold/",
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
    install_requires=get_install_requirements(),
    # taken from list: https://pypi.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3 :: Only",
        "Topic :: Scientific/Engineering",
    ],
    # see https://stackoverflow.com/a/14159430
    # Both a MANIFEST.in and package_data is required that bdist and sdist
    # installations include the files.
    # The requirements.txt is required for setup.py and must also be copied to
    # source distributions (setup.py install sdist)
    package_data={
        ".": ["requirements.txt", "LICENSE", "LICENSES_bundled", "CONTRIBUTORS"]
    },
)
