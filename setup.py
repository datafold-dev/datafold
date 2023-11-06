#!/usr/bin/env python

import importlib.util
from pathlib import Path

from setuptools import find_packages, setup

# see documentation
# https://packaging.python.org/guides/distributing-packages-using-setuptools/

author = "datafold development team"
email = "d.lehmberg@tum.de"

setuppy_dir = Path(__file__).absolute().parent
docs_dir = setuppy_dir / "doc"
docs_conf_dir = docs_dir / "source"

assert setuppy_dir.is_dir() and docs_dir.is_dir() and docs_conf_dir.is_dir()

requirements_filepath = (setuppy_dir / "requirements.txt").resolve()


def read_datafold_version():
    """This reads the current version from datafold/version.py without importing parts of
    datafold (which would require some of the dependencies already installed).
    """
    # code parts taken from https://stackoverflow.com/a/67692

    version_file = Path.joinpath(setuppy_dir, "datafold", "_version.py")
    spec = importlib.util.spec_from_file_location("version", version_file)
    version = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(version)
    return version.Version.v_short


def get_install_requirements():
    with requirements_filepath.open(mode="r") as f:
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
        raise RuntimeError("The short description must not contain newline '\\n'.")

    return short_description


def get_long_description():
    # use README.rst for text in PyPI:
    with (setuppy_dir / "README.rst").open(mode="r") as readme_file:
        long_description = readme_file.read()
    return long_description


project_urls = {
    "Bug Tracker": "https://gitlab.com/datafold-dev/datafold/-/issues",
    "Documentation": "https://datafold-dev.gitlab.io/datafold/",
    "Source Code": "https://gitlab.com/datafold-dev/datafold",
}

setup(
    name="datafold",
    author=author,
    version=read_datafold_version(),
    description=get_short_description(),
    long_description_content_type="text/x-rst",
    long_description=get_long_description(),
    license="MIT",
    url="https://datafold-dev.gitlab.io/datafold",
    project_urls=project_urls,
    download_url="https://pypi.org/project/datafold/",
    keywords=[
        "mathematics, machine learning, dynamical system, data-driven, time series, "
        "regression, forecasting, manifold learning, diffusion map, koopman operator, "
        "nonlinear"
    ],
    author_email=email,
    packages=find_packages(),
    package_dir={"datafold": "datafold"},
    python_requires=">=3.9",
    install_requires=get_install_requirements(),
    # classifiers from list: https://pypi.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3 :: Only",
        "Topic :: Scientific/Engineering",
        "Topic :: Software Development",
    ],
    # see https://stackoverflow.com/a/14159430
    # Both a MANIFEST.in and package_data is required such that an "bdist" or "sdist"
    # installation includes the additional files.
    # The "requirements.txt" is required for setup.py and must also be copied to
    # source distributions (setup.py install sdist)
    package_data={
        ".": ["requirements.txt", "LICENSE", "LICENSES_bundled", "CONTRIBUTORS"]
    },
)
