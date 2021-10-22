#!/usr/bin/env python

import distutils.cmd
import importlib.util
import os
import shutil
from pathlib import Path

from setuptools import find_packages, setup

# see documentation
# https://packaging.python.org/guides/distributing-packages-using-setuptools/

author = "datafold development team"
email = "daniel.lehmberg@hm.edu"

setuppy_dir = Path(__file__).absolute().parent
docs_dir = os.path.join(setuppy_dir, "doc")
docs_conf_dir = os.path.join(docs_dir, "source")

requirements_filepath = os.path.abspath(os.path.join(setuppy_dir, "requirements.txt"))


class BuildDocs(distutils.cmd.Command):
    description = "Build html documentation with Sphinx"

    user_options = [
        ("outdir=", None, "Directory path to write the html docs to."),
        (
            "runtutorials",
            None,
            "Flag to execute all cells in the tutorial Jupyter notebooks.",
        ),
    ]

    def initialize_options(self):
        self.outdir = os.path.join(docs_dir, "build")
        self.runtutorials = False

    def finalize_options(self):
        _parent_path = Path(self.outdir).parent

        if os.path.isfile(self.outdir):
            raise NotADirectoryError(f"Path {self.outdir} is a path to a file.")

        if not os.path.exists(_parent_path):
            raise NotADirectoryError(f"Path {_parent_path} is not a directory.")

        if os.path.exists(self.outdir):
            # remove existing directory
            shutil.rmtree(self.outdir)

        if not isinstance(self.runtutorials, bool) or self.runtutorials:
            self.runtutorials = True
        else:
            self.runtutorials = False

    def run(self):
        # only import here, if this fails requirements-dev are not installed
        from sphinx.application import Sphinx

        if self.runtutorials:
            os.environ["DATAFOLD_NBSPHINX_EXECUTE"] = "always"

        docs_build_dir = os.path.join(self.outdir, "doctrees")

        sph = Sphinx(
            srcdir=docs_conf_dir,
            confdir=docs_conf_dir,
            outdir=self.outdir,
            doctreedir=docs_build_dir,
            buildername="html",
        )
        self.announce("Running command:")
        sph.build(force_all=True)


def read_datafold_version():
    """This reads the current version from datafold/version.py without importing parts of
    datafold (which would require some of the dependencies already installed)."""
    # code parts taken from https://stackoverflow.com/a/67692

    version_file = Path.joinpath(setuppy_dir, "datafold", "_version.py")
    spec = importlib.util.spec_from_file_location("version", version_file)
    version = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(version)
    return version.Version.v_short


def get_install_requirements():
    with open(requirements_filepath, "r", newline="\n") as f:
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
    with open(
        os.path.join(setuppy_dir, "README.rst"), "r", newline="\n"
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
    python_requires=">=3.7",
    install_requires=get_install_requirements(),
    # classifiers from list: https://pypi.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3 :: Only",
        "Topic :: Scientific/Engineering",
    ],
    cmdclass={"build_docs": BuildDocs},
    # see https://stackoverflow.com/a/14159430
    # Both a MANIFEST.in and package_data is required such that an "bdist" or "sdist"
    # installation includes the additional files.
    # The "requirements.txt" is required for setup.py and must also be copied to
    # source distributions (setup.py install sdist)
    package_data={
        ".": ["requirements.txt", "LICENSE", "LICENSES_bundled", "CONTRIBUTORS"]
    },
)
