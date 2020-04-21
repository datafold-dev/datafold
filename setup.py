#!/usr/bin/env python

from setuptools import find_packages, setup

import datafold

# see documentation
# https://packaging.python.org/guides/distributing-packages-using-setuptools/

AUTHOR = "datafold development team"
# TODO: in future, if appropriate, can set up a mailing list
EMAIL = "daniel.lehmberg@hm.edu"

setup(
    name="datafold",
    author=AUTHOR,
    version=datafold.__version__,
    description="datafold processes high-dimensional data (point clouds or time "
    "series) to learn hidden geometric structures.",
    long_description="""
datafold is a Python package consisting of *data*-driven algorithms with 
mani*fold* assumption. That is to process high-dimensional data (including time series) 
that lie on an (unknown) manifold with intrinsic lower-dimension.""",
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
    install_requires=["numpy", "scikit-learn", "scipy", "pandas", "matplotlib"],
    test_suite="nose.collector",
    tests_require=["nose"],
    extras_require={"pydmd": ["pydmd"],},
    # taken from list: https://pypi.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3 :: Only",
        "Topic :: Scientific/Engineering",
    ],
)
