#!/usr/bin/env python

from setuptools import find_packages, setup

import datafold

# see documentation
# https://packaging.python.org/guides/distributing-packages-using-setuptools/

AUTHOR = "datafold development team"
EMAIL = None


setup(
    name="datafold",
    author=AUTHOR,
    version=datafold.__version__,
    description="Python package to deal with dynamical systems represented by manifold "
    "data.",
    long_description="TODO: provide long description",
    license="MIT",
    author_email=EMAIL,
    packages=find_packages(),
    package_dir={"datafold": "datafold"},
    package_data={"": ["LICENSE"]},
    python_requires=">=3.6",  # uses f-strings
    install_requires=["numpy", "scikit-learn", "scipy", "pandas"],
    test_suite="nose.collector",
    tests_require=["nose"],
    extras_require={"pydmd": ["pydmd"], "plot": ["matplotlib"],},
    # taken from list: https://pypi.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3 :: Only",
        "Topic :: Scientific/Engineering",
    ],
)
