#!/usr/bin/env python

from setuptools import find_packages, setup

import datafold


# see documentation
# https://packaging.python.org/guides/distributing-packages-using-setuptools/

AUTHOR = "The datafold development team"
EMAIL = None  # TODO: maybe provide here once a mailing-list email (send email to all group members)


setup(name='datafold',
      maintainer=AUTHOR,
      version=datafold.__version__,
      description="Python package to deal with dynamical systems represented by manifold data.",
      long_description="TODO: provide long description",
      license='MIT License',
      author=AUTHOR,
      author_email=EMAIL,
      packages=find_packages(),
      package_dir={'datafold': 'datafold'},
      package_data={'': ['LICENSE']},
      python_requires='>=3.6',  # uses f-strings
      test_suite='nose.collector',
      tests_require=['nose'],
      install_requires=['numpy', 'scikit-learn', 'scipy'],
      extras_require={
          'plotting': ['matplotlib', 'ipywidgets'],
          'cuda': ['numba'],
          'documenation': ['Sphinx']},
      # taken from list https://pypi.org/pypi?%3Aaction=list_classifiers
      classifiers=["Intended Audience :: Science/Research",
                   "License :: OSI Approved :: MIT License",
                   "Operating System :: OS Independent",
                   "Programming Language :: C",
                   "Programming Language :: C++",
                   "Programming Language :: Python :: 3.6",
                   "Programming Language :: Python :: 3 :: Only",
                   "Topic :: Scientific/Engineering",
                   "Topic :: Software Development :: Version Control"]
      )
