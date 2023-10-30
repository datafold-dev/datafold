Quick links
^^^^^^^^^^^

`Source repository <https://gitlab.com/datafold-dev/datafold>`__ |
`Contributing and feedback <https://datafold-dev.gitlab.io/datafold/contributing.html>`__ |
`PyPI <https://pypi.org/project/datafold/>`__ |
`Documentation <https://datafold-dev.gitlab.io/datafold/>`__ |
`Tutorials <https://datafold-dev.gitlab.io/datafold/tutorial_index.html>`__ |
`Scientific literature <https://datafold-dev.gitlab.io/datafold/references.html>`__

What is *datafold*?
====================

*datafold* is a `MIT-licensed <https://gitlab.com/datafold-dev/datafold/-/blob/master/LICENSE>`__
Python package containing operator-theoretic, data-driven models to identify dynamical
systems from time series data and to infer geometrical structures in point clouds.

The package includes:

* Implementations and variants of the Dynamic Mode Decomposition as data-driven methods to
  identify and analyze dynamical systems from time series collection data. This incldues:

  * ``DMDFull`` or ``DMDEco`` as standard methods of DMD
  * ``OnlineDMD`` or ``StreamingDMD`` modify the DMD to handle streaming data
  * ``DMDControl`` augments the DMD to handle additional control input
  * ``EDMD`` - The Extended-DMD, which allows setting up a highly flexible dictionary to
    decompose and embed time series data and thereby handle nonlinear dynamics within the
    Koopman operator framework. ``EDMD`` wraps an arbitrary DMD variation for the decomposition.
    The key advantage of this is, that the ``EDMD`` directly profits from the above
    functionalities. ``EDMD`` can be used in control or streaming settings. Furthermore, the
    dictionary can also be learnt from the data, corresponding to the EDMD-DL.
* An efficient implementation of the ``DiffusionMaps`` model to infer geometric
  meaningful structures from (time series) data, such as the eigenfunctions of the
  Laplace-Beltrami operator. As a distinguishing factor to other implementations, the
  model can handle a sparse kernel matrix and allows setting an arbitrary kernel,
  including the standard Gaussian kernel,
  `continuous k-nearest neighbor kernel <https://arxiv.org/abs/1606.02353>`__, or
  `dynamics-adapted cone kernel <https://cims.nyu.edu/~dimitris/files/Giannakis15_cone_kernels.pdf>`__.
* Cross-validation. The method ``EDMDCV`` allows model parameters to be optimized with
  cross-validation splittings that account for the temporal order in time series data.
* Methods to perform Model Predictive Control (MPC) with Koopman operator-based methods (
  mainly the ``EDMD``).
* Regression models for high-dimensional data, which are commonly used for out-of-sample
  extensions for the Diffusion Maps model. This includes the (auto-tuned) Laplacian Pyramids
  or Geometric Harmonics to interpolate general function values on a point cloud manifold.
* A data structure ``TSCDataFrame`` to handle time series collection (TSC) data. It simplifies
  model inputs/output and make it easier to describe various forms of time series data.

See also `this introduction page <https://datafold-dev.gitlab.io/datafold/intro.html>`__.
For a mathematical thorough introduction, we refer to the `scientific literature
<https://datafold-dev.gitlab.io/datafold/references.html>`__.

.. note::
    The project is under active development in a research-driven environment.

    * Code quality varies from "experimental/early stage" to "well-tested". Well tested
      code is listed in the
      `software documentation <https://datafold-dev.gitlab.io/datafold/api.html>`__
      and are directly accessible through the highest module level (e.g.
      :code:`from datafold import ...`). Experimental code is
      only accessible via "deep imports" (e.g.
      :code:`from datafol.dynfold.outofsample import ...`) and may raise a warning when using
      it.
    * The interfaces within *datafold* are not stable. The software is **not** intended for
      production. Nevertheless, if we break something it is intentional and we hope that such
      adaptations become less over time.
    * There is no deprecation cycle. The software uses
      `semantic versioning <https://semver.org/>`__ policy `[major].[minor].[patch]`, i.e.

         * `major` - making incompatible changes in the (documented) API
         * `minor` - adding functionality in a backwards-compatible manner
         * `patch` - backwards-compatible bug fixes

      We do not intend to indicate a feature complete milestone with version `1.0`.

Cite
====

If you use *datafold* in your research, please cite
`this paper <https://joss.theoj.org/papers/10.21105/joss.02283>`__ published in the
*Journal of Open Source Software* (`JOSS <https://joss.theoj.org/>`__).

*Lehmberg et al., (2020). datafold: data-driven models for point clouds and time series on
manifolds. Journal of Open Source Software, 5(51), 2283,* https://doi.org/10.21105/joss.02283

BibTeX:

.. code-block:: latex

    @article{Lehmberg2020,
             doi       = {10.21105/joss.02283},
             url       = {https://doi.org/10.21105/joss.02283},
             year      = {2020},
             publisher = {The Open Journal},
             volume    = {5},
             number    = {51},
             pages     = {2283},
             author    = {Daniel Lehmberg and Felix Dietrich and Gerta K{\"o}ster and Hans-Joachim Bungartz},
             title     = {datafold: data-driven models for point clouds and time series on manifolds},
             journal   = {Journal of Open Source Software}}

How to get it?
==============

Installation requires `Python>=3.9 <https://www.python.org/>`__ with
`pip <https://pip.pypa.io/en/stable/>`__ and
`setuptools <https://setuptools.pypa.io/en/latest/>`__ installed (both packages ship with a
standard Python installation). The package dependencies
install automatically. The main dependencies and their usage in *datafold* are listed
in the section "Dependencies" below.

There are two ways to install *datafold*:

1. From PyPI
------------

This is the standard way for users. The package is hosted on the official Python package
index (PyPI) and installs the core package (excluding tutorials and tests). The tutorial
files can be downloaded separately
`here <https://datafold-dev.gitlab.io/datafold/tutorial_index.html>`__.

To install the package and its dependencies with :code:`pip`, run

.. code-block:: bash

   python -m pip install datafold

.. note::

    If you run Python in an Anaconda environment you can use pip from within ``conda``.
    See also
    `official instructions <https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-pkgs.html#installing-non-conda-packages>`__.

    .. code-block:: bash

        conda activate venv
        conda install pip
        pip install datafold

2. From source
--------------

This way is recommended if you want to access the latest (but potentially unstable)
development state, run tests or wish to contribute (see section "Contributing" for details).
Download or git-clone the source code repository.

1. Download the repository

   a. If you wish to contribute code, it is required to have
      `git <https://git-scm.com/>`__ installed. Clone the repository with

      .. code-block:: bash

        git clone https://gitlab.com/datafold-dev/datafold.git

   b. If you only want access to the source code (current ``master`` branch), download one
      of the compressed file types
      (`zip <https://gitlab.com/datafold-dev/datafold/-/archive/master/datafold-master.zip>`__,
      `tar.gz <https://gitlab.com/datafold-dev/datafold/-/archive/master/datafold-master.tar.gz>`__,
      `tar.bz2 <https://gitlab.com/datafold-dev/datafold/-/archive/master/datafold-master.tar.bz2>`__,
      `tar <https://gitlab.com/datafold-dev/datafold/-/archive/master/datafold-master.tar>`__)

2. Install the package from the downloaded repository

   .. code-block:: bash

       python -m pip install .

Contributing
============

Any contribution (code/tutorials/documentation improvements), question or feedback is
very welcome. Either use the
`issue tracker <https://gitlab.com/datafold-dev/datafold/-/issues>`__ or
`Email <incoming+datafold-dev-datafold-14878376-issue-@incoming.gitlab.com>`__ us.
Instructions to set up *datafold* for development can be found
`here <https://datafold-dev.gitlab.io/datafold/contributing.html>`__.

Dependencies
============

The dependencies of the core package are managed in the file
`requirements.txt <https://gitlab.com/datafold-dev/datafold/-/blob/master/requirements.txt>`__
and install with *datafold*. The tests, tutorials, documentation and code analysis
require additional dependencies which are managed in
`requirements-dev.txt <https://gitlab.com/datafold-dev/datafold/-/blob/master/requirements-dev.txt>`__.

*datafold* integrates with common packages from the
`Python scientific computing stack <https://scipy.org/about/>`__:

* `NumPy <https://numpy.org/>`__
   NumPy is used throughout *datafold* and is the default package for numerical
   data and algorithms.

* `pandas <https://pandas.pydata.org/pandas-docs/stable/index.html>`__
   *datafold* uses pandas'
   `DataFrame <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html>`__
   as a base class for ``TSCDataFrame`` to capture various forms of time series data. The data
   It includes specific time series collection functionality and is mostly compatible with
   pandas' rich functionality.

* `scikit-learn <https://scikit-learn.org/stable/>`__
   All *datafold* algorithms that are part of the "machine learning pipeline" align
   to the scikit-learn `API <https://scikit-learn.org/stable/developers/develop.html>`__.
   This is done by deriving the models from
   `BaseEstimator <https://scikit-learn.org/stable/modules/generated/sklearn.base.BaseEstimator.html>`__.
   and appropriate ``MixIns``. *datafold* defines own ``MixIns`` that align with the
   API in a duck-typing fashion to allow identifying dynamical systems from temporal data
   in ``TSCDataFrame``.

* `SciPy <https://docs.scipy.org/doc/scipy/reference/index.html>`__
   The package is used for elementary numerical algorithms and data structures in
   conjunction with NumPy. This includes (sparse) linear least
   square regression, (sparse) eigenpairs solver and sparse matrices as
   optional data structure for kernel matrices.

How does it compare to other software?
======================================

*Note: This list covers only Python packages.*

* `scikit-learn <https://scikit-learn.org/stable/>`__
   provides algorithms and models along the entire machine learning pipeline, with a
   strong focus on static data (i.e. without temporal context). *datafold* integrates
   into scikit-learn' API and all data-driven models are subclasses of
   `BaseEstimator <https://scikit-learn.org/stable/modules/generated/sklearn.base.BaseEstimator.html>`__.
   An important contribution of *datafold* is the ``DiffusionMaps`` model as popular
   framework for manifold learning, which is not contained in scikit-learn's `set of
   algorithms <https://scikit-learn.org/stable/auto_examples/manifold/plot_compare_methods
   .html#sphx-glr-auto-examples-manifold-plot-compare-methods-py>`__.
   Furthermore, *datafold* includes dynamical systems as a new model class that is
   operable with scikit-learn - the attributes align to supervised learning tasks.
   The key differences are that a model processes data of type ``TSCDataFrame``
   and instead of a one-to-one relation in the model's input/output, the model can return
   arbitrary many output samples (a time series) for a single input
   (an initial condition).

* `PyDMD <https://github.com/PyDMD/PyDMD>`__
   provides many variants of the `Dynamic Mode Decomposition (DMD)
   <https://en.wikipedia.org/wiki/Dynamic_mode_decomposition>`__. *datafold* provides a wrapper
   to make models of ``PyDMD`` accessible. However, a limitation of ``PyDMD`` is that it only
   processes single coherent time series, see `PyDMD issue 86
   <https://github.com/PyDMD/PyDMD/issues/86>`__. The DMD models that are directly included
   in *datafold* utilize the functionality of the data structure ``TSCDataFrame`` and can
   therefore process time series collections - in an extreme case only containing snapshot
   pairs.

* `PySINDy <https://pysindy.readthedocs.io/en/latest/>`__
   specializes on a *sparse* system identification of nonlinear dynamical systems to
   infer governing equations.
