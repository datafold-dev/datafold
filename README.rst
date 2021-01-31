What is *datafold*?
====================

*datafold* is a Python package providing **data**-driven models for point clouds that are
sampled on (or near) a mani**fold** (i.e. a geometrical structure with certain
properties). Associating data to manifolds, which are often of much lower dimension than
the ambient point dimension, is typically referred to as the "manifold assumption".
Successful models can extract the intrinsic coordinates of the manifold and generalize it
to the neighbourhood in the vicinity of the available training data. *datafold* includes
models with an explicit manifold parametrization. The models can therefore give insight
into the analyzed process and can uncover spatial or spatio-temporal patterns of the
process.

*datafold* includes

* Data structures to handle point clouds on manifolds (``PCManifold``) and time series
  collections (``TSCDataFrame``). The data structures are used both internally and for
  model input/outputs (if applicable).
* An efficient implementation of the ``DiffusionMaps`` model to parametrize
  a manifold from point cloud data by approximating the eigenfunctions of the
  Laplace-Beltrami operator. As a distinguishing factor to other implementations, the the
  model allows setting an arbitrary kernel, for example, a standard Gaussian kernel,
  `continuous `k` nearest neighbor kernel <https://arxiv.org/abs/1606.02353>`__, or
  `dynamics-adapted kernel (cone kernel) <https://cims.nyu.edu/~dimitris/files/Giannakis15_cone_kernels.pdf>`__.
* Out-of-sample methods such as the (auto-tuned) Laplacian Pyramids or Geometric
  Harmonics to interpolate general function values on manifold point clouds.
* (Extended-) Dynamic Mode Decomposition (e.g. model ``DMDFull`` or ``EDMD``) which
  are data-driven dynamical models built from time series data. To improve the
  model's accuracy, EDMD provides a framework to express the data in a more suitable
  intrinsic feature state in accordance to the Koopman operator theory. This includes
  scaling of heterogeneous time series quantities and more involved transformations to
  geometrically informed function basis, such as with ``DiffusionMaps``.
  Furthermore, it is possible to time-delay time series for phase space reconstruction
  (cf. `Takens theorem <https://en.wikipedia.org/wiki/Takens%27s_theorem>`__).
* ``EDMDCV`` allows model parameters (including the
  transformation model parameters) to be optimized with cross-validation and
  also accounts for time series splittings.

See also `this introduction page <https://datafold-dev.gitlab.io/datafold/intro.html>`__.
For a mathematical thorough introduction, we refer to the
`scientific literature <https://datafold-dev.gitlab.io/datafold/references.html>`__.

The source code is distributed under the
`MIT license <https://gitlab.com/datafold-dev/datafold/-/blob/master/LICENSE>`__.

Any contribution (code/tutorials/documentation improvements) and feedback is
very welcome. Either use the
`issue tracker <https://gitlab.com/datafold-dev/datafold/-/issues>`__ or
`service desk email <incoming+datafold-dev-datafold-14878376-issue-@incoming.gitlab.com>`__.
See also the "Contributing" section further below.

.. note::
    The project is under active development in a research-driven environment.

    * Code quality varies ranging from "experimental/early stage" to "well-tested". In
      general, well tested classes are listed in the software documentation and are
      directly accessible through the package levels `pcfold`, `dynfold` or `appfold`
      (e.g. :code:`from datafold.dynfold import ...`. Experimental code is only
      accessible via "deep imports"
      (e.g. :code:`from datafol.dynfold.outofsample import ...`) and may raise a warning
      when using it.
    * There is no deprecation cycle. Backwards compatibility is indicated by the
      package version, where we use a `semantic versioning <https://semver.org/>`__
      policy `[major].[minor].[patch]`, i.e.

         * `major` - making incompatible changes in the (documented) API
         * `minor` - adding functionality in a backwards-compatible manner
         * `patch` - backwards-compatible bug fixes

      We do not intend to indicate a feature complete milestone with version `1.0`.

Quick links
===========

* `Project repository <https://gitlab.com/datafold-dev/datafold>`__
* `Software documentation <https://datafold-dev.gitlab.io/datafold/>`__
* `Tutorials <https://datafold-dev.gitlab.io/datafold/tutorial_index.html>`__
* Feedback of any kind, usage questions, feature requests and bug reports

  * `Issue tracker <https://gitlab.com/datafold-dev/datafold/-/issues>`__,
    requires gitlab account
  * `Email <incoming+datafold-dev-datafold-14878376-issue-@incoming.gitlab.com>`__,
    requires no gitlab account and creates a confident issue via gitlab's
    `service desk <https://docs.gitlab.com/ee/user/project/service_desk.html#how-it-works>`__.

* `Scientific literature <https://datafold-dev.gitlab.io/datafold/references.html>`__


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

Installation of *datafold* requires `Python>=3.7 <https://www.python.org/>`__ with
`pip <https://pip.pypa.io/en/stable/>`__ and
`setuptools <https://setuptools.readthedocs.io/en/latest/>`__ installed (both
packages usually ship with a standard Python installation). The *datafold* package
dependencies are listed in the next section and install automatically.

There are two ways to install *datafold*.

1. **PyPI**: install the core package (excluding tutorials and tests). This
   is the standard way for users. To download the tutorial files separately go to
   `Tutorials <https://datafold-dev.gitlab.io/datafold/tutorial_index.html>`__.
2. **Source**: download or git-clone the entire repository. This way is recommended if you
   want to access the latest (but potentially unstable) development, run tests
   or contribute to *datafold* (see Contributing for details).

From PyPI
---------

*datafold* is hosted on the official Python package index (PyPI)
(https://pypi.org/project/datafold/). To install *datafold* and its dependencies with
:code:`pip` run

.. code-block:: bash

   pip install datafold

.. note::
    If you installed Python with Anaconda, also consider
    `Installation with Anaconda <https://datafold-dev.gitlab.io/datafold/conda_install_info.html>`__.

From source
-----------

1. Download the git repository

   a. If you wish to contribute code, it is required to have
      `git <https://git-scm.com/>`__ installed. Clone the repository with

      .. code-block:: bash

        git clone https://gitlab.com/datafold-dev/datafold.git

   b. Download the source code from the ``master`` branch
      (`zip <https://gitlab.com/datafold-dev/datafold/-/archive/master/datafold-master.zip>`__,
      `tar.gz <https://gitlab.com/datafold-dev/datafold/-/archive/master/datafold-master.tar.gz>`__,
      `tar.bz2 <https://gitlab.com/datafold-dev/datafold/-/archive/master/datafold-master.tar.bz2>`__,
      `tar <https://gitlab.com/datafold-dev/datafold/-/archive/master/datafold-master.tar>`__)

2. Install *datafold* from the root folder of the repository with

   .. code-block:: bash

       python setup.py install

   Add the :code:`--user` flag to install the package and dependencies for the
   current user only.

Dependencies
============

The *datafold* dependencies are managed in
`requirements.txt <https://gitlab.com/datafold-dev/datafold/-/blob/master/requirements.txt>`__
and install with the package manager ``pip``. Note that the tests and tutorials require
further dependencies which are managed in
`requirements-dev.txt <https://gitlab.com/datafold-dev/datafold/-/blob/master/requirements-dev.txt>`__.

*datafold* integrates with common packages from the
`Python scientific computing stack <https://www.scipy.org/about.html>`__:

* `NumPy <https://numpy.org/>`__
   The data structure ``PCManifold`` subclasses from NumPy's
   `ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__
   to capture point clouds with the assumption of being sampled on or near a manifold.
   NumPy is used throughout *datafold* and is the default package for numerical
   data and algorithms.

* `pandas <https://pandas.pydata.org/pandas-docs/stable/index.html>`__
   *datafold* uses pandas'
   `DataFrame <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html>`__
   as a base class for ``TSCDataFrame`` to capture time series data and
   collections thereof. The data structure indexes time, time series ID and
   multiple spatial features. The data is contained in a single object with
   pandas rich functionality to access data - *datafold* includes time series specific
   functionality.

* `scikit-learn <https://scikit-learn.org/stable/>`__
   All *datafold* algorithms that are part of the "machine learning pipeline" align
   to the scikit-learn `API <https://scikit-learn.org/stable/developers/develop.html>`__.
   This is done by deriving the models from
   `BaseEstimator <https://scikit-learn.org/stable/modules/generated/sklearn.base.BaseEstimator.html>`__.
   and appropriate ``MixIns``. *datafold* defines own base classes
   that align with the API in a duck-typing fashion to allow identifying
   dynamics from time series data in ``TSCDataFrame`` objects.

* `SciPy <https://docs.scipy.org/doc/scipy/reference/index.html>`__
   The package is used for elementary numerical algorithms and data structures in
   conjunction with NumPy. This includes (sparse) linear least
   square regression, (sparse) eigenpairs solver and sparse matrices as
   optional data structure for kernel matrices.

How does it compare to other software?
======================================

*Note: the selection only includes other Python packages.*

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

* `PyDMD <https://mathlab.github.io/PyDMD/build/html/index.html>`__
   provides many variants of the `Dynamic Mode Decomposition (DMD) <https://en.wikipedia.org/wiki/Dynamic_mode_decomposition>`__.
   *datafold* provides a wrapper to make models of ``PyDMD`` accessible. However, a
   limitation of ``PyDMD`` is that it only processes single coherent time series, see
   `PyDMD issue 86 <https://github.com/mathLab/PyDMD/issues/86>`__. The DMD models that
   are directly included in *datafold* utilize the functionality of the data
   structure ``TSCDataFrame`` and can therefore process multiple time
   series - in an extreme case only snapshot pairs.

* `PySINDy <https://pysindy.readthedocs.io/en/latest/>`__ specializes on a *sparse*
   identification of dynamical systems to infer governing equations.


Contributing
============

Bug reports, feature requests and user questions
------------------------------------------------

Any contribution (code/tutorials/documentation changes) and feedback is very
welcome. For all correspondence regarding the software please open a new issue in the
*datafold* `issue tracker <https://gitlab.com/datafold-dev/datafold/-/issues>`__ or
`email <incoming+datafold-dev-datafold-14878376-issue-@incoming.gitlab.com>`__ if do not
have a gitlab account (this opens a confidential issue).

All code contributors are listed in the
`contributors file <https://gitlab.com/datafold-dev/datafold/-/blob/master/CONTRIBUTORS>`__.

Setting up *datafold* for development
-------------------------------------

This section describes all steps to set up *datafold* for code development and should be
read before contributing. The *datafold* repository must be cloned via ``git``
(see section to install *datafold* from source above).

Quick set up
^^^^^^^^^^^^

The following bash commands include all steps described in detail below for a quick
set up.

.. code-block:: bash

   # Clone repository (replace FORK_NAMESPACE after forking datafold)
   git clone git@gitlab.com:[FORK_NAMESPACE]/datafold.git
   cd ./datafold/

   # Optional: set up virtual environment
   # Note: if you use Python with Anaconda create a conda environment instead and install pip in it
   #       https://datafold-dev.gitlab.io/datafold/conda_install_info.html
   python -m venv .venv
   source .venv/bin/activate
   pip install --upgrade pip

   # Optional: install datafold
   #   not required if the repository path is included in `PYTHONPATH`
   python setup.py install

   # Install development dependencies and code
   pip install -r requirements-dev.txt

   # Install and run code formatting tools (pre-commit is included in requirements-dev)
   pre-commit install
   pre-commit run --all-files

   # Optional: run tests
   python setup.py test

   # Optional: build documentation
   sphinx-apigen -f -o ./doc/source/_apidoc/ ./datafold/
   sphinx-build -b html ./doc/source/ ./public/

Fork and create merge requests to *datafold*
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Please read and follow the steps of gitlab's
`"Project forking workflow" <https://docs.gitlab.com/ee/user/project/repository/forking_workflow.html>`__.

* `How to create a fork <https://docs.gitlab.com/ee/user/project/repository/forking_workflow.html#creating-a-fork>`__
* `How to create a merge request <https://docs.gitlab.com/ee/user/project/repository/forking_workflow.html#merging-upstream>`__

.. note::
    We have set up a "Continuous Integration" (CI) pipeline. However, the worker (a
    `gitlab-runner`) of the parent repository is not available for forked projects (for
    reasons see
    `here <https://docs.gitlab.com/ee/ci/merge_request_pipelines/#important-notes-about-merge-requests-from-forked-projects>`__).

After you have created a fork you can clone the repository with

 .. code-block:: bash

   git clone git@gitlab.com:[FORK_NAMESPACE]/datafold.git


Install developer dependencies
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The file ``requirements-dev.txt`` in the root directory of the repository contains all
developing dependencies and is readable with :code:`pip`.

The recommended (but optional) way is to install all dependencies into a
`virtual environment <https://virtualenv.pypa.io/en/stable/>`__. This avoids conflicts
with other installed packages. Run from the root directory:

.. code-block:: bash

    python -m venv .venv
    source .venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements-dev.txt

.. note::
    If you are using Python with Anaconda go to
    `Installation with Anaconda <https://datafold-dev.gitlab.io/datafold/conda_install_info.html>`__,
    to set up a ``conda`` environment instead of a ``virtualenv``.

To install the dependencies without a virtual environment run:

.. code-block:: bash

   pip install -r requirements-dev.txt

Install git pre-commit hooks
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The *datafold* source code is automatically formatted with

* `black <https://black.readthedocs.io/en/stable/>`__ for general code formatting
* `isort <https://timothycrosley.github.io/isort/>`__ for sorting Python :code:`import`
  statements alphabetically and in sections.
* `nbstripout <https://github.com/kynan/nbstripout>`__ for removing potentially large
  binary formatted output cells in a Jupyter notebook before the content gets into the git
  history.

It is highly recommended that the tools inspect and format the code *before* the code is
committed to the git history. The tools alter the source code in a deterministic
way, meaning each tool should only format the code once to obtain the desired format.
None of the tool should break the code.

The most convenient way to set up the tools is to install the git commit-hooks via
`pre-commit <https://pre-commit.com/>`__ (installs with the development
dependencies). To install the git-hooks run from root directory:

.. code-block:: bash

   pre-commit install

The installed git-hooks then run automatically prior to each ``git commit``. To format
the current source code without a commit (e.g., for testing purposes or during
development), run from the root directory:

.. code-block:: bash

   pre-commit run --all-files

Run tests
^^^^^^^^^

The tests are executed with Python package
`nose <https://nose.readthedocs.io/en/latest/>`__ (installs with the development
dependencies).

To execute all *datafold* unit tests locally run from the root directory:

.. code-block:: bash

    python setup.py test

Alternatively, you can also run the tests using ``nosetests`` directly, which provides
further options (see ``nosetests --help``)

.. code-block:: bash

    nosetests datafold/ -v

To test whether the tuturials run without raising an error run:

.. code-block:: bash

   nosetests tutorials/ -v

All tests (unit and tutorials) can also be executed remotely in a gitlab "Continuous
Integration" (CI) setup. The pipeline runs for every push to the main repository.

Visit `"gitlab pipelines" <https://docs.gitlab.com/ee/ci/pipelines/>`__ for an
introduction. *datafold*'s pipeline configuration is located in the file
`.gitlab-ci.yml <https://gitlab.com/datafold-dev/datafold/-/blob/master/.gitlab-ci.yml>`__.

Compile and build documentation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The documentation is built with `Sphinx <https://www.sphinx-doc.org/en/stable/>`__ and
various extensions (install with the development dependencies). The source
code is documented with
`numpydoc <https://numpydoc.readthedocs.io/en/latest/format.html#overview>`__ style.

Additional dependencies to build the documentation that *not* install with the
development dependencies:

* `LaTex <https://www.latex-project.org/>`__ to render equations,
* `mathjax <https://www.mathjax.org/>`__ to display equations in the browser
* `graphviz <https://graphviz.org/>`__ to render class dependency graphs
* `pandoc <https://pandoc.org/index.html>`__ to convert between formats (required by
  `nbsphinx` extension that includes the tutorials to the web page).

In a Linux environment, install the packages with

.. code-block:: bash

    apt install libjs-mathjax fonts-mathjax dvipng pandoc graphviz

(This excludes the Latex installation, see available `texlive` packages).

To build the documentation with `Sphinx <https://www.sphinx-doc.org/en/master/>`__:

.. code-block:: bash

   sphinx-apigen -f -o ./doc/source/_apidoc/ ./datafold/
   sphinx-build -b html ./doc/source/ ./public/

The page entry is then located at ``./public/index.html``. Please make sure that the
installation of Sphinx is in the path environment variable.
