What is *datafold*?
=====================

*datafold* is a Python package that provides **data**-driven models for point clouds to
find an *explicit* mani-**fold** parametrization and to identify non-linear
dynamical systems on these manifolds. Informally, a manifold is an usually unknown
geometrical structure on which data is sampled. For point clouds a typical
use case is to parametrize the manifold with an intrinsic lower dimension to enable
non-linear dimension reduction. For time series data the underlying dynamical system is
assumed to have a phase space that is a manifold. The models contained in *datafold* can
process potentially high-dimensional data that lie close to manifolds. For more, please
visit this `introduction <https://datafold-dev.gitlab.io/datafold/intro.html>`_ page.

The software documentation is available at
`https://datafold-dev.gitlab.io/datafold <https://datafold-dev.gitlab.io/datafold>`_,
with executed
`tutorials <https://datafold-dev.gitlab.io/datafold/tutorial_index.html>`_ , which can be
downloaded as
`Jupyter notebooks <https://gitlab.com/datafold-dev/datafold/-/tree/master/tutorials>`_.

The source code is distributed under the `MIT license <https://gitlab
.com/datafold-dev/datafold/-/blob/master/LICENSE>`_.

Any contribution (code/tutorials/documentation improvements) and feedback is
very welcome. Please see the "Contributing" section below for further details.

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
      package version, where we use a `semantic versioning <https://semver.org/>`_
      policy `[major].[minor].[patch]`, i.e.

         * `major` - making incompatible changes in the (documented) API
         * `minor` - adding functionality in a backwards-compatible manner
         * `patch` - backwards-compatible bug fixes

      We do not intend to indicate a feature complete milestone with version `1.0`.

Highlights
==========

*datafold* includes:

* Data structures to handle point clouds on manifolds (``PCManifold``) and time series
  collections (``TSCDataFrame``). The data structures are used both internally and for
  model input/outputs (if applicable).
* An efficient implementation of the ``DiffusionMaps`` model to parametrize a manifold
  from point cloud data or to approximate the Laplace-Beltrami operator eigenfunctions.
* Out-of-sample methods such as the (auto-tuned) Laplacian Pyramids or Geometric
  Harmonics to interpolate general function values on manifold point clouds.
* (Extended-) Dynamic Mode Decomposition (e.g. ``DMDFull`` or ``EDMD``) which
  are data-driven dynamical models built from time series data. To improve the
  model's accuracy, the available data can be transformed with a variety of functions.
  This includes scaling of heterogenous time series features, representing the
  time series in another coordinate system (e.g. Laplace-Beltrami operator) or to
  reconstruct a diffeomorphic copy of the phase space with time delay embedding (cf.
  `Takens theorem <https://en.wikipedia.org/wiki/Takens%27s_theorem>`_).
* ``EDMDCV`` allows the model parameters (including the
  transformation model parameters) to be optimized with cross-validation and
  also accounts for time series splitting.

How does it compare to other software?
======================================

*This section only includes other Python packages, and does not compare the size
(e.g. active developers) of the projects.*

* `scikit-learn <https://scikit-learn.org/stable/>`_
   provides algorithms for the entire machine learning pipeline. The main
   class of models in scikit-learn map feature inputs to a fixed number of target
   outputs for tasks like regression or classification. *datafold* is integrated into the
   scikit-learn API and focuses on the
   `manifold learning algorithms <https://scikit-learn.org/stable/auto_examples/manifold/plot_compare_methods.html#sphx-glr-auto-examples-manifold-plot-compare-methods-py>`_.
   Furthermore, *datafold* includes a model class that can process time
   series data from dynamical systems. The number of outputs may vary: a
   user provides an initial condition (the input) and an arbitrary sampling frequency
   and prediction horizon.

* `PyDMD <https://mathlab.github.io/PyDMD/build/html/index.html>`_
   provides many \
   variants of the `Dynamic Mode Decomposition (DMD) <https://en.wikipedia
   .org/wiki/Dynamic_mode_decomposition>`_. Some of the DMD models are special
   cases of a dictionary of the `Extended Dynamic Mode Decomposition`, while other DMD
   variants are currently not covered in *datafold*. ``datafold.dynfold.dmd.py`` includes
   an (experimental) wrapper for the ``PyDMD`` package to make use of missing DMD models.
   However, a major limitation of ``PyDMD`` is that it only allows single time series as
   input (``numpy.ndarray``), see `PyDMD issue 86 <https://github.com/mathLab/PyDMD/issues/86>`_.
   *datafold* addresses this issue with the data structure ``TSCDataFrame``.

* `PySINDy <https://pysindy.readthedocs.io/en/latest/>`_
   specializes on a *sparse* identification of dynamical systems to infer governing
   equations. `SINDy` is basically a DMD variant and not in the scope of *datafold* and
   note yet included. `PySINDy` also provides time series transformations, which
   are referred to as `library`. This matches the definition of
   `dictionary` in  the `Extended Dynamic Mode Decomposition`). `PySINDy` also supports
   multiple time series but these are managed in lists and not in a single data
   structure.

* `TensorFlow <https://www.tensorflow.org/>`_
   allows data-driven regression/prediction with the main model type
   (deep) neural networks. For manifold learning (Variational) Auto-Encoders are
   suitable and for time series predictions there are recurrent networks such as
   the `Long Short-Term Memory` (LSTM) are a good choice. In general neural networks
   lack a mathematical background theory and are black-box models with a
   non-deterministic learning process that require medium to large sized datasets.
   Nonetheless, for many applications the models are very successful. The models in
   *datafold*, in contrast, have a strong mathematical background, can often be used as
   part of the analysis, have deterministic results and are capable to handle smaller data
   sets.


How to get it?
==============

Installation of *datafold* requires `Python>=3.6 <https://www.python.org/>`_ with
`pip <https://pip.pypa.io/en/stable/>`_ and
`setuptools <https://setuptools.readthedocs.io/en/latest/>`_ installed (the two
packages usually ship with Python). Package dependencies are listed in the
next section.

From PyPI
---------

*datafold* is hosted on the official Python package index (PyPI)
(https://pypi.org/project/datafold/) and can be installed with: 

.. code-block:: bash

   pip install datafold

Alternatively, use :code:`pip3`` if :code:`pip` is reserved for :code:`Python<3`.

From source
-----------

(requires: `git <https://git-scm.com/>`_)

#. Clone the repository

.. code-block:: bash

   git clone git@gitlab.com:datafold-dev/datafold.git


#. Install datafold by executing ``setup.py`` in the root folder

.. code-block:: bash

   python setup.py install

Alternatively, use ``python3`` if ``python`` is reserved for ``Python<3``. Add
a :code:`--user` flag to install datafold only for the current user.


Dependencies
============

The package dependencies are managed in `setup.py <https://gitlab
.com/datafold-dev/datafold/-/blob/master/setup.py>`_ and install with the package
manager ``pip`` (if not already installed).

*datafold* integrates with common packages from the
`Python scientific computing stack <https://www.scipy.org/about.html>`_. Specifically,
this is:

* `NumPy <https://numpy.org/>`_
   The data structure ``PCManifold`` in *datafold* subclasses from NumPy's ``ndarray``
   to model a point cloud sampled on a manifold. A ``PCManifold`` is
   associated with a ``PCManifoldKernel`` that describes the data locality and hence
   the geometry. NumPy is used throughout *datafold* and is the default for numerical
   data and algorithms.

* `pandas <https://pandas.pydata.org/pandas-docs/stable/index.html>`_
   *datafold* addresses time series data in the data structure ``TSCDataFrame``
   which subclasses from Pandas' rich data structure
   `DataFrame <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html>`_.
   Internally, this is again a NumPy array, but a data frame can index time values,
   multiple time series and multiple features. The available time series data can
   then be captured in a single object with easy data slicing and dedicated time series
   functionality.

* `scikit-learn <https://scikit-learn.org/stable/>`_
   All *datafold* algorithms that are part of the "machine learning pipeline" align
   to the scikit-learn `API <https://scikit-learn.org/stable/developers/develop.html>`_.
   This is done by deriving the models from
   `BaseEstimator <https://scikit-learn.org/stable/modules/generated/sklearn.base.BaseEstimator.html>`_.
   or appropriate MixIns. *datafold* also defines own base classes
   that align with ``scikit-learn`` in a duck-typing fashion to allow processing
   time series data in a ``TSCDataFrame`` object.

* `SciPy <https://docs.scipy.org/doc/scipy/reference/index.html>`_
   The package is used for elementary numerical algorithms and data structures in
   conjunction with NumPy. Examples in *datafold* include the (sparse) linear least
   square regression, (sparse) solving for eigenpairs and sparse matrices as optional
   data structure for kernel matrices.

Additional developer dependencies are discussed in the next section.

Contributing
============

Bug reports, feature requests and user questions
------------------------------------------------

Any contribution (code/tutorials/documentation changes) and feedback is very
welcome. For all correspondence regarding the software please open a new issue in the
*datafold* `issue tracker <https://gitlab.com/datafold-dev/datafold/-/issues>`_

All code contributors are listed in the
`contributor list <https://gitlab.com/datafold-dev/datafold/-/blob/master/CONTRIBUTORS>`_.

Setting up development environment
----------------------------------

Install developer dependencies
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The file ``requirements-dev.txt`` lists all developing dependencies and is readable
with ``pip`` (/ ``pip3``) to install the packages:

.. code-block:: bash

   pip install -r requirements-dev.txt

The recommended (but optional) setup is to install all package dependencies into a
`virtual environment <https://virtualenv.pypa.io/en/stable/>`_. This avoids conflicts
with other packages on system level. In order to set up a virtual environment run from
the root directory:

.. code-block:: bash

    python -m venv .venv
    source .venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements-dev.txt

Alternatively, use ``python3`` if ``python`` is reserved for :code:`Python<3`.

Install git pre-commit hooks
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The *datafold* source code is automatically formatted with

* `black <https://black.readthedocs.io/en/stable/>`_ for general formatting
* `isort <https://timothycrosley.github.io/isort/>`_ for sorting :code:`import` statements
  in the source code alphabetically and in sections.
* `nbstripout <https://github.com/kynan/nbstripout>`_ for removing potentially large
  binary formatted output cells in a Jupyter notebook before the content gets into the git
  history.

It is highly recommended that the tools inspect and format the code *before* the code is
committed to the git history. The tools alter the source code in an deterministic
way. That is each tool should only format the code once to obtain the desired source
code format. Also no tool should break the code. To set up the tools, the most
convenient way is to install the git commit-hooks via the tool
`pre-commit <https://pre-commit.com/>`_ (it installs with the development dependencies).
To install the hooks run from root directory:

.. code-block:: bash

   pre-commit install

The installed hooks run then automatically before each ``git commit``. To also execute
the hooks without a commit or for testing purposes run from root directory:

.. code-block:: bash

   pre-commit run --all-files

Run tests
^^^^^^^^^

The tests are executed with `nose <https://nose.readthedocs.io/en/latest/>`_ (installs
with development dependencies). 

To execute all *datafold* unit tests locally run from the root directory:

.. code-block:: bash

   nosetests datafold/ -v

To execute the tutorials (tests check only if an error occurs in the tutorial) run from
the root directory:

.. code-block:: bash

   nosetests tutorials/ -v

All tests (unit and tutorials) are also executed remotely in a gitlab "Continuous
Integration" (CI) setup. The pipeline runs for every push to the
`remote repository <https://gitlab.com/datafold-dev/datafold>`_.

Compile and build documentation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The documentation uses `Sphinx <https://www.sphinx-doc.org/en/stable/>`_ and multiple \
extensions (all install with the development dependencies). The source code is documented
with `numpydoc <https://numpydoc.readthedocs.io/en/latest/format.html#overview>`_ style.

Additional dependencies for building the documentation (**not** contained in
``requirements-dev.txt``):

* `LaTex <https://www.latex-project.org/>`_ to render maths equations,
* `mathjax <https://www.mathjax.org/>`_ to display the LaTex equations in the browser (for
  Linux install packages `libjs-mathjax`, `fonts-mathjax` and `dvipng`).
* `graphviz <https://graphviz.org/>`_ to render class dependency graphs, and
* `pandoc <https://pandoc.org/index.html>`_ to convert between formats (required by
  `nbsphinx` extension that includes the tutorials into the web page documentation).


Note that the documentation also builds remotely in the CI pipeline, either as a
test (all branches except `master`) or to update the web page (only for `master`).

To build the documentation run from root directory

.. code-block:: bash

   sphinx-apigen -f -o ./doc/source/_apidoc/ ./datafold/
   sphinx-build -b html ./doc/source/ ./public/

The html entry is then located at ``./public/index.html``. Please make sure that the
installation of Sphinx is in the path so the Sphinx tools are accessible.
