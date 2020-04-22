
What is **datafold**?
=====================

**datafold** is a Python package consisting of data-driven algorithms with
manifold assumption. This means that **datafold** aims to process high
-dimensional data that lie on an (unknown) geometry (manifold) with intrinsic
lower-dimension. One major objective of **datafold** is to build build non-parametric
models of available time series data.

The software documentation is available at https://datafold-dev.gitlab.io/datafold ,
and also includes tutorials that can be downloaded as Jupyter notebooks
`here <https://gitlab.com/datafold-dev/datafold/-/tree/master/tutorials>`_.

The source code is distributed under the `MIT license <https://gitlab
.com/datafold-dev/datafold/-/blob/master/LICENSE>`_.

.. note::
    The project is under active development in an early stage.

    * Code quality varies ranging from "experimental/early stage" to "well-tested". In
      general, well tested classes are listed in the software documentation and are
      accessible through the package levels `pcfold`, `dynfold` or `appfold` directly
      (e.g. :code:`from datafold.dynfold import DiffusionMaps`. Experimental code is only
      accessible with "deep imports"
      (e.g. :code:`from datafol.dynfold.outofsample import ...`) and may raise a warning
      when using it.
    * There is no deprecation cycle and backwards compatibility is indicated by the
      package version, where we use `semantic versioning <https://semver.org/>`_
      policy `[major].[minor].[patch]`, i.e.

         * `major` - making incompatible changes in the (documented) API
         * `minor` - adding functionality in a backwards-compatible manner
         * `patch` - backwards-compatible bug fixes

Highlights
==========

**datafold** provides:

* Data structures to handle point clouds on manifolds (`PCManifold`) and time series
  collections (`TSCDataFrame`). The data structures are both used internally and for
  user input.
* Access to various algorithms to compute distance metrics and kernels (sparse/dense).
* An efficient implementation of the Diffusion Map algorithm to parametrize a manifold
  from point cloud or to approximate the the eigenfunctions of the Laplace-Beltrami
  operator.
* Out-of-sample methods such as the (auto-tuned) Laplacian Pyramids or Geometric
  Harmonics to interpolate general function values on manifold point cloud data.
* (Extended-) Dynamic Mode Decomposition (e.g. classes ``DMDFull`` or ``EDMD``) which
  are data-driven dynamical models built from time series samples. To improve the
  model accuracy There are classes that work with time series collections to transform
  the time series data. This can be scaling of non-homogenous features or to represent
  the time series in other coordinates (e.g. time delay embedding). Furthermore,
  ``EDMDCV``  allows to optimize the model parameters, which include set up
  time series transformations.

How does it compare to other software?
--------------------------------------

(Disclaimer: we only consider other Python packages)

* `scikit-learn <https://scikit-learn.org/stable/>`_
   provides "classical" algorithms of the entire machine learning pipeline. The main
   class of models map feature inputs to a fixed number of target output(s), such as in
   regression or classification. **datafold** also includes models
   of this class (Note the scikit learn integration of datafold in section
   "Dependencies") but also include models that generalize to data-driven models of
   dynamical systems. The number of outputs from a single input (i.e. initial
   condition) then vary depending on the user of what time interval and what sampling
   rate to sample.


* `PyDMD <https://mathlab.github.io/PyDMD/build/html/index.html>`_
   provides numerous \
   variants of the `Dynamic Mode Decomposition (DMD) <https://en.wikipedia
   .org/wiki/Dynamic_mode_decomposition>`_ . Some of the variants are special cases of
   a dictionary of the `Extended Dynamic Mode Decomposition`, other DMD variants are
   currently not covered in **datafold**. In `datafold.dynfold.dmd.py` is an
   (experimental) wrapper for the `PyDMD` package. A major limitation of `PyDMD`,
   however, is that it only allows single time series as input (`numpy.ndarray`),
   see `PyDMD issue 86 <https://github.com/mathLab/PyDMD/issues/86>`_. In **datafold**
   multiple time series (of different length) is solved with the data structure
   `TSCDataFrame`.

* `PySINDy <https://pysindy.readthedocs.io/en/latest/>`_
   specializes on a `sparse` identification of dynamical systems and infer governing
   equations. `SINDy` is basically another DMD variant and not yet implemented in
   **datafold**. `PySINDy` also provides time series transformations, which
   in `PySINDy` are included in a `library` (which matches the defintion of `dictionary`
   in  the `Extended Dynamic Mode Decomposition`). `PySINDy` also supports multiple time
   series (lists of time series and correspinding time values).

* `tensorflow <https://www.tensorflow.org/>`_
   allows data-driven time series regression/prediction. The main model type are (deep)
   neural networks. For manifold learning (Variational) Autoencoders aim to learn
   manifold geometries. For time series predictions the recurrent networks such as
   the `Long Short-Term Memory` (LSTM) are suitable. Neural networks lack of
   mathematical background theory and the learning process is not deterministic. The
   models are basically a black box but nonetheless often very successful. There are
   scientific works that combine neural networks, but this is currently not
   covered by **datafold**. Instead, the focus is on operator theoretic
   models which at the core are linear dynamical systems and, therefore, allow the model
   to be better analyzed or extended (e.g. to control). Furthermore, the models in
   **datafold** can also deal with small data sets while neural networks often need a
   medium to large number of samples.


How to get it?
==============

Installation of **datafold** requires ``Python>=3.6``, `pip <https://pip.pypa.io/en/stable
/>`_ and `setuptools <https://setuptools.readthedocs.io/en/latest/>`_ installed
(the two packages usually ship with Python). Package dependencies are listed in the
next section.

From PyPI
---------

**datafold** is hosted on the official Python package index (PyPI)
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


#. Install datafold by executing ``setup.py`` from the root folder

.. code-block:: bash

   python setup.py install

Alternatively use ``python3`` if ``python`` is reserved for ``Python<3``.

add :code:`--user` flag to install it only for the current user.


Dependencies
============

The dependencies are managed in `setup.py <https://gitlab
.com/datafold-dev/datafold/-/blob/master/setup.py>`_ and install
(if required) with the package manager `pip`.

**datafold** integrates with common packages from the
`Python scientific computing stack <https://www.scipy.org/about.html>`_. Specifically,
this is:

* `NumPy <https://numpy.org/>`_
    The data structure ``PCManifold`` in **datafold** subclasses from NumPy's ``ndarray``
    to represent a point cloud on a manifold. A `PCManifold` point cloud is associated
    with a kernel that describes the data locality and hence the geometry.

* `pandas <https://pandas.pydata.org/pandas-docs/stable/index.html>`_
   **datafold** addresses time series data in the data structure ``TSCDataFrame``
   which subclasses from Pandas' rich data structure
   `DataFrame <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html>`_.
   The entire time series data is captured in a single object but accessing single time
   series, features or time values is then easy.

* `scikit-learn <https://scikit-learn.org/stable/>`_
   All **datafold** algorithms that part of the "machine learning
   pipeline" align to the
   `API <https://scikit-learn.org/stable/developers/develop.html>`_ of scikit-learn.
   All models subclass from
   `BaseEstimator <https://scikit-learn.org/stable/modules/generated/sklearn.base.BaseEstimator.html>`_.
   **datafold** provides also own base classes
   that orientate with scikit-learn (in a duck-typing way) for required
   generalizations, such as dealing with ``TSCDataFrame`` as input and output), .

* `SciPy <https://docs.scipy.org/doc/scipy/reference/index.html>`_
    Used for elementary numerical algorithms and data structures, such as linear least
    square regression, solving for eigenpairs and sparse matrices.

Additional developer dependencies are in the next section.


Contributing
============

Bug reports and user questions
------------------------------

For all correspondence regarding the software please open a new issue in the
**datafold** `issue tracker <https://gitlab.com/datafold-dev/datafold/-/issues>`_

All code contributors are listed in the
`contributor list <https://gitlab.com/datafold-dev/datafold/-/blob/master/CONTRIBUTORS>`_.

Setting up development environment
----------------------------------

Install developer dependencies
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In the file ``requirements-dev.txt`` all developing dependencies are listed. Install the
dependencies with ``pip`` (or ``pip3``):

.. code-block:: bash

   pip install -r requirements-dev.txt

The recommended (but optional) way is to install all packages into a
`virtual environment <https://virtualenv.pypa.io/en/stable/>`_ such that there are no
conflicting dependencies with other system packages. Setting up the environment run from
the root directory:

.. code-block:: bash

    python -m venv .venv
    source .venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements-dev.txt

Install git pre-commit hooks
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The **datafold** source code is automatically formatted with


* `black <https://black.readthedocs.io/en/stable/>`_ for code auto formatting
* `isort <https://timothycrosley.github.io/isort/>`_ for sorting :code:`import` statements
  alphabetically and sections.
* `nbstripout <https://github.com/kynan/nbstripout>`_ for removing potentially large (in
  mega bytes) binary formatted output cells of Jupyter notebooks before they get
  into the git history.

It is highly recommended that the tools inspect and format the code *before* the code is
commited to the git history. The lsited tools alter the source code in an deterministic
way and should not break the code. To set up the tools, the most convenient way is to
install the git commit-hooks via the tool `pre-commit <https://pre-commit.com/>`_ (it
installs with the development dependencies). To install the hooks run from root directory:

.. code-block:: bash

   pre-commit install

The installed hooks run before each commit. To also execute the hooks without a commit or
for testing purposes) run from root directory:

.. code-block:: bash

   pre-commit run --all-files

Run tests
^^^^^^^^^

The tests are executed with `nose <https://nose.readthedocs.io/en/latest/>`_ (installs
with development dependencies). 

To execute all **datafold** unit tests locally run from the root directory:

.. code-block:: bash

   nosetests datafold/ -v

To execute the tutorials (checks only if an error occurs) run from the root
directory:

.. code-block:: bash

   nosetests tutorials/ -v

All tests (unit and tutorials) are executed remotely in a gitlab "Continuous Integration"
(CI) setup. The pipeline runs for every push to the
`remote repository <https://gitlab.com/datafold-dev/datafold>`_.

Compile and build documentation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The documentation uses `Sphinx <https://www.sphinx-doc.org/en/stable/>`_ and multiple \
extensions (all install with the development dependencies).

Additional dependencies (not contained in ``requirements-dev.txt``):

* `LaTex <https://www.latex-project.org/>`_ to render maths equations,
* `graphviz <https://graphviz.org/>`_ to render class dependency graphs, and
* `pandoc <https://pandoc.org/index.html>`_ to convert between formats (required by
  `nbsphinx` extension that includes tutorials into the documentation).

Note that the documentation also builds remotely in the CI pipeline, either as a
test (all branches but `master`) or to update the web page (only on `master` branch).

The **datafold** source code is documented with \
`numpydoc <https://numpydoc.readthedocs.io/en/latest/format.html#overview>`_ style. To
build the documentation run from root directory

.. code-block:: bash

   sphinx-apigen -f -o ./doc/source/_apidoc/ ./datafold/
   sphinx-build -b html ./doc/source/ ./public/

The html entry is then located at ``./public/index.html``.