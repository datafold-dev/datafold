
What is **datafold**?
=====================

**datafold** is a Python package consisting of data-driven algorithms with
manifold assumption. This means that **datafold** aims to process high
-dimensional data that lie on an (unknown) manifold with intrinsic lower-dimension. One
major objective of **datafold** is to build build non-parametric models of available time
series data.

The software documentation is available at https://datafold-dev.gitlab.io/datafold ,
which also includes tutorials that can be downloaded as Jupyter notebooks
`here <https://gitlab.com/datafold-dev/datafold/-/tree/master/tutorials>`_.

The source code is distributed under the `MIT license <https://gitlab
.com/datafold-dev/datafold/-/blob/master/LICENSE>`_.

.. warning::
    The project is under active development in an early stage.

    * Code quality varies ranging from "experimental" to "well-tested". If using
      experimental code warnings show up.
    * There is no backwards compatibility, yet.
    * The API may change without warning or deprecation cycle.

    We use a semantic versioning policy `[major].[minor].[patch]`, i.e.

    * `major` - making incompatible API changes
    * `minor` - adding functionality in a backwards-compatible manner
    * `patch` - backwards-compatible bug fixes

Highlights
==========

* Data structures to handle point clouds on manifolds and collections of
  series data. 
* Provides various distance algorithms and kernels (sparse/dense).  
* Efficient implementation of Diffusion Map algorithm, to parametrize a manifold
  from data or to approximate the Laplace-Beltrami operator.
* Out-of-sample methods: (auto-tuned) Laplacian Pyramids or Geometric Harmonics
* (Extended-) Dynamic Mode Decomposition (e.g. classes ``DMDFull`` and ``EDMD``) which uses
  the introduced data structures to transform time series data with a dictionary (e.g.
  scaling, delay embedding, ...) and to extract the dynamics via the Koopman operator
  approximation. Furthermore, ``EDMDCV``  allows to optimize the parameters of the EDMD
  model with time series cross validation methods.

How does it compare to other software?
--------------------------------------

The comparison is only to Python packages.

* `scikit-learn <https://scikit-learn.org/stable/>`_
   provides classical algorithms of the entire machine learning pipeline. However, the
   main model cases are single input to single target output. **datafold** integrates
   into scikit-learn (like `PyDMD` and `PySINDy` below) but datafold also generalizes
   models to learn dynamical systems to time series predictions or regressions (i.e.
   initial condition to a variable number of time time value evaluations of the system).

* `PyDMD <https://mathlab.github.io/PyDMD/build/html/index.html>`_
   provides numerous \
   variants of the `Dynamic Mode Decomposition (DMD) <https://en.wikipedia
   .org/wiki/Dynamic_mode_decomposition>`_ . Some of the variants are special cases of
   a dictionary of the `Extended Dynamic Mode Decomposition`, other cases are currently
   not covered in **datafold**. An (experimental) wrapper to use from the `PyDMD` package
   is also provided in `datafold.dmd.py` module. A major limitation of `PyDMD` is that
   it only allows single time series (`PyDMD issue 86 <https://github.com/mathLab/PyDMD/issues/86>`_),
   represented as NumPy array and therefore without time
   information. **datafold** provides data structures that allow multiple time series
   (also of different lengths). The situation of having multiple time series arises if
   splitting time series for  cross-validation of for transient dynamical systems where
   multiple time series are required to sample the phase space. Furthermore,
   **datafold** provides an `Extended Dynamic Mode Decomposition` (EDMD) which gives a
   formed structure of transforming the time series before applying the dynamic
   mode decomposition -- that otherwise have to be carried out by the user before using
   **datafold**.

* `PySINDy <https://pysindy.readthedocs.io/en/latest/>`_
   specializes on `sparse` identification of dynamical systems and provides
   optimization routines. Like **datafold** it provides time series transformations (e.g.
   Fourier basis or polynomials), which in PySINDy is referred to as `library` and
   matches the defintion of `dictionary` in the `Extended Dynamic Mode Decomposition`.
   Multiple time series are supported by providing a list of NumPy arrays with a
   separate list of arrays about time information.

* `tensorflow <https://www.tensorflow.org/>`_
   allows data-driven time series regression/prediction. The main model type are (deep)
   neural networks, especially recurrent networks such as the `Long Short-Term Memory`
   (LSTM). Neural networks often lack of theory and can are viewed as a black box
   (nonetheless often quite successful). **datafold** focuses on operator theoretic
   models which at the core are linear dynamical systems and therefore allow the model
   to be better analyzed or extended (e.g. with control). Furthermore, the models of
   **datafold** can also deal with smaller datasets well (but also depending on the
   systems properties).


How to get it?
==============

Installation of **datafold** requires ``Python>=3.6``, `pip <https://pip.pypa.io/en/stable
/>`_ and `setuptools <https://setuptools.readthedocs.io/en/latest/>`_ installed
(usually already set up when installing Python). Package dependencies are listed in the
next section.

From PyPI (NOTE: project not public yet)
----------------------------------------

**datafold** is also hosted on the official Python package index PyPI
(https://pypi.org/project/datafold/) and can be installed with: 

.. code-block:: bash

   pip install datafold

Alternatively use ``pip3`` if ``pip`` is reserved for ``Python<3``.

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
(if required) with the package manager (see next section).

**datafold** integrates tightly with common packages from the `Python scientific computing stack <https://www.scipy.org/about.html>`_. Mainly this is:

* `NumPy <https://numpy.org/>`_
    ``PCManifold`` subclasses from NumPy's ``ndarray`` to represent point clouds lying
    on a manifold. For this to every array a kernel to describe local properties is
    attached. Also all models (except time series predictions) accept ``ndarrays``
    as input.

* `pandas <https://pandas.pydata.org/pandas-docs/stable/index.html>`_
   To deal with multiple time series with different properties (e.g. different
   lengths and time values), the data structure ``TSCDataFrame`` subclasses from
   pandas' rich data structure `DataFrame <https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html>`_
   bring the data into a defined format and allow easier access via new attributes.

* `scikit-learn <https://scikit-learn.org/stable/>`_
    All **datafold** algorithms that part of the "machine learning
    pipeline" subclass from the `BaseEstimator <https://scikit-learn.org/stable/modules
    /generated/sklearn.base.BaseEstimator.html>`_. **datafold** follows the philosopy of
    scikit-learn where ever possible. For required generalizations (such as dealing
    with ``TSCDataFrame`` as input and output) own base classes are provided.

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
conflicting dependencies with other packages. Setting up the environment run from the
root directory:

.. code-block:: bash

    python -m venv .venv
    source .venv/bin/activate
    pip3 install --upgrade pip
    pip3 install -r requirements-dev.txt


Install git pre-commit hooks
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The **datafold** source code is automatically formatted with


* `black <https://black.readthedocs.io/en/stable/>`_ for code auto formatting
* `isort <https://timothycrosley.github.io/isort/>`_ for sorting :code:`import` statements
   alphabetically and sections.
* `nbstripout <https://github.com/kynan/nbstripout>`_ for removing potentially large (in
   mega bytes) binary formatted output cells of Jupyter notebooks before they get
   into the git history.

It is highly recommended that the tools inspect format the code *before* the code is
commited to the git history. The tools change the source code in an deterministic way
and should not alter the behavior. To set this up, the most convenient way is to
install git commit-hooks via the tool `pre-commit <https://pre-commit.com/>`_ (it
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

To execute the tutorials (only error checks) run from the root directory:

.. code-block:: bash

   nosetests tutorials/ -v

All tests (unit and tutorials) are executed remotely in a "Continuous Integration" (CI) 
setup. The pipeline runs for every push to the
`remote repository <https://gitlab.com/datafold-dev/datafold>`_.

Compile and build documentation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The documentation uses `Sphinx <https://www.sphinx-doc.org/en/stable/>`_ and multiple \
extensions (all install with the development dependencies).

Additional dependencies (not contained in ``requirements-dev.txt``):

* `LaTex <https://www.latex-project.org/>`_ to render maths equations,
* `graphviz <https://graphviz.org/>`_ to render class dependency graphs, and
* `pandoc <https://pandoc.org/index.html>`_ to convert between formats (required by
   `nbsphinx` extension).

Note that the documentation also builds remotely in the CI pipeline, either as a
test (all branches but `master`) or build to update the hosted docuementation web page
(only on master).

The **datafold** source code is documented with \
`numpydoc <https://numpydoc.readthedocs.io/en/latest/format.html#overview>`_ style.

To build the documentation run from root

.. code-block:: bash

   sphinx-apigen -f -o ./doc/source/_apidoc/ ./datafold/
   sphinx-build -b html ./doc/source/ ./public/

The html entry is then located at ``./public/index.html``.
