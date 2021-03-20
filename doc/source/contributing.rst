.. _contribution:

============
Contributing
============

.. toctree::
   :hidden:

   conda_install_info

All code contributors are listed in the
`contributors file <https://gitlab.com/datafold-dev/datafold/-/blob/master/CONTRIBUTORS>`__.

Bug reports, feature requests and user questions
------------------------------------------------

Any contribution (code/tutorials/documentation changes) and feedback is very
welcome. Please open a new issue in the

* `issue tracker <https://gitlab.com/datafold-dev/datafold/-/issues>`__ or
* `email <incoming+datafold-dev-datafold-14878376-issue-@incoming.gitlab.com>`__ if you do
  not have a gitlab account (this opens a confidential issue).


Setting up *datafold* for development
-------------------------------------

This section describes all steps for code development and should be
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
   # Note: If you use Python with Anaconda create a conda environment instead and install pip in it
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
    background see
    `here <https://docs.gitlab.com/ee/ci/merge_request_pipelines/#important-notes-about-merge-requests-from-forked-projects>`__).

After you have created a fork you can clone the repository with

 .. code-block:: bash

   git clone git@gitlab.com:[FORK_NAMESPACE]/datafold.git


Install development dependencies
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

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

To install the dependencies without a virtual environment only run the last statement.

.. note::
    If you are using Python with Anaconda go to :ref:`install_conda`
    to set up a ``conda`` environment instead of a ``virtualenv``.


Install git pre-commit hooks
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The *datafold* source code and configuration files are automatically formatted and checked
with

* `black <https://black.readthedocs.io/en/stable/>`__ for general code formatting
* `isort <https://timothycrosley.github.io/isort/>`__ for sorting Python :code:`import`
  statements alphabetically and in sections.
* `nbstripout <https://github.com/kynan/nbstripout>`__ to remove potentially large
  binary formatted output cells in Jupyter notebooks before the content bloats the
  git history.
* `mypy <http://mypy-lang.org/>`__ for static type checking (if applicable).
* Diverse, such as removing trailing whitespaces, validating configuration files or
  sorting the requirement files.

It is highly recommended that the tools inspect and format the code *before* the code is
committed to the git history. The tools alter the source code in a deterministic
way. Each tool should therefore only format the code once to obtain the desired format.
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

The tests are executed with the test suite
`pytest <https://docs.pytest.org/en/stable/contents.html>`__ and
`coverage.py <https://coverage.readthedocs.io/en/latest/>`__
(both install with ``requirements-dev.txt``)

To execute all unit tests locally run from the root directory:

.. code-block:: bash

    coverage run -m pytest datafold/
    coverage html -d coverage/
    coverage report

To test whether the tuturials run without raising an error run:

.. code-block:: bash

   pytest tutorials/

All tests can also be executed remotely in a gitlab
`"Continuous Integration" (CI) setup <https://docs.gitlab.com/ee/ci/pipelines/>`__.
The pipeline runs with every push to the main repository. The CI configuration is located
in the file
`.gitlab-ci.yml <https://gitlab.com/datafold-dev/datafold/-/blob/master/.gitlab-ci.yml>`__
.

Compile and build documentation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The documentation is built with `Sphinx <https://www.sphinx-doc.org/en/stable/>`__ and
various extensions (install with the development dependencies). The source
code is documented with
`numpydoc <https://numpydoc.readthedocs.io/en/latest/format.html#overview>`__ style.

Additional dependencies to build the documentation that do *not* install with the
development dependencies:

* `LaTex <https://www.latex-project.org/>`__ to render equations,
* `mathjax <https://www.mathjax.org/>`__ to display equations in the browser
* `graphviz <https://graphviz.org/>`__ to render class dependency graphs
* `pandoc <https://pandoc.org/index.html>`__ to convert between formats (required by
  `nbsphinx` Sphinx extension that includes the Jupyter tutorials to the web page).

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
