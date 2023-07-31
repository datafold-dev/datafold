.. _contribution:

============
Contributing
============

The maintainers of datafold and code contributors are listed
`here <https://gitlab.com/datafold-dev/datafold/-/blob/master/CONTRIBUTORS>`__.

Getting in touch
----------------

Any code contribution (bug fixes/tutorials/documentation changes) and feedback is very
welcome. Please open a new issue via

* `issue tracker <https://gitlab.com/datafold-dev/datafold/-/issues>`__ or
* `email <incoming+datafold-dev-datafold-14878376-issue-@incoming.gitlab.com>`__ if you
  have no gitlab account (this opens a confidential issue).

Setting up *datafold* for development
-------------------------------------

This section describes all steps to set up *datafold* for code development.

.. note::

    Many tasks of setting up the development environment are also included in the
    `Makefile <https://gitlab.com/datafold-dev/datafold/-/blob/master/Makefile>`__. Run

    .. code-block:: bash

        make help

    in the shell to view the available targets with a short description.

    .. tab-set::

        .. tab-item:: Linux

            In Linux ``make`` is a standard tool and pre-installed.

        .. tab-item:: Windows

            .. warning::

                The ``make`` targets are not fully tested for Windows. Please file an issue if you
                encounter problems.

            In Windows the recommended way is to use ``make`` in the
            `git bash <https://gitforwindows.org/>`__. For this you may
            `install Chocolatey <https://docs.chocolatey.org/en-us/choco/setup>`__ first
            (with administrator rights) and then use the ``choco`` software manger tool to install
            ``make`` with

            .. code-block:: bash

                choco install make

            Chocolatey is also suitable to install non-Python dependencies required for building
            the *datafold*'s html documentation.

.. note::

    The *datafold* repository also includes a
    `Dockerfile <https://gitlab.com/datafold-dev/datafold/-/blob/master/Dockerfile>`__ which
    creates a Docker image suitable for development (e.g. it automatically installs all
    non-Python dependencies necessary to build the documentation). In Linux run

    .. code-block:: bash

        docker build -t datafold .

    to create the Docker image (possibly requires ``sudo`` rights). To start a new Docker
    container in the interactive session run

    .. code-block:: bash

       docker run -v `pwd`:/home/datafold-mount -w /home/datafold-mount/ -it --rm --net=host datafold bash

    This mounts the *datafold* repository within the container (all data is shared
    between the host system and container). To install the dependencies within
    the container execute:

    .. code-block::

        make install_devdeps


Quick set up
^^^^^^^^^^^^

The bash script includes all steps that are detailed below.

.. tab-set::
    .. tab-item:: pip

        .. code-block:: bash

           # Clone repository (replace [NAMESPACE] with your fork or "datafold-dev")
           git clone git@gitlab.com:[NAMESPACE]/datafold.git
           cd ./datafold/

           # Set up Python virtual environment
           python -m venv .venv
           source .venv/bin/activate
           python -m pip install --upgrade pip

           # Install package and development dependencies
           python -m pip install -r requirements-dev.txt

           # Install and run git hooks managed by pre-commit
           python -m pre_commit run --all-files

           # Run tests with coverage and pytest
           python -m coverage run -m pytest datafold/
           python -m coverage html -d coverage/
           python -m coverage report

           # Test if tutorials run without error
           python -m pytest tutorials/

           # Build documentation (writes to "docs/build/")
           # Note that this requires additional third-party dependencies

	       python -m sphinx -M html doc/source/ pages/

    .. tab-item:: conda

        **datafold is not available from the conda package manager**. If you run
        Python with Anaconda's package manager, the recommended way is to set up
        *datafold* in a ``conda`` environment by using ``pip``.

        Also note the
        `official instructions <https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-pkgs.html>`__
        for package management in Anaconda, particularly the subsection on how to
        `install non-conda packages <https://docs.conda
        .io/projects/conda/en/latest/user-guide/tasks/manage-pkgs.html#installing-non-conda-packages>`__.

        .. code-block:: bash

           # Clone repository (replace [NAMESPACE] with your fork or "datafold-dev")
           git clone git@gitlab.com:[NAMESPACE]/datafold.git
           cd ./datafold/

           # Create new conda environment with pip installed
           conda create -n .venv
           conda activate .venv
           conda install pip  # use pip from within the conda environment

           # Install package and development dependencies
           pip install -r requirements-dev.txt

           # Install and run git hooks managed by pre-commit
           python -m pre_commit run --all-files

           # Run tests with coverage and pytest
           python -m coverage run -m pytest datafold/
           python -m coverage html -d coverage/
           python -m coverage report

           # Test if tutorials run without error
           python -m pytest tutorials/

           # Build documentation (writes to "docs/build/")
           # Note that this requires additional third-party dependencies
	       python -m sphinx -M html doc/source/ pages/

Fork and create merge requests to *datafold*
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Please read and follow the steps of gitlab's
`"Project forking workflow" <https://docs.gitlab.com/ee/user/project/repository/forking_workflow.html>`__.

* `How to create a fork <https://docs.gitlab.com/ee/user/project/repository/forking_workflow.html#creating-a-fork>`__
* `How to create a merge request <https://docs.gitlab.com/ee/user/project/repository/forking_workflow.html#merging-upstream>`__

.. note::
    We set up a "Continuous Integration" (CI) pipeline. However, the worker (a
    `gitlab-runner`) of the *datafold* repository is not available for forked projects
    (for background information see
    `here <https://docs.gitlab.com/ee/ci/pipelines/merge_request_pipelines.html#use-with-forked-projects>`__).

After you have created a fork you can clone the repository with:

 .. code-block:: bash

   git clone git@gitlab.com:[NAMESPACE]/datafold.git

(replace [NAMESPACE] accordingly)

Install development dependencies
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The file ``requirements-dev.txt`` in the root directory of the repository contains all
developing dependencies and is readable with :code:`pip`.

.. tab-set::
    .. tab-item:: pip

        The recommended (but optional) way is to install all dependencies into a
        `virtual environment <https://virtualenv.pypa.io/en/stable/>`__. This avoids conflicts
        with other installed packages.

        .. code-block:: bash

            # Create and activate new virtual environment
            python -m venv .venv
            source .venv/bin/activate
            pip install --upgrade pip

            # Install package and extra dependencies
            pip install -r requirements-dev.txt

        To install the dependencies without a virtual environment only run the last statement.

    .. tab-item:: conda

        .. code-block:: bash

               # Create new conda environment with pip installed
               conda create -n .venv
               conda activate .venv
               conda install pip  # use pip from within the conda environment

               # Install package and extra dependencies
               pip install -r requirements-dev.txt

        .. note::
            While the above procedure works, you may also want to follow the best practices
            from `Anaconda <https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-pkgs.html#installing-non-conda-packages>`__
            more strictly. In particular, it is recommended to install package dependencies
            listed in ``requirements-dev.txt`` separately with
            :code:`conda install package_name`, if the package is hosted on ``conda``.


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
* Diverse hooks, such as removing trailing whitespaces, validating configuration
  files or sorting the requirement files.

It is highly recommended that the tools inspect and format the code *before* the code is
committed to the git history. The git hooks alter the source code in a deterministic
way. Each hook should therefore only format the code once to obtain the desired format and
none of the tool should break the code.

Conveniently, all of this is managed via `pre-commit <https://pre-commit.com/>`__
(installs with ``requirements-dev.txt``) and the configuration in
`.pre-commit-config.yaml <https://gitlab.com/datafold-dev/datafold/-/blob/master/.pre-commit-config.yaml>`__

To install the git-hooks locally run from the root directory:

.. code-block:: bash

      python -m pre_commit install

The git-hooks then run automatically prior to each ``git commit``. To format the
current source code without a commit (e.g. for testing purposes or during development),
run from the root directory:

.. code-block:: bash

   python -m pre_commit run --all-files

Run tests
^^^^^^^^^

The unit tests are executed with the test suite
`pytest <https://docs.pytest.org/en/stable/contents.html>`__ and
`coverage.py <https://coverage.readthedocs.io/en/latest/>`__
(both install with ``requirements-dev.txt``)

To execute all unit tests locally run from the root directory:

.. code-block:: bash

    python -m coverage run --branch -m pytest datafold/; \
    python -m coverage html -d ./coverage/; \
    python -m coverage report;

A html coverage report is then located in the folder ``coverage/``. To test if the
tutorials run without raising an error run:

.. code-block:: bash

    python -m pytest tutorials/;

All tests can also be executed remotely in a
`Continuous Integration (CI) setup <https://docs.gitlab.com/ee/ci/pipelines/>`__.
The pipeline runs with every push to the main *datafold* repository. The CI configuration is
located in the
`.gitlab-ci.yml <https://gitlab.com/datafold-dev/datafold/-/blob/master/.gitlab-ci.yml>`__
file.

Compile and build documentation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The `documentation page <https://datafold-dev.gitlab.io/datafold/index.html>`__ is
built with `Sphinx <https://www.sphinx-doc.org/en/master/>`__ and various extensions
(install with ``requirements-dev.txt``). The source code is documented with
`numpydoc <https://numpydoc.readthedocs.io/en/latest/format.html#overview>`__ style.

Additional dependencies to build the documentation that do *not* install with the
development dependencies:

* `LaTex <https://www.latex-project.org/>`__ to render equations,
* `mathjax <https://www.mathjax.org/>`__ to display equations in the browser
* `graphviz <https://graphviz.org/>`__ to render class dependency graphs
* `pandoc <https://pandoc.org/index.html>`__ to convert between formats (required by
  ``nbsphinx`` Sphinx extension that includes the
  `Jupyter tutorials <https://datafold-dev.gitlab.io/datafold/tutorial_index.html>`__
  to the web page).

.. tab-set::

    .. tab-item:: Linux (Debian-based)

        Install the non-Python software with (preferably with `sudo`)

        .. code-block:: bash

            apt install libjs-mathjax fonts-mathjax dvipng pandoc graphviz texlive-base texlive-latex-extra

    .. tab-item:: Windows

        Install the non-Python software with (preferably with administrator rights in the bash)

        .. code-block:: bash

            choco install pandoc miktex graphviz

    .. tab-item:: make

        Install the non-Python software with (best with administrator rights)

        .. code-block:: bash

            make install_docdeps

To build the documentation run:

.. code-block:: bash

    python setup.py build_docs --outdir="./public"

The page entry is then located at ``./public/index.html``. To execute all cells in the
tutorials (Jupyter notebooks) add the flag ``--runtutorials``.
