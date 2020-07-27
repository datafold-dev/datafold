Installation with Anaconda
==========================

**datafold is not available from the conda package manager**.

If you have installed Python with Anaconda on your computer, this page highlights how
to install *datafold* with ``conda`` and ``pip``.

Please also note the
`official instructions <https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-pkgs.html>`__
for package management in Anaconda, and particularly the section
`"Installing non-conda packages" <https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-pkgs.html#installing-non-conda-packages>`__.

While both ``conda`` and ``pip`` are package managers that allow Python packages to be
installed, it is highly recommended to use ``pip`` from within a ``conda`` environment.
Note, when using a ``conda`` environment, a ``virtualenv`` used with ``pip`` is no
longer required.

The following bash commands show how to set up a new ``conda`` environment
(`datafold_venv`), set up ``pip`` in this environment and install *datafold*

.. code-block:: bash

    # Create new environment with pip installed
    conda create -n datafold_venv
    conda activate datafold_venv
    conda install pip

    # Install from pip directly
    pip install datafold

    # Alternatively, install from repository (if downloaded)
    python setup.py install

If the environment is activated, then

.. code-block:: bash

    which pip

should point to ``[conda_envs]/datafold_env/bin`` and **not** to another ``pip``
installation on system level.

.. note::
    While the above procedure should work, to follow the best practices from
    `here <https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-pkgs.html#installing-non-conda-packages>`__
    more strictly, it is recommended to install all packages available from ``conda``
    first, before installing packages via ``pip``. This means, it is recommended to
    install *datafold*'s dependencies (listed in
    `setup.py <https://gitlab.com/datafold-dev/datafold/-/blob/master/setup.py>`__ and/or
    `requirements-dev.txt <https://gitlab.com/datafold-dev/datafold/-/blob/master/requirements-dev.txt>`__
    with :code:`conda install package_name` if applicable.
