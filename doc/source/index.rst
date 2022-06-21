===================
Welcome to datafold
===================

**Version**: |version| **Date**: |today|

.. this toctree defines the order and pages that are set at the top of the page

.. toctree::
   :maxdepth: 1
   :hidden:

   gettingstarted
   tutorial_index
   api
   contributing
   references

.. see docu for panels: https://sphinx-panels.readthedocs.io/en/latest/
.. see and select (free) fontawesome icons https://fontawesome.com/

**What is datafold?**

*datafold* is a `Python <https://www.python.org/>`__ package containing operator-theoretic
models to identify dynamical systems from time series data and infer geometrical
structures from point clouds. |MIT-license|

.. inline useage of the badge
.. |MIT-license| image:: https://img.shields.io/badge/license-MIT-blue
   :target: https://gitlab.com/datafold-dev/datafold/-/blob/master/LICENSE
   :alt: license

See also the `Introduction <https://datafold-dev.gitlab.io/datafold/intro.html>`__ page.

---------------------------------------------------------------------------------

Install the package with ``Python>=3.8``:

.. code-block::

    python -m pip install datafold

.. grid:: 2 2 2 2
    :gutter: 2

    .. grid-item-card::

        .. raw:: html

            <div class="tutorials"><i class="fas fa-lightbulb fa-7x"></i></i></div>

        +++++++++++++++++++++++++++++++++++++++++
        :ref:`gettingstarted` | :ref:`tutorialnb`

    .. grid-item-card::

        .. raw:: html

            <div class="documentation"><i class="fas fa-book fa-7x"></i></i></div>

        +++++++++++++++++++++
        :ref:`documentation`

    .. grid-item-card::

        .. raw:: html

            <div class="contribution"><i class="fas fa-laptop-code fa-7x"></i></i></div>

        +++++++++++++++++++
        :ref:`contribution`

    .. grid-item-card::

        .. raw:: html

            <div class="references"><i class="fas fa-graduation-cap fa-7x"></i></i></div>

        +++++++++++++++++
        :ref:`references`


Software management
===================

.. list-table::
   :widths: 25 25
   :header-rows: 1

   * -
     - Badge (Link)
   * - Feedback, questions and bug reports
     - .. image:: https://img.shields.io/badge/gitlab-issue--tracker-blue?logo=gitlab
          :target: https://gitlab.com/datafold-dev/datafold/-/issues

       .. image:: https://img.shields.io/badge/gitlab-service--desk-blue?logo=Minutemailer
          :target: mailto:incoming+datafold-dev-datafold-14878376-issue-@incoming.gitlab.com
   * - Packaging
     - .. image:: https://badge.fury.io/py/datafold.svg
          :target: https://pypi.org/project/datafold/
          :alt: pypi
   * - Latest CI pipeline (branch ``master``)
     - .. image:: https://gitlab.com/datafold-dev/datafold/badges/master/pipeline.svg
          :target: https://gitlab.com/datafold-dev/datafold/pipelines/master/latest
          :alt: pipeline status
   * - Latest test coverage (branch ``master``)
     - .. image:: https://gitlab.com/datafold-dev/datafold/badges/master/coverage.svg
          :target: https://gitlab.com/datafold-dev/datafold/-/jobs/artifacts/master/file/coverage/index.html?job=unittests
          :alt: coverage report
   * - Code-related
     - .. image:: https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white
          :target: https://pre-commit.com/
          :alt: pre-commit
       .. image:: https://img.shields.io/badge/code%20style-flake8-black
          :target: https://flake8.pycqa.org/en/latest/
          :alt: python-flake8
       .. image:: https://img.shields.io/badge/code%20format-black-000000.svg
          :target: https://black.readthedocs.io/en/stable/
          :alt: python-black


Cite
====

If you use *datafold* in your research, please cite our paper |joss-paper| that we
published in the *Journal of Open Source Software* (`JOSS <https://joss.theoj.org/>`__).

.. |joss-paper| image:: https://joss.theoj.org/papers/10.21105/joss.02283/status.svg
   :target: https://doi.org/10.21105/joss.02283
   :alt: joss-paper

*Lehmberg et al., (2020). datafold: data-driven models for point clouds and time series on
manifolds. Journal of Open Source Software, 5(51), 2283,*
https://doi.org/10.21105/joss.02283

.. dropdown:: Bibtex

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

Software maintainer and affiliation
===================================

* **Daniel Lehmberg** (`link <https://www.cs.cit.tum.de/sccs/personen/personen/daniel-lehmberg/>`__)

  - from 2022 (1): Postdoctoral researcher
  - from 2018-2022 (1,2): PhD candidate with funding from the German Research Foundation
    (`DFG <https://www.dfg.de/en/index.jsp>`__), grant no. KO 5257/3-1 and support by the
    research office (FORWIN) at (2) and the Center of Doctoral Studies in Informatics
    (`CeDoSIA <https://www.gs.tum.de/en/cedosia/home/>`__) at (1).

* **Felix Dietrich** (`link <https://fd-research.com/>`__) (1)

All source code contributors are listed
`here <https://gitlab.com/datafold-dev/datafold/-/blob/master/CONTRIBUTORS>`__.


1. Technical University of Munich
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Chair of Scientific Computing in Computer Science
(`link <https://www.in.tum.de/en/i05/home/>`__)

.. image:: _static/img/tum_logo.png
   :height: 75px
   :width: 150px
   :target: https://www.tum.de/en/

2. Munich University of Applied Sciences HM
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Faculty of Computer Science and Mathematics in
Pedestrian Dynamics Research Group  (`link <http://www.vadere.org/>`__)

.. image:: _static/img/hm_logo.png
   :height: 81px
   :width: 300px
   :target: https://www.hm.edu/en/index.en.html
