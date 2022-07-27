===================
Welcome to datafold
===================

**Version**: |version| **Date**: |today|

.. the toctree below defines the order and pages that are placed at the top of the page of the
   pydata Sphinx theme

.. toctree::
   :maxdepth: 1
   :hidden:

   gettingstarted
   tutorial_index
   api
   contributing
   references


**What is datafold?**

*datafold* is a |python| :octicon:`link-external` package
containing operator-theoretic models to identify dynamical systems from time series data and
infer geometrical structures from point clouds. |MIT-license|

.. inline usage of a badge
.. |MIT-license| image:: https://img.shields.io/badge/license-MIT-blue
   :target: https://gitlab.com/datafold-dev/datafold/-/blob/master/LICENSE
   :alt: license

See also the `Introduction <https://datafold-dev.gitlab.io/datafold/intro.html>`__ page.

---------------------------------------------------------------------------------

Install the package with ``Python>=3.8``:

.. docu for grid: https://sphinx-design.readthedocs.io/en/furo-theme/grids.html#grid-options
.. free icons from https://fontawesome.com/

.. code-block::

    python -m pip install datafold

.. grid:: 2 2 2 2
    :gutter: 5
    :margin: 1 2 2 5

    .. grid-item-card::
        :text-align: center

        .. raw:: html

            <div class="tutorials"><i class="fas fa-lightbulb fa-7x"></i></i></div>

        :ref:`gettingstarted` | :ref:`tutorialnb`

    .. grid-item-card::
        :text-align: center

        .. raw:: html

            <div class="documentation"><i class="fas fa-book fa-7x"></i></i></div>

        :ref:`documentation`

    .. grid-item-card::
        :text-align: center

        .. raw:: html

            <div class="contribution"><i class="fas fa-laptop-code fa-7x"></i></i></div>

        :ref:`contribution`

    .. grid-item-card::
        :text-align: center

        .. raw:: html

            <div class="references"><i class="fas fa-graduation-cap fa-7x"></i></i></div>

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
   * - Source code formatting and style
     - .. image:: https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white
          :target: https://pre-commit.com/
          :alt: pre-commit
       .. image:: https://img.shields.io/badge/code%20style-flake8-black.svg
          :target: https://flake8.pycqa.org/en/latest/
          :alt: python-flake8
       .. image:: https://img.shields.io/badge/code%20format-black-000000.svg
          :target: https://black.readthedocs.io/en/stable/
          :alt: python-black
       .. image:: https://img.shields.io/badge/code%20format-isort-blue.svg
          :target: https://pycqa.github.io/isort/


Cite
====

If you use *datafold* in your research, please cite our paper |joss-paper| that we
published in the *Journal of Open Source Software* (|joss| :octicon:`link-external`).

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

* **Daniel Lehmberg** (|dl-link| :octicon:`link-external`)

  - from 5/2022 postdoctoral researcher at (1)
  - from 3/2018-5/2022 PhD candidate at both (1) and (2) with funding from the German Research
    Foundation (|dfg| :octicon:`link-external`), grant no. KO 5257/3-1.

* **Felix Dietrich** (1, |fd-link| :octicon:`link-external`)

All source code contributors are listed
`here <https://gitlab.com/datafold-dev/datafold/-/blob/master/CONTRIBUTORS>`__.

(1) Technical University of Munich
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Chair of Scientific Computing in Computer Science (|sccs| :octicon:`link-external`)

.. image:: _static/img/tum_logo.png
   :height: 75px
   :width: 150px
   :target: https://www.tum.de/en/


(2) Munich University of Applied Sciences HM
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Faculty of Computer Science and Mathematics
(|cs-hm| :octicon:`link-external`) in Pedestrian Dynamics Research Group
(|vadere| :octicon:`link-external`)

.. image:: _static/img/hm_logo.png
   :height: 81px
   :width: 300px
   :target: https://www.hm.edu/en/index.en.html

.. Links used above -- the external links (outside datafold) open a new tab (which is why a
.. raw-html directive is used (used from https://stackoverflow.com/a/11718325)

.. |python| raw:: html

   <a href="https://www.python.org/" target="_blank">Python</a>

.. |joss| raw:: html

   <a href="https://joss.theoj.org/" target="_blank">JOSS</a>

.. |cs-hm| raw:: html

   <a href="https://www.cs.hm.edu/en/home/index.en.html" target="_blank">link</a>

.. |sccs| raw:: html

   <a href="https://www.in.tum.de/en/i05/home/" target="_blank">link</a>

.. |dfg| raw:: html

   <a href="https://www.dfg.de/en/index.jsp" target="_blank">DFG</a>

.. |vadere| raw:: html

   <a href="http://www.vadere.org/" target="_blank">link</a>

.. |dl-link| raw:: html

   <a href="https://www.cs.cit.tum.de/sccs/personen/personen/daniel-lehmberg/" target="_blank">link</a>

.. |fd-link| raw:: html

   <a href="https://fd-research.com/" target="_blank">link</a>
