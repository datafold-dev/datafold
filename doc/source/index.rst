==========================
**datafold** documentation
==========================

**Date**: |today| **Version**: |version|

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

*datafold* is a `MIT-licensed <https://gitlab.com/datafold-dev/datafold/-/blob/master/LICENSE>`__
Python package containing operator-theoretic models that can identify
dynamical systems from time series data and infer geometrical structures from point
clouds.

Install with

.. code-block::

    pip install datafold

.. panels::

    .. raw:: html

        <div class="tutorials"><i class="fas fa-lightbulb fa-7x"></i></i></div>

    +++++++++++++++++++++++++++++++++++++++++
    :ref:`gettingstarted` | :ref:`tutorialnb`

    ---

    .. raw:: html

        <div class="documentation"><i class="fas fa-book fa-7x"></i></i></div>

    +++++++++++++++++++++
    :ref:`documentation`

    ---

    .. raw:: html

        <div class="contribution"><i class="fas fa-laptop-code fa-7x"></i></i></div>

    +++++++++++++++++++
    :ref:`contribution`

    ---

    .. raw:: html

        <div class="references"><i class="fas fa-graduation-cap fa-7x"></i></i></div>

    +++++++++++++++++
    :ref:`references`

Cite
====

If you use *datafold* in your research, please cite
`this paper <https://joss.theoj.org/papers/10.21105/joss.02283>`__ published in the
*Journal of Open Source Software* (`JOSS <https://joss.theoj.org/>`__).

*Lehmberg et al., (2020). datafold: data-driven models for point clouds and time series on
manifolds. Journal of Open Source Software, 5(51), 2283,* https://doi.org/10.21105/joss.02283

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

Affiliations
============

* **Daniel Lehmberg** (1,2) DL is supported by the German Research Foundation (DFG),
  grant no. KO 5257/3-1 and thanks the research office (FORWIN) of Munich University of
  Applied Sciences and CeDoSIA of TUM Graduate School at the Technical
  University of Munich for their support.

* **Felix Dietrich** (2) FD thanks the Technical University of Munich for their support.


1. Munich University of Applied Sciences
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Faculty of Computer Science and Mathematics in
research group "Pedestrian Dynamics" `www.vadere.org <http://www.vadere.org/>`_,

.. image:: _static/img/hm_logo.png
   :height: 81px
   :width: 300px
   :target: https://www.hm.edu/en/index.en.html

2. Technical University of Munich
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Chair of Scientific Computing in Computer Science

.. image:: _static/img/tum_logo.png
   :height: 75px
   :width: 150px
   :target: https://www.tum.de/en/
