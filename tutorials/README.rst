.. NOTE: this file was automatically generated with 'generate_tutorials_page.py' (located in 'datafold/doc/source/'). Navigate to this file, if you wish to change the content of this page.

.. _tutorialnb:

=========
Tutorials
=========

This page contains tutorials and code snippets to
showcase *datafold's* API. All tutorials can be viewed online. If you want to
execute the notebooks in Jupyter, please follow the instructions below in section
"Run notebooks with Jupyter".

List
----

`Download <https://gitlab.com/datafold-dev/datafold/-/archive/master/datafold-master.zip?path=tutorials/>`__ all tutorials in a zipped file.

.. toctree::
    :hidden:

    tutorial_01_datastructures
    tutorial_02_pcm_subsampling
    tutorial_03_dmap_scurve
    tutorial_04_dmap_digitclustering
    tutorial_05_roseland_scurve_digits
    tutorial_06_dmap_mahalanobis_kernel
    tutorial_07_jsf_common_eigensystem
    tutorial_08_gh_oos
    tutorial_09_edmd_limitcycle
    tutorial_10_dmd_control
    tutorial_11_edmd_control
    tutorial_12_kmpc_flowcontrol
    tutorial_13_kmpc_motor_engine
    tutorial_14_online_dmd


* ``01_datastructures.ipynb`` (`download <https://gitlab.com/datafold-dev/datafold/-/raw/master/tutorials/01_datastructures.ipynb?inline=false>`__, `doc <https://datafold-dev.gitlab.io/datafold/tutorial_01_datastructures.html>`__)
    We introduce *datafold*'s basic data structures for time series collection data. The data structures are used in model implementations and for input/output specification of models.
* ``02_pcm_subsampling.ipynb`` (`download <https://gitlab.com/datafold-dev/datafold/-/raw/master/tutorials/02_pcm_subsampling.ipynb?inline=false>`__, `doc <https://datafold-dev.gitlab.io/datafold/tutorial_02_pcm_subsampling.html>`__)
    We show how the ``PCManifold`` data structure can be used to subsample a manifold point cloud uniformly.

    **Warning**
        The tutorial generates a large dataset with 10 Mio. samples by default. This may have to be reduced, depending on the available computer memory.

* ``03_dmap_scurve.ipynb`` (`download <https://gitlab.com/datafold-dev/datafold/-/raw/master/tutorials/03_dmap_scurve.ipynb?inline=false>`__, `doc <https://datafold-dev.gitlab.io/datafold/tutorial_03_dmap_scurve.html>`__)
    We use the ``DiffusionMaps`` model to compute lower dimensional embeddings of an S-curved point cloud manifold. We also select the best combination of intrinsic parameters automatically with an optimization routine.

    **References**: `scikit-learn tutorial <https://scikit-learn.org/stable/auto_examples/manifold/plot_compare_methods.html#sphx-glr-auto-examples-manifold-plot-compare-methods-py>`__ :octicon:`link-external`
* ``04_dmap_digitclustering.ipynb`` (`download <https://gitlab.com/datafold-dev/datafold/-/raw/master/tutorials/04_dmap_digitclustering.ipynb?inline=false>`__, `doc <https://datafold-dev.gitlab.io/datafold/tutorial_04_dmap_digitclustering.html>`__)
    We use the ``DiffusionMaps`` model to cluster data from handwritten digits and highlight its out-of-sample capabilities. This example is taken from the scikit-learn package, so the the method can be compared against other common manifold learning algorithms.

    **References**: `scikit-learn tutorial <https://scikit-learn.org/stable/auto_examples/manifold/plot_lle_digits.html>`__ :octicon:`link-external`
* ``05_roseland_scurve_digits.ipynb`` (`download <https://gitlab.com/datafold-dev/datafold/-/raw/master/tutorials/05_roseland_scurve_digits.ipynb?inline=false>`__, `doc <https://datafold-dev.gitlab.io/datafold/tutorial_05_roseland_scurve_digits.html>`__)
    We use a ``Roseland`` model to compute lower dimensional embeddings of an S-curved point cloud manifold and to cluster data from handwritten digit.

    **References**: `scikit-learn tutorial 1 <https://scikit-learn.org/stable/auto_examples/manifold/plot_compare_methods.html#sphx-glr-auto-examples-manifold-plot-compare-methods-py>`__ :octicon:`link-external` | `scikit-learn tutorial 2 <https://scikit-learn.org/stable/auto_examples/manifold/plot_lle_digits.html>`__ :octicon:`link-external`
* ``06_dmap_mahalanobis_kernel.ipynb`` (`download <https://gitlab.com/datafold-dev/datafold/-/raw/master/tutorials/06_dmap_mahalanobis_kernel.ipynb?inline=false>`__, `doc <https://datafold-dev.gitlab.io/datafold/tutorial_06_dmap_mahalanobis_kernel.html>`__)
    We highlight how to use the Mahalanobis kernel within ``DiffusionMaps``. The key feature of this kernel is that it can yield embeddings that are invariant to the observation function.

    **Warning**
        The implementation of Mahalanobis kernel in datafold is still experimental and should be used with care. Contributions (mainly testing / documentation) are welcome!

* ``07_jsf_common_eigensystem.ipynb`` (`download <https://gitlab.com/datafold-dev/datafold/-/raw/master/tutorials/07_jsf_common_eigensystem.ipynb?inline=false>`__, `doc <https://datafold-dev.gitlab.io/datafold/tutorial_07_jsf_common_eigensystem.html>`__)
    We use ``JointlySmoothFunctions`` to learn commonly smooth functions from multimodal data. We also demonstrate the out-of-sample capabilities of the method.
* ``08_gh_oos.ipynb`` (`download <https://gitlab.com/datafold-dev/datafold/-/raw/master/tutorials/08_gh_oos.ipynb?inline=false>`__, `doc <https://datafold-dev.gitlab.io/datafold/tutorial_08_gh_oos.html>`__)
    We showcase the out-of-sample extension for manifold learning models such as the ``DiffusionMaps`` model. For this we use the ``GeometricHarmonicsInterpolator`` for forward and backwards interpolation.

    **Warning**
        The tutorial requires also the Python package `scikit-optimize <https://github.com/scikit-optimize/scikit-optimize>`__ which does **not** install with *datafold*.

* ``09_edmd_limitcycle.ipynb`` (`download <https://gitlab.com/datafold-dev/datafold/-/raw/master/tutorials/09_edmd_limitcycle.ipynb?inline=false>`__, `doc <https://datafold-dev.gitlab.io/datafold/tutorial_09_edmd_limitcycle.html>`__)
    We generate data from the Hopf system (an ODE system) and compare different dictionaries of the Extended Dynamic Mode Decomposition (``EDMD``). We also showcase out-of-sample predictions with a time horizon that exceeds the sampled time series in the training.
* ``10_dmd_control.ipynb`` (`download <https://gitlab.com/datafold-dev/datafold/-/raw/master/tutorials/10_dmd_control.ipynb?inline=false>`__, `doc <https://datafold-dev.gitlab.io/datafold/tutorial_10_dmd_control.html>`__)
    We introduce the dynamic mode decomposition with control. In this tutorial origins from the PyDMD package. Here we use it to compare the interface and highlight that the results are identical.

    **References**: `Original PyDMD tutorial <https://github.com/mathLab/PyDMD/blob/master/tutorials/tutorial7/tutorial-7-dmdc.ipynb>`__ :octicon:`link-external`
* ``11_edmd_control.ipynb`` (`download <https://gitlab.com/datafold-dev/datafold/-/raw/master/tutorials/11_edmd_control.ipynb?inline=false>`__, `doc <https://datafold-dev.gitlab.io/datafold/tutorial_11_edmd_control.html>`__)
    This tutorial demonstrates how to use extended dynamic mode decomposition (EDMD) and a linear quadratic regulator (LQR) for controlling the Van der Pol oscillator in a closed-loop. The goal is to show how EDMD can be an effective alternative for modeling and controlling non-linear dynamic systems.

    **References**: `Templated tutorial <https://github.com/i-abr/mpc-koopman/blob/master/mpc_with_koopman_op.ipynb>`__ :octicon:`link-external`
* ``12_kmpc_flowcontrol.ipynb`` (`download <https://gitlab.com/datafold-dev/datafold/-/raw/master/tutorials/12_kmpc_flowcontrol.ipynb?inline=false>`__, `doc <https://datafold-dev.gitlab.io/datafold/tutorial_12_kmpc_flowcontrol.html>`__)
    We take the 1D Burger equation with periodic boundary conditions as an example to showcase how the Koopman operator can be utilized for model predictive control (MPC) in flow systems.

    **References**: `Original code (Matlab) <https://github.com/arbabiha/KoopmanMPC_for_flowcontrol>`__ :octicon:`link-external` |  :cite:t:`arbabi-2018`
* ``13_kmpc_motor_engine.ipynb`` (`download <https://gitlab.com/datafold-dev/datafold/-/raw/master/tutorials/13_kmpc_motor_engine.ipynb?inline=false>`__, `doc <https://datafold-dev.gitlab.io/datafold/tutorial_13_kmpc_motor_engine.html>`__)
    This tutorial will demonstrate how to utilize the Extended Dynamic Mode Decomposition (EDMD) to estimate the Koopman operator in controlled dynamical systems. The nonlinear behavior of a motor engine model will be transformed into a higher dimensional space, which will result in an approximately linear evolution. This will allow the use of EDMD as a linearly controlled dynamical system within the Koopman Model Predictive Control (KMPC) framework.

    **References**: `Original code (Matlab) <https://github.com/MilanKorda/KoopmanMPC/>`__ :octicon:`link-external` |  :cite:t:`korda-2018` (Sect. 8.2)
* ``14_online_dmd.ipynb`` (`download <https://gitlab.com/datafold-dev/datafold/-/raw/master/tutorials/14_online_dmd.ipynb?inline=false>`__, `doc <https://datafold-dev.gitlab.io/datafold/tutorial_14_online_dmd.html>`__)
    This tutorial showcases the online dynamic mode decomposition (``OnlineDMD``) at the example of a simple 2D time-varying system. The performance of the online DMD is compared with batch DMD and the analytical solution of the system. Following the online update scheme the model is updated once new data becomes available, which is particularly useful in time-varying systems.

    **References**: `Original demo <https://github.com/haozhg/odmd/blob/master/demo/demo_online.ipynb>`__ :octicon:`link-external` |  :cite:t:`zhang-2019`




Run notebooks with Jupyter
--------------------------

Download files
^^^^^^^^^^^^^^

* **If datafold was installed from PyPI, ...**

  ... the tutorials are *not* included in the package. Download the tutorials separately from
  the list.

* **If the datafold repository was downloaded, ...**

  ... then the tutorials are located ``path/to/repo/tutorials/``. Before executing the
  tutorials, please make sure that *datafold* is either installed

  .. code-block:: bash

    python -m pip install .

  **or** that ``path/to/datafold/`` is included in the ``PYTHONPATH`` environment variable

  .. code-block:: bash

    export PYTHONPATH=$PYTHONPATH:/path/to/datafold/repository/


Start Jupyter
^^^^^^^^^^^^^

All tutorials are Jupyter notebooks (``.ipynb`` file ending). The Jupyter environment and its
dependencies install with

.. code-block:: bash

    python -m pip install jupyter

For further information visit the `Jupyter homepage <https://jupyter.org/>`__. To open a
Jupyter notebook in a web browser, run

.. code-block:: bash

    jupyter notebook path/to/datafold/tutorial_folder

or use the target in the Makefile (if the gir repository was downloaded):

.. code-block:: bash

    make tutorial
