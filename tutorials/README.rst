.. NOTE: this file was automatically generated with 'generate_tutorials_page.py' (located in 'datafold/doc/source/'). Navigate to this file, if you wish to change the content of this page.

.. _tutorialnb:

=========
Tutorials
=========

This page contains tutorials and code snippets to
showcase *datafold's* API. All tutorials can be viewed online below. If you want to
execute the notebooks in Jupyter, please also note the instructions in
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
    tutorial_10_online_dmd


* `01_datastructures.ipynb` (`download <https://gitlab.com/datafold-dev/datafold/-/raw/master/tutorials/01_datastructures.ipynb?inline=false>`__, `web <https://datafold-dev.gitlab.io/datafold/tutorial_01_datastructures.html>`__)
    We introduce *datafold*'s basic data structures for time series collection data and kernel-based algorithms. They are both used internally in model implementations and for input/output.
* `02_pcm_subsampling.ipynb` (`download <https://gitlab.com/datafold-dev/datafold/-/raw/master/tutorials/02_pcm_subsampling.ipynb?inline=false>`__, `web <https://datafold-dev.gitlab.io/datafold/tutorial_02_pcm_subsampling.html>`__)
    We show how the ``PCManifold`` data structure can be used to subsample a manifold point cloud uniformly.

    **Warning**
        The tutorial generates a large dataset with 10 Mio. samples by default. This may have to be reduced, depending on the available computer memory.

* `03_dmap_scurve.ipynb` (`download <https://gitlab.com/datafold-dev/datafold/-/raw/master/tutorials/03_dmap_scurve.ipynb?inline=false>`__, `web <https://datafold-dev.gitlab.io/datafold/tutorial_03_dmap_scurve.html>`__)
    We use a ``DiffusionMaps`` model to compute lower dimensional embeddings of an S-curved point cloud manifold. We also select the best combination of intrinsic parameters automatically with an optimization routine.
* `04_dmap_digitclustering.ipynb` (`download <https://gitlab.com/datafold-dev/datafold/-/raw/master/tutorials/04_dmap_digitclustering.ipynb?inline=false>`__, `web <https://datafold-dev.gitlab.io/datafold/tutorial_04_dmap_digitclustering.html>`__)
    We use the ``DiffusionMaps`` model to cluster data from handwritten digits and perform an out-of-sample embedding. This example is taken from the scikit-learn project and can be compared against other manifold learning algorithms.
* `05_roseland_scurve_digits.ipynb` (`download <https://gitlab.com/datafold-dev/datafold/-/raw/master/tutorials/05_roseland_scurve_digits.ipynb?inline=false>`__, `web <https://datafold-dev.gitlab.io/datafold/tutorial_05_roseland_scurve_digits.html>`__)
    We use a ``Roseland`` model to compute lower dimensional embeddings of an S-curved point cloud manifold and to cluster data from handwritten digit. We also select the best combination of intrinsic parameters automatically with an optimization routine and demonstrate how to do include this in an scikit-learn pipeline. Based on the Diffusion Maps tutorials.
* `06_dmap_mahalanobis_kernel.ipynb` (`download <https://gitlab.com/datafold-dev/datafold/-/raw/master/tutorials/06_dmap_mahalanobis_kernel.ipynb?inline=false>`__, `web <https://datafold-dev.gitlab.io/datafold/tutorial_06_dmap_mahalanobis_kernel.html>`__)
    We highlight how to use the Mahalanobis kernel within Diffusion Maps. With this we can obtain embeddings that are invariant to the observation function.

    **Warning**
        The implementation of the Mahalanobis kernel is still experimental and should be used with care. Contributions are welcome!

* `07_jsf_common_eigensystem.ipynb` (`download <https://gitlab.com/datafold-dev/datafold/-/raw/master/tutorials/07_jsf_common_eigensystem.ipynb?inline=false>`__, `web <https://datafold-dev.gitlab.io/datafold/tutorial_07_jsf_common_eigensystem.html>`__)
    We use ``JointlySmoothFunctions`` to learn commonly smooth functions from multimodal data. We also demonstrate the out-of-sample extension.
* `08_gh_oos.ipynb` (`download <https://gitlab.com/datafold-dev/datafold/-/raw/master/tutorials/08_gh_oos.ipynb?inline=false>`__, `web <https://datafold-dev.gitlab.io/datafold/tutorial_08_gh_oos.html>`__)
    We showcase the out-of-sample extension for manifold learning models such as the ``DiffusionMaps`` model. For this we use the ``GeometricHarmonicsInterpolator`` for forward and backwards interpolation.

    **Warning**
        The tutorial requires also the Python package `scikit-optimize <https://github.com/scikit-optimize/scikit-optimize>`__ which does not install with *datafold*.

* `09_edmd_limitcycle.ipynb` (`download <https://gitlab.com/datafold-dev/datafold/-/raw/master/tutorials/09_edmd_limitcycle.ipynb?inline=false>`__, `web <https://datafold-dev.gitlab.io/datafold/tutorial_09_edmd_limitcycle.html>`__)
    We generate data from a dynamical system (Hopf system) and compare different dictionaries of the Extended Dynamic Mode Decomposition (EDMD). We also evaluate out-of-sample predictions with time ranges exceeding the time horizon of the training data.
* `10_online_dmd.ipynb` (`download <https://gitlab.com/datafold-dev/datafold/-/raw/master/tutorials/10_online_dmd.ipynb?inline=false>`__, `web <https://datafold-dev.gitlab.io/datafold/tutorial_10_online_dmd.html>`__)
    We highlight ``OnlineDMD`` at the example of a simple system. The dynamic mode decomposition is updated once new data becomes available. This is particularly useful for time-varying systems. The notebook is taken from the original work by Zhang and Rowley, 2019; for reference see notebook.




Run notebooks with Jupyter
--------------------------

Download files
^^^^^^^^^^^^^^

* **If datafold was installed via PyPI, ...**

  ... the tutorials are *not* included in the package. Download them separately from the
  above list.

* **If the datafold repository was downloaded, ...**

  ... navigate to the folder ``/path/to/datafold/tutorials/``. Before executing the
  tutorials, please make sure that the package is either installed
  (:code:`python setup.py install`) or that ``path/to/datafold/`` is
  included in the `PYTHONPATH` environment variable
  (:code:`export PYTHONPATH=$PYTHONPATH:/path/to/datafold/`).

Start Jupyter
^^^^^^^^^^^^^

All tutorials are Jupyter notebooks (``.ipynb`` file ending). The Jupyter
package and dependencies install with

.. code-block:: bash

    python -m pip install jupyter

For further information visit the `Jupyter homepage <https://jupyter.org/>`__. To open a
Jupyter notebook in a web browser, run

.. code-block:: bash

    jupyter notebook path/to/datafold/tutorials
