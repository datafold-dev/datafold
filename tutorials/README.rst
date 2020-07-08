.. NOTE: this file was automatically generated with 'generate_tutorials_page.py' (located in 'datafold/doc/source/'). Navigate to this file, if you wish to change the content.

Tutorials
=========

This page contains tutorials and code snippets to showcase 
*datafold's* API. All tutorials can be viewed online or downloaded in the following list 
below. If you wish to execute the notebooks in Jupyter locally, follow the steps in the 
section below the tutorial list. 

List
----

`Download <https://gitlab.com/datafold-dev/datafold/-/archive/master/datafold-master.zip?path=tutorials/docs>`__ all tutorials in a zipped file.

.. toctree::
    :hidden:

    tutorial_01_basic_datastructures 
    tutorial_02_basic_pcm_subsampling 
    tutorial_03_basic_dmap_scurve 
    tutorial_04_basic_dmap_digitclustering 
    tutorial_05_basic_gh_oos 
    tutorial_06_basic_edmd_limitcycle 


* `01_basic_datastructures.ipynb` (`download <https://gitlab.com/datafold-dev/datafold/-/raw/master/tutorials/docs/01_basic_datastructures.ipynb?inline=false>`__ , `doc <https://datafold-dev.gitlab.io/datafold/tutorial_01_basic_datastructures.html>`__))
    We introduce the data structures that are implemented and used in *datafold*. The data structures are either used internally in model implementations, but can also be required as a data format to fit a model or be useful to estimate model parameters.
* `02_basic_pcm_subsampling.ipynb` (`download <https://gitlab.com/datafold-dev/datafold/-/raw/master/tutorials/docs/02_basic_pcm_subsampling.ipynb?inline=false>`__ , `doc <https://datafold-dev.gitlab.io/datafold/tutorial_02_basic_pcm_subsampling.html>`__))
    We show how the ``PCManifold`` data structure can be used to subsample a manifold point cloud uniformly.
 
    **Warning**
        The tutorial generates a large dataset with 10 Mio. samples by default. This may have to be reduced, depending on the available computer memory. 

* `03_basic_dmap_scurve.ipynb` (`download <https://gitlab.com/datafold-dev/datafold/-/raw/master/tutorials/docs/03_basic_dmap_scurve.ipynb?inline=false>`__ , `doc <https://datafold-dev.gitlab.io/datafold/tutorial_03_basic_dmap_scurve.html>`__))
    We use a ``DiffusionMaps`` model to compute possible lower dimensional embeddings of an S-curved point cloud manifold. Furthermore, we select the best combination of a new coordinates automatically with an optimization routine.
* `04_basic_dmap_digitclustering.ipynb` (`download <https://gitlab.com/datafold-dev/datafold/-/raw/master/tutorials/docs/04_basic_dmap_digitclustering.ipynb?inline=false>`__ , `doc <https://datafold-dev.gitlab.io/datafold/tutorial_04_basic_dmap_digitclustering.html>`__))
    We use the ``DiffusionMaps`` model to cluster handwritten digits and perform out-of-sample embeddings. The example is taken from the scikit-learn project and can be compared against the other manifold learning algorithms.
* `05_basic_gh_oos.ipynb` (`download <https://gitlab.com/datafold-dev/datafold/-/raw/master/tutorials/docs/05_basic_gh_oos.ipynb?inline=false>`__ , `doc <https://datafold-dev.gitlab.io/datafold/tutorial_05_basic_gh_oos.html>`__))
    We showcase the out-of-sample problem for manifold learning models such as the *DiffusionMaps* model. For this we use the *GeometricHarmonicsInterpolator* for forward and backwards interpolation mappings.
 
    **Warning**
        The tutorial requires also the Python package `scikit-optimize <https://github.com/scikit-optimize/scikit-optimize>`_ which does not install with *datafold*. 

* `06_basic_edmd_limitcycle.ipynb` (`download <https://gitlab.com/datafold-dev/datafold/-/raw/master/tutorials/docs/06_basic_edmd_limitcycle.ipynb?inline=false>`__ , `doc <https://datafold-dev.gitlab.io/datafold/tutorial_06_basic_edmd_limitcycle.html>`__))
    We generate data from a dynamical system (Hopf system) and compare different dictionaries of the 'Extended Dynamic Mode Decomposition'. We also perform out-of-sample predictions with time ranges exceeding the time horizon of the training data.




Run notebooks in Jupyter
------------------------

Download files
^^^^^^^^^^^^^^

* **If *datafold* was installed via PyPI, ...**
 
  the tutorials must be downloaded separately from the list above. An alternative way 
  is to naviagte to the 
  `tutorials repository page <https://gitlab.com/datafold-dev/datafold/-/tree/master/tutorials/docs>`__. 

   * To download all files, click the download button, select "Download this directory" 
     and extract the content of the compressed file.
   * To download a specific tutorial file, navigate to the file and click the download 
     button. 

* **If the datafold repository was cloned or downloaded, ...**
  
  navigate to the folder ``[root_repository]/tutorials/``. Before executing the 
  tutorials, please make sure that *datafold* is either installed 
  (:code:`python setup.py install`) or that the ``[root_repository]`` folder path is 
  included in the environment variable `PYTHONPATH` 
  (:code:`export PYTHONPATH=$PYTHONPATH:/path/to/datafold/repository/`). 

Start Jupyter
^^^^^^^^^^^^^

All tutorials are written as Jupyter notebooks (``.ipynb`` file ending). The Jupyter 
software installs with  

.. code-block:: bash

    pip install jupyter

For further information visit the `Jupyter homepage <https://jupyter.org/>`_. To open a 
Jupyter notebook in a web browser, run

.. code-block:: bash

    jupyter notebook path/to/tutorials/folder


