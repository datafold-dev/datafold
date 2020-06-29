#!/usr/bin/env python3

import glob
import os

PATH2DOC = os.path.abspath(".")
PATH2ROOT = os.path.abspath(os.path.join(".", "..", ".."))
PATH2TUTORIAL = os.path.abspath(os.path.join(PATH2ROOT, "tutorials"))
PATH2TUTORIAL_DOC = os.path.abspath(os.path.join(PATH2TUTORIAL, "docs"))

rst_before_tutorials_list = """This page contains tutorials and code snippets to showcase 
*datafold's* API. All tutorials can be viewed online in the list below. If you wish to 
execute the notebooks in a Jupyter notebook locally, follow the steps in "Execute 
interactively" below. 
"""

rst_after_tutorials_list = """

Warnings if executing the files locally 
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. warning::
     The tutorial "Geometric Harmonics: interpolate function values on data manifold" (
     file `05_basic_gh_oos.ipynb`) uses also the Python package 
     `scikit-optimize <https://github.com/scikit-optimize/scikit-optimize>`_ which does 
     not install with *datafold*.
 
.. warning::
    The tutorial "Uniform subsampling of point cloud manifold" 
    (file `02_basic_pcm_subsampling.ipynb`) generates a large dataset with 10 
    Mio. samples. This number may have to be reduced, depending on the 
    available computer memory.

Execute interactively
---------------------

Download files
^^^^^^^^^^^^^^

* **If *datafold* was installed via PyPI, ...**
 
  the tutorials must be downloaded separately from the 
  `tutorials repository page <https://gitlab.com/datafold-dev/datafold/-/tree/master/tutorials/docs>`_. 

   * To download all files, click the download button, select "Download this directory" 
     and extract the content of the compressed file.
   * To download a specific tutorial file, navigate to the file and click the download 
     button. 

* **If the datafold repository was cloned or downloaded, ...**
  
  navigate to the folder ``[root_repository]/tutorials/docs``. Before executing the 
  tutorials, please make sure that *datafold* is either installed 
  (:code:`python setup.py install`) or that the ``[root_repository]`` folder path is 
  included in the environment variable `PYTHONPATH` 
  (:code:`export PYTHONPATH=$PYTHONPATH:/path/to/datafold/repository`).

Execute notebooks
^^^^^^^^^^^^^^^^^

All tutorials are written as Jupyter notebooks (``.ipynb`` file ending). The Jupyter 
software installs with  

.. code-block:: bash

    pip install jupyter

For further information visit the `Jupyter homepage <https://jupyter.org/>`_. To open a 
Jupyter notebook in a web browser, run

.. code-block:: bash

    jupyter notebook path/to/tutorials/folder
"""


def remove_existing_nblinks_and_indexfile(tutorial_index_filename):
    for file in os.listdir(PATH2DOC):
        if file.endswith(".nblink"):
            os.remove(file)
    try:
        os.remove(tutorial_index_filename)
    except FileNotFoundError:
        pass  # don't worry


def setup_tutorials():
    tutorial_index_filename = "tutorial_index.rst"
    remove_existing_nblinks_and_indexfile(tutorial_index_filename)

    nblink_content = """
    {
    "path": "??INSERT??"
    }
    """

    ws = "    "
    tutorial_index_content = "Tutorials\n"
    tutorial_index_content += "=========\n"
    tutorial_index_content += "\n"
    tutorial_index_content += rst_before_tutorials_list
    tutorial_index_content += "\n"
    tutorial_index_content += "Online list\n"
    tutorial_index_content += "-----------\n"
    tutorial_index_content += "\n"
    tutorial_index_content += ".. toctree::\n"
    tutorial_index_content += f"{ws}:maxdepth: 1\n"
    tutorial_index_content += "???INSERT_FILES???\n"
    tutorial_index_content += "\n"
    tutorial_index_content += rst_after_tutorials_list

    prefix = "tutorial_"

    abs_path_tutorial_files = sorted(
        glob.glob(os.path.join(PATH2TUTORIAL_DOC, "*.ipynb"))
    )

    files_list_str = "\n"  # generate string to insert in tutorial_index_content

    for filepath in abs_path_tutorial_files:
        filename_tutorial = os.path.basename(filepath).replace(".ipynb", "")
        filename_nblink = f"{prefix}{filename_tutorial}"
        with open(f"{filename_nblink}.nblink", "w") as nblinkfile:
            nblinkfile.write(
                nblink_content.replace(
                    "??INSERT??", os.path.normpath(filepath).replace("\\", "/")
                )
            )

        files_list_str += f"{ws}{filename_nblink}\n"

    tutorial_index_content = tutorial_index_content.replace(
        "???INSERT_FILES???", files_list_str
    )

    # write content to rst file
    with open(tutorial_index_filename, "w") as indexfile:
        indexfile.write(tutorial_index_content)
