#!/usr/bin/env python3

import glob
import os
import pathlib
import warnings

import requests  # type: ignore

# path to current file location
PATH2DOCSOURCE = pathlib.Path(__file__).parent.resolve()
PATH2ROOT = PATH2DOCSOURCE.parent.parent
PATH2TUTORIAL = PATH2ROOT.joinpath("tutorials")

rst_text_before_tutorials_list = """This page contains tutorials and code snippets to
showcase *datafold's* API. All tutorials can be viewed online below. If you want to
execute the notebooks in Jupyter, please also note the instructions in
"Run notebooks with Jupyter".
"""

rst_text_after_tutorials_list = """

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
"""

DESCRIPTIVE_TUTORIALS = dict()
# prefix required as a file pattern in .gitignore (change also there!)
PREFIX_DOC_FILES = "tutorial_"
# Indentation for to easier format .rst file
INDENT = "    "


def get_nblink(filename):
    filename_tutorial = os.path.basename(filename).replace(".ipynb", "")
    filename_nblink = f"{PREFIX_DOC_FILES}{filename_tutorial}"
    return filename_nblink


def add_tutorial(filename, description, warning=None):
    assert filename not in DESCRIPTIVE_TUTORIALS

    if not os.path.exists(os.path.join(PATH2TUTORIAL, filename)):
        raise FileNotFoundError(
            f"The filepath {os.path.join(PATH2TUTORIAL, filename)} does not exist."
        )

    download_link = (
        f"https://gitlab.com/datafold-dev/datafold/-/raw/master/"
        f"tutorials/{filename}?inline=false"
    )

    nblink_filename = get_nblink(filename)
    web_link = f"https://datafold-dev.gitlab.io/datafold/{nblink_filename}.html"

    DESCRIPTIVE_TUTORIALS[filename] = dict(
        description=description.rstrip(),
        download_link=download_link.rstrip(),
        web_link=web_link.rstrip(),
        warning=warning.rstrip() if isinstance(warning, str) else None,
    )

    _req_download_file = requests.head(download_link)
    if _req_download_file.status_code != 200:
        warnings.warn(
            f"The download link \n{download_link} \n does not exist. Check if "
            f"the tutorial will be published soon and that the link is correct."
        )

    _req_weblink_doc = requests.head(web_link)
    if _req_weblink_doc.status_code != 200:
        print(
            f"WARNING: The web link \n{web_link} does not exist. Check if "
            f"the tutorial will be published soon and that the link is correct."
        )


def get_tutorial_text_doc(filename, target):

    filename_nblink = get_nblink(filename)
    _dict = DESCRIPTIVE_TUTORIALS[filename]

    # TODO: make more readable code by using string replacements
    if target == "docs":
        # "page_nblink (download_link)" in docs
        _str = (
            f"#. :doc:`{filename_nblink}` (`download <{_dict['download_link']}>`__)\n"
        )
        _str += f"{INDENT}{_dict['description']}"

        if _dict["warning"] is not None:
            _str += "\n\n"
            _str += f"{INDENT}.. warning::\n"
            _str += f"{INDENT}{INDENT}{_dict['warning']}\n"

    elif target == "readme":
        # "filename (download_link, doc_link)" in readme
        _str = (
            f"* `{filename}` (`download <{_dict['download_link']}>`__, "
            f"`web <{_dict['web_link']}>`__)\n"
        )
        _str += f"{INDENT}{_dict['description']}"

        if _dict["warning"] is not None:
            _str += "\n\n"
            _str += f"{INDENT}**Warning**\n"
            _str += f"{INDENT}{INDENT}{_dict['warning']}\n"
    else:
        raise ValueError(f"'target={target}' not known")

    _str += "\n"
    return _str


add_tutorial(
    filename="01_basic_datastructures.ipynb",
    description="We introduce *datafold*'s basic data structures for time series collection "
    "data and kernel-based algorithms. They are both used internally in model implementations "
    "and for input/output.",
)

add_tutorial(
    filename="02_basic_pcm_subsampling.ipynb",
    description="We show how the ``PCManifold`` data structure can be used to subsample a "
    "manifold point cloud uniformly.",
    warning="The tutorial generates a large dataset with 10 Mio. samples by default. "
    "This may have to be reduced, depending on the available computer memory.",
)

add_tutorial(
    filename="03_basic_dmap_scurve.ipynb",
    description="We use a ``DiffusionMaps`` model to compute lower dimensional embeddings of "
    "an S-curved point cloud manifold. We also select the best combination of intrinsic "
    "parameters automatically with an optimization routine.",
)

add_tutorial(
    filename="04_basic_dmap_digitclustering.ipynb",
    description="We use the ``DiffusionMaps`` model to cluster data from handwritten digits "
    "and perform an out-of-sample embedding. This example is taken from the scikit-learn "
    "project and can be compared against other manifold learning algorithms.",
)

add_tutorial(
    filename="05_basic_gh_oos.ipynb",
    description="We showcase the out-of-sample extension for manifold learning "
    "models such as the ``DiffusionMaps`` model. For this we use the "
    "``GeometricHarmonicsInterpolator`` for forward and backwards interpolation.",
    warning="The tutorial requires also the Python package "
    "`scikit-optimize <https://github.com/scikit-optimize/scikit-optimize>`__ "
    "which does not install with *datafold*.",
)

add_tutorial(
    filename="06_basic_edmd_limitcycle.ipynb",
    description="We generate data from a dynamical system (Hopf system) and compare different "
    "dictionaries of the Extended Dynamic Mode Decomposition (EDMD). We also evaluate "
    "out-of-sample predictions with time ranges exceeding the time horizon of the "
    "training data.",
)

add_tutorial(
    filename="06_basic_edmd_limitcycle.ipynb",
    description="We generate data from a dynamical system (Hopf system) and compare different "
    "dictionaries of the Extended Dynamic Mode Decomposition (EDMD). We also evaluate "
    "out-of-sample predictions with time ranges exceeding the time horizon of the "
    "training data.",
)

add_tutorial(
    filename="07_online_dmd.ipynb",
    description="We highlight an example of ``OnlineDMD`` where the dynamic mode "
    "decomposition is updated once new data becomes available. This is particularly useful "
    "for time-varying systems. The notebook is taken from the original work by Zhang and "
    "Rowley, 2019; for reference see notebook.",
)

add_tutorial(
    filename="08_basic_jsf_common_eigensystem.ipynb",
    description="We use ``JointlySmoothFunctions`` to learn commonly smooth functions "
    "from multimodal data. We also demonstrate the out-of-sample extension.",
)

add_tutorial(
    filename="09_basic_roseland_scurve_digits.ipynb",
    description="We use a ``Roseland`` model to compute lower dimensional embeddings of an "
    "S-curved point cloud manifold and to cluster data from handwritten digit. "
    "We also select the best combination of intrinsic parameters automatically "
    "with an optimization routine and demonstrate how to do include this in an "
    "scikit-learn pipeline. Based on the Diffusion Maps tutorials.",
)


def remove_existing_nblinks_and_indexfile(tutorial_index_filename):
    for file in os.listdir(PATH2DOCSOURCE):
        if file.endswith(".nblink"):
            os.remove(file)
    try:
        os.remove(tutorial_index_filename)
    except FileNotFoundError:
        pass  # don't worry


def generate_nblink_files():

    nblink_content = """
    {
    "path": "??INSERT??"
    }
    """

    abs_path_tutorial_files = sorted(glob.glob(os.path.join(PATH2TUTORIAL, "*.ipynb")))

    for filepath in abs_path_tutorial_files:
        filename_nblink = get_nblink(filepath)

        with open(f"{filename_nblink}.nblink", "w") as nblinkfile:
            nblinkfile.write(
                nblink_content.replace(
                    "??INSERT??", os.path.normpath(filepath).replace("\\", "/")
                )
            )


def generate_docs_str(target):

    assert target in ["docs", "readme"]

    tutorial_page_content = (
        f".. NOTE: this file was automatically generated with "
        f"'{os.path.basename(__file__)}' (located in 'datafold/doc/source/'). Navigate "
        f"to this file, if you wish to change the content of this page.\n\n"
    )

    tutorial_page_content += ".. _tutorialnb:\n"
    tutorial_page_content += "\n"
    tutorial_page_content += "=========\n"
    tutorial_page_content += "Tutorials\n"
    tutorial_page_content += "=========\n"
    tutorial_page_content += "\n"
    tutorial_page_content += rst_text_before_tutorials_list
    tutorial_page_content += "\n"
    tutorial_page_content += "List\n"
    tutorial_page_content += "----\n"
    tutorial_page_content += "\n"
    tutorial_page_content += (
        "`Download "
        "<https://gitlab.com/datafold-dev/datafold/-/archive/master/datafold-master."
        "zip?path=tutorials/>`__ "
        "all tutorials in a zipped file.\n\n"
    )
    tutorial_page_content += ".. toctree::\n"
    tutorial_page_content += f"{INDENT}:hidden:\n"

    # use easy replacement strings
    tutorial_page_content += "???INSERT_TOC_FILELIST???\n"
    tutorial_page_content += "???INSERT_TUTORIAL_LIST???\n"
    tutorial_page_content += "\n"
    tutorial_page_content += rst_text_after_tutorials_list

    abs_path_tutorial_files = sorted(glob.glob(os.path.join(PATH2TUTORIAL, "*.ipynb")))

    tutorials_list = "\n"  # generate string to insert in tutorial_page_content
    files_list = "\n"  # generate string to insert in tutorial_page_content

    for filepath in abs_path_tutorial_files:
        filename = os.path.basename(filepath)
        filename_nblink = get_nblink(filepath)

        files_list += f"{INDENT}{filename_nblink}\n"
        tutorials_list += get_tutorial_text_doc(filename, target=target)

    tutorial_page_content = tutorial_page_content.replace(
        "???INSERT_TOC_FILELIST???", files_list
    )

    tutorial_page_content = tutorial_page_content.replace(
        "???INSERT_TUTORIAL_LIST???", tutorials_list
    )
    return tutorial_page_content


def setup_tutorials():

    # PART 1: Online documentation
    tutorial_index_filename = "tutorial_index.rst"

    # clean
    remove_existing_nblinks_and_indexfile(tutorial_index_filename)

    # generate links to Jupyter files
    generate_nblink_files()

    # generate and write content to rst file
    tutorial_page_content_docs = generate_docs_str(target="docs")
    with open(tutorial_index_filename, "w") as indexfile:
        indexfile.write(tutorial_page_content_docs)

    # PART 2: README.rst file in tutorials

    # clean
    tutorial_readme_filename = os.path.join(PATH2TUTORIAL, "README.rst")
    try:
        os.remove(tutorial_readme_filename)
    except FileNotFoundError:
        pass  # don't worry

    tutorial_page_content_readme = generate_docs_str(target="readme")

    # write content to rst file
    with open(tutorial_readme_filename, "w") as indexfile:
        indexfile.write(tutorial_page_content_readme)
