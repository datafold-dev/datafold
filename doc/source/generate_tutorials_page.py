#!/usr/bin/env python3

import json
import nbformat
import os
import pathlib
import shutil
import subprocess
import warnings

from nbconvert.preprocessors import ExecutePreprocessor

import requests  # type: ignore

# path to current file location
PATH2DOCSOURCE = pathlib.Path(__file__).parent.resolve()
PATH2ROOT = PATH2DOCSOURCE.parent.parent
PATH2TUTORIAL = PATH2ROOT.joinpath("tutorials")

BASE_URL = "https://datafold-dev.gitlab.io/datafold"

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


class Tutorial:
    def __init__(self, filename, description, **kwargs):
        warning = kwargs.get("warning")
        if warning is not None:
            warning = warning.rstrip()

        self.filename = filename
        self.description = description.rstrip()
        self.warning = warning
        self.archive = kwargs.get("archive", False)

        self.nblink_args = kwargs.get('nblink_args')

        assert self.fullpath

    @property
    def fullpath(self):
        return os.path.join(PATH2TUTORIAL, self.filename)

    @property
    def relpath(self):
        return os.path.relpath(self.fullpath, ".")

    @property
    def nblink_filename(self):
        return get_nblink(self.filename)

    @property
    def web_link(self):
        nblink_filename = self.nblink_filename
        return f"{BASE_URL}/{nblink_filename}.html"

    @property
    def download_link(self):
        filename = self.filename
        name = self.name

        if self.archive:
            return (
                "https://gitlab.com/datafold-dev/datafold/-/raw/master/"
                f"tutorial-{name}.zip?path=tutorials/{name}/"
            )
        else:
            return (
                f"https://gitlab.com/datafold-dev/datafold/-/raw/master/"
                f"tutorials/{filename}?inline=false"
            )

    @property
    def download_path(self):
        if self.archive:
            return self.archive_path
        else:
            return self.relpath

    @property
    def nblink(self):
        return get_nblink(self.filename)

    @property
    def name(self):
        return os.path.splitext(os.path.basename(self.filename))[0]

    @property
    def archive_path(self):
        if self.archive:
            return f"{self.name}.zip"
        return None


def add_tutorial(filename, description, warning=None, archive=False,
                 nblink_args=None):
    assert filename not in DESCRIPTIVE_TUTORIALS

    tutorial = Tutorial(filename, description, warning=warning,
                        archive=archive, nblink_args=nblink_args)
    DESCRIPTIVE_TUTORIALS[filename] = tutorial

    fullpath = tutorial.fullpath
    if not os.path.exists(fullpath):
        raise FileNotFoundError(
            f"The filepath {os.path.join(PATH2TUTORIAL, filename)} does not exist."
        )

    download_link = tutorial.download_link
    _req_download_file = requests.head(download_link)
    if _req_download_file.status_code != 200:
        warnings.warn(
            f"The download link \n{download_link} \n does not exist. Check if "
            f"the tutorial will be published soon and that the link is correct."
        )

    web_link = tutorial.web_link
    _req_weblink_doc = requests.head(web_link)
    if _req_weblink_doc.status_code != 200:
        print(
            f"WARNING: The web link \n{web_link} does not exist. Check if "
            f"the tutorial will be published soon and that the link is correct."
        )


class TutorialStringBuilder:
    # fmt: off
    _templates = {
        "docs": {
            "download":
                "#. :doc:`{filename_nblink}` (:download:`download <{download_path}>`)\n",
            "warning":
                "\n\n"
                "{INDENT}.. warning::\n"
                "{INDENT}{INDENT}{warning}\n",
            "description":
                "{INDENT}{description}",
        },
        "readme": {
            # "filename (download_link, doc_link)" in readme
            "download":
                "* `{filename}` (`download <{download_link}>`__ , `doc <{web_link}>`__)\n",
            "warning":
                "\n\n"
                "{INDENT}**Warning**\n"
                "{INDENT}{INDENT}{warning}\n",
            "description":
                "{INDENT}{description}",
        },
    }
    # fmt: on

    @classmethod
    def build(cls, target, tutorial: Tutorial):
        if target not in cls._templates:
            raise ValueError(f"'target={target}' not known")

        templates = cls._templates[target]

        subs = {
            "INDENT": INDENT,
            "web_link": tutorial.web_link,
            "filename_nblink": tutorial.nblink,
            "download_link": tutorial.download_link,
            "download_path": tutorial.download_path,
            "filename": tutorial.filename,
            "description": tutorial.description,
        }

        s = ""
        s += templates["download"]
        s += templates["description"]
        if tutorial.warning is not None:
            subs["warning"] = tutorial.warning
            s += templates["warning"]
        s += "\n"

        return s.format(**subs)


def get_tutorial_text_doc(filename, target):
    tutorial = DESCRIPTIVE_TUTORIALS[filename]
    return TutorialStringBuilder.build(target, tutorial)


def init_tutorials():
    add_tutorial(
        "01_basic_datastructures.ipynb",
        "We introduce *datafold*'s basic data structures for time series collection data and "
        "kernel-based algorithms. They are both used internally in model implementations and "
        "for input/output.",
    )

    add_tutorial(
        "02_basic_pcm_subsampling.ipynb",
        "We show how the ``PCManifold`` data structure can be used to subsample a "
        "manifold point cloud uniformly.",
        warning="The tutorial generates a large dataset with 10 Mio. samples by default. "
        "This may have to be reduced, depending on the available computer memory.",
    )

    add_tutorial(
        "03_basic_dmap_scurve.ipynb",
        "We use a ``DiffusionMaps`` model to compute lower dimensional embeddings of an "
        "S-curved point cloud manifold. We also select the best combination of intrinsic "
        "parameters automatically with an optimization routine.",
    )

    add_tutorial(
        "04_basic_dmap_digitclustering.ipynb",
        "We use the ``DiffusionMaps`` model to cluster data from handwritten digits and "
        "perform an out-of-sample embedding. This example is taken from the scikit-learn "
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
        "06_basic_edmd_limitcycle.ipynb",
        "We generate data from a dynamical system (Hopf system) and compare different "
        "dictionaries of the Extended Dynamic Mode Decomposition (EDMD). We also evaluate "
        "out-of-sample predictions with time ranges exceeding the time horizon of the "
        "training data.",
    )

    add_tutorial(
        filename="07_basic_jsf_common_eigensystem.ipynb",
        description="We use ``JointlySmoothFunctions`` to learn commonly smooth functions "
        "from multimodal data. Also, we introduce ``JsfDataset``, which is used to make "
        "``JointlySmoothFunctions`` consistent with scikit-learn's estimator and transformer "
        "APIs. Finally, we demonstrate the out-of-sample extension.",
        warning="The code for jointly smooth functions inside this notebook is experimental.",
    )

    add_tutorial(
        "08_basic_roseland_scurve_digits.ipynb",
        "We use a ``Roseland`` model to compute lower dimensional embeddings of an "
        "S-curved point cloud manifold and to cluster data from handwritten digit. "
        "We also select the best combination of intrinsic parameters automatically "
        "with an optimization routine and demonstrate how to do include this in an "
        "scikit-learn pipeline. Based on the Diffusion Maps tutorials.",
    )

    add_tutorial(
        "10_koopman_mpc/10_koopman_mpc.ipynb",
        "Walkthrough for doing Model Predictive Control (MPC) based on the Koopman "
        "operator. We apply MPC using an EDMD predictor to a toy model: the "
        "inverted pendulum, sometimes referred to as a cartpole.",
        archive=True,
        nblink_args={
            'extra-media': [f'{PATH2TUTORIAL}/10_koopman_mpc/cartpole2.png']
        }
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
    for tutorial in DESCRIPTIVE_TUTORIALS.values():
        filepath = tutorial.fullpath
        filename_nblink = get_nblink(filepath)

        extras = {}
        if tutorial.nblink_args is not None:
            extras.update(tutorial.nblink_args)

        data = {
            "path": os.path.normpath(filepath).replace("\\", "/"),
            **extras
        }
        fname = f"{filename_nblink}.nblink"
        with open(fname, "w") as nblinkfile:
            json.dump(data, nblinkfile)


def generate_tutorial_archives():
    for tutorial in DESCRIPTIVE_TUTORIALS.values():
        if tutorial.archive is True:
            path = os.path.dirname(tutorial.fullpath)
            archive_path = tutorial.archive_path
            archive_name = os.path.splitext(os.path.basename(archive_path))[0]

            root_dir = os.path.dirname(path)
            base_dir = os.path.basename(path)

            # Delete transient files like __pycache__ and .ipynb_checkpoints
            # before creating the tutorial archive
            cmd1 = ['find', root_dir, '-name', '__pycache__',
                    '-or', '-name', '.ipynb_checkpoints']
            cmd2 = ['xargs', '-r', 'rm', '-R']
            p1 = subprocess.Popen(cmd1, stdout=subprocess.PIPE)
            p2 = subprocess.run(cmd2, stdin=p1.stdout, check=True)
            p1.stdout.close()

            archive_path_ = shutil.make_archive(archive_name, "zip", root_dir, base_dir)
            assert archive_path_ == os.path.abspath(archive_path)


def execute_tutorials(extra_arguments):
    print('###########################')
    print('### Executing notebooks ###')
    print('###########################')
    for tutorial in DESCRIPTIVE_TUTORIALS.values():
        print('Executing tutorial `{}`'.format(tutorial.name))
        nbpath = tutorial.fullpath
        ex_path = os.path.dirname(nbpath)
        with open(nbpath, 'r') as f:
            nb = nbformat.read(f, as_version=4)
            ep = ExecutePreprocessor(extra_arguments=extra_arguments)
            ep.preprocess(nb, {'metadata': {'path': ex_path}})

        with open(nbpath, 'w', encoding='utf-8') as f:
            nbformat.write(nb, f)


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

    tutorials_list = "\n"  # generate string to insert in tutorial_page_content
    files_list = "\n"  # generate string to insert in tutorial_page_content

    for filename, tutorial in DESCRIPTIVE_TUTORIALS.items():
        filepath = tutorial.fullpath
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
    # Initialize list of tutorials
    init_tutorials()

    # PART 1: Online documentation
    tutorial_index_filename = "tutorial_index.rst"

    # clean
    remove_existing_nblinks_and_indexfile(tutorial_index_filename)

    # generate links to Jupyter files
    generate_nblink_files()

    # generate archives for certain jupyter notebooks
    generate_tutorial_archives()

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
