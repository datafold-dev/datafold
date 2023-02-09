#!/usr/bin/env python3

import json
import os
import pathlib
import warnings

import nbformat
import requests  # type: ignore
from nbconvert.preprocessors import ExecutePreprocessor

# path to current file location
PATH2DOCSOURCE = pathlib.Path(__file__).parent.resolve()
PATH2ROOT = PATH2DOCSOURCE.parent.parent

BASE_URL = "https://datafold-dev.gitlab.io/datafold"

rst_text_before_tutorials_list = """This page contains tutorials and code snippets to
showcase *datafold's* API. All tutorials can be viewed online. If you want to
execute the notebooks in Jupyter, please follow the instructions below in section
"Run notebooks with Jupyter".
"""

rst_text_after_tutorials_list = """

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

"""  # NOTE: make sure to include a single new line at the end

ALL_TUTORIALS = dict()
# prefix required as a file pattern in .gitignore (change also there!)
PREFIX_DOC_FILES = "tutorial_"
# Indentation to easier format .rst files
INDENT = "    "


def get_nblink(filename):
    filename_tutorial = os.path.basename(filename).replace(".ipynb", "")
    filename_nblink = f"{PREFIX_DOC_FILES}{filename_tutorial}"
    return filename_nblink


class Tutorial:
    FOLDER = os.path.abspath(os.path.join(PATH2ROOT, "tutorials"))

    def __init__(
        self,
        filename,
        description,
        reference=None,
        warning=None,
        is_archive=False,
        nblink_kwargs=None,
    ):
        self.tutorial_path = filename
        self.description = description.rstrip()
        self.reference = reference
        self.warning = warning.rstrip() if warning is not None else None
        self.is_archive = is_archive
        self.nblink_kwargs = nblink_kwargs
        assert self.fullpath

    @property
    def fullpath(self):
        fullpath = os.path.join(Tutorial.FOLDER, self.tutorial_path)
        if not os.path.exists(fullpath):
            raise FileNotFoundError(f"Tutorial '{fullpath=}' does not exist")

        if (
            not os.path.isfile(fullpath)
            and not os.path.splitext(fullpath)[1] != ".ipynb"
        ):
            raise ValueError(f"The path must point to a '.ipynb' file. Got {fullpath=}")

        return fullpath

    @property
    def relpath(self):
        return os.path.relpath(self.fullpath, ".")

    @property
    def nblink_filename(self):
        return get_nblink(self.tutorial_path)

    @property
    def web_link(self):
        nblink_filename = self.nblink_filename
        return f"{BASE_URL}/{nblink_filename}.html"

    @property
    def download_link(self):
        if self.is_archive:
            return (
                "https://gitlab.com/datafold-dev/datafold/-/archive/master/"
                f"datafold-master.zip?path=tutorials/{self.name}"
            )

        else:
            return (
                f"https://gitlab.com/datafold-dev/datafold/-/raw/master/"
                f"tutorials/{self.tutorial_path}?inline=false"
            )

    @property
    def nblink(self):
        return get_nblink(self.tutorial_path)

    @property
    def name(self):
        return os.path.splitext(os.path.basename(self.tutorial_path))[0]


def add_tutorial(
    filename,
    description,
    reference=None,
    warning=None,
    archive=False,
    nblink_kwargs=None,
):
    assert filename not in ALL_TUTORIALS

    tutorial = Tutorial(
        filename,
        description,
        reference=reference,
        warning=warning,
        is_archive=archive,
        nblink_kwargs=nblink_kwargs,
    )
    ALL_TUTORIALS[filename] = tutorial

    _req_download_file = requests.head(tutorial.download_link)
    if _req_download_file.status_code != 200:
        warnings.warn(
            f"The download link \n{tutorial.download_link} \n does not exist. Check if "
            f"the tutorial will be published soon and that the link is correct."
        )

    web_link = tutorial.web_link
    _req_weblink_doc = requests.head(web_link)
    if _req_weblink_doc.status_code != 200:
        print(
            f"\nWARNING: The web link \n{web_link} does not exist. Check if "
            f"the tutorial will be published soon and that the link is correct.\n"
        )


class TutorialStringBuilder:
    # fmt: off
    _templates = {
        "docs": {
            "download":
                "#. :doc:`{filename_nblink}` (:download:`download <{download_link}>`)\n",
            "reference": "\n\n{INDENT}**References**: {reference}",
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
                "* `{filename}` (`download <{download_link}>`__, `doc <{web_link}>`__)\n",
            "reference": "\n\n{INDENT}**References**: {reference}",
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
            "filename": tutorial.tutorial_path,
            "description": tutorial.description,
        }

        s = ""
        s += templates["download"]
        s += templates["description"]
        if tutorial.warning is not None:
            subs["warning"] = tutorial.warning
            s += templates["warning"]

        if tutorial.reference is not None:
            subs["reference"] = ""
            for k, v in tutorial.reference.items():
                if v[1] == "ext_link":
                    subs["reference"] += f"`{k} <{v[0]}>`__ :octicon:`link-external` | "
                elif v[1] == "ref":
                    subs["reference"] += f"{k} {v[0]} | "
                else:
                    raise ValueError("Reference key not understood")

            subs["reference"] = subs["reference"].rstrip(" | ")
            s += templates["reference"]

        s += "\n"
        return s.format(**subs)


def get_tutorial_text_doc(filename, target):
    tutorial = ALL_TUTORIALS[filename]
    return TutorialStringBuilder.build(target, tutorial)


def init_tutorials():
    add_tutorial(
        "01_datastructures.ipynb",
        "We introduce *datafold*'s basic data structures for time series collection data. The "
        "data structures are used in model implementations and for input/output specification "
        "of models.",
    )

    add_tutorial(
        "02_pcm_subsampling.ipynb",
        "We show how the ``PCManifold`` data structure can be used to subsample a "
        "manifold point cloud uniformly.",
        warning="The tutorial generates a large dataset with 10 Mio. samples by default. "
        "This may have to be reduced, depending on the available computer memory.",
    )

    add_tutorial(
        "03_dmap_scurve.ipynb",
        "We use the ``DiffusionMaps`` model to compute lower dimensional embeddings of an "
        "S-curved point cloud manifold. We also select the best combination of intrinsic "
        "parameters automatically with an optimization routine.",
        reference={
            "scikit-learn tutorial": (
                "https://scikit-learn.org/stable/auto_examples/manifold/plot_compare_methods.html#sphx-glr-auto-examples-manifold-plot-compare-methods-py",  # noqa: E501
                "ext_link",
            )
        },
    )

    add_tutorial(
        "04_dmap_digitclustering.ipynb",
        "We use the ``DiffusionMaps`` model to cluster data from handwritten digits and "
        "highlight its out-of-sample capabilities. This example is taken from the "
        "scikit-learn package, so the the method can be compared against other common "
        "manifold learning algorithms.",
        reference={
            "scikit-learn tutorial": (
                "https://scikit-learn.org/stable/auto_examples/manifold/plot_lle_digits.html",
                "ext_link",
            )
        },
    )

    add_tutorial(
        "05_roseland_scurve_digits.ipynb",
        "We use a ``Roseland`` model to compute lower dimensional embeddings of an "
        "S-curved point cloud manifold and to cluster data from handwritten digit.",
        reference={
            "scikit-learn tutorial 1": (
                "https://scikit-learn.org/stable/auto_examples/manifold/plot_compare_methods.html#sphx-glr-auto-examples-manifold-plot-compare-methods-py",  # noqa: E501
                "ext_link",
            ),
            "scikit-learn tutorial 2": (
                "https://scikit-learn.org/stable/auto_examples/manifold/plot_lle_digits.html",
                "ext_link",
            ),
        },
    )

    add_tutorial(
        "06_dmap_mahalanobis_kernel.ipynb",
        "We highlight how to use the Mahalanobis kernel within ``DiffusionMaps``. The key "
        "feature of this kernel is that it can yield embeddings that are invariant to the "
        "observation function.",
        warning="The implementation of the Mahalanobis kernel is still experimental and "
        "should be used with care. Contributions are welcome!",
    )

    add_tutorial(
        "07_jsf_common_eigensystem.ipynb",
        "We use ``JointlySmoothFunctions`` to learn commonly smooth functions "
        "from multimodal data. We also demonstrate the out-of-sample capabilities of the "
        "method.",
    )

    add_tutorial(
        "08_gh_oos.ipynb",
        "We showcase the out-of-sample extension for manifold learning "
        "models such as the ``DiffusionMaps`` model. For this we use the "
        "``GeometricHarmonicsInterpolator`` for forward and backwards interpolation.",
        warning="The tutorial requires also the Python package "
        "`scikit-optimize <https://github.com/scikit-optimize/scikit-optimize>`__ "
        "which does **not** install with *datafold*.",
    )

    add_tutorial(
        "09_edmd_limitcycle.ipynb",
        "We generate data from the Hopf system (an ODE system) and compare different "
        "dictionaries of the Extended Dynamic Mode Decomposition (``EDMD``). We also showcase "
        "out-of-sample predictions with a time horizon that exceeds the sampled time series "
        "in the training.",
    )

    add_tutorial(
        "10_dmd_control.ipynb",
        "We introduce the dynamic mode decomposition with control. In this tutorial origins "
        "from the PyDMD package. Here we use it to compare the interface and highlight that "
        "the results are identical.",
        reference={
            "Original PyDMD tutorial": (
                "https://github.com/mathLab/PyDMD/blob/master/tutorials/tutorial7/tutorial-7-dmdc.ipynb",  # noqa: E501
                "ext_link",
            )
        },
    )

    add_tutorial(
        "11_edmd_control.ipynb",
        "This tutorial demonstrates how to use extended dynamic mode decomposition (EDMD) and "
        "a linear quadratic regulator (LQR) for controlling the Van der Pol oscillator in a "
        "closed-loop. The goal is to show how EDMD can be an effective alternative for "
        "modeling and controlling non-linear dynamic systems.",
        reference={
            "Templated tutorial": (
                "https://github.com/i-abr/mpc-koopman/blob/master/mpc_with_koopman_op.ipynb",
                "ext_link",
            )
        },
    )

    add_tutorial(
        "12_kmpc_flowcontrol.ipynb",
        "We take the 1D Burger equation with periodic boundary conditions as an example to "
        "showcase how the Koopman operator can be utilized for model predictive control (MPC) "
        "in flow systems.",
        reference={
            "Original code (Matlab)": (
                "https://github.com/arbabiha/KoopmanMPC_for_flowcontrol",
                "ext_link",
            ),
            "": (":cite:t:`arbabi-2018`", "ref"),
        },
    )

    add_tutorial(
        "13_kmpc_motor_engine.ipynb",
        "This tutorial will demonstrate how to utilize the Extended Dynamic Mode "
        "Decomposition (EDMD) to estimate the Koopman operator in controlled dynamical "
        "systems. The nonlinear behavior of a motor engine model will be transformed into "
        "a higher dimensional space, which will result in an approximately linear evolution. "
        "This will allow the use of EDMD as a linearly controlled dynamical system within the "
        "Koopman Model Predictive Control (KMPC) framework.",
        reference={
            "Original code (Matlab)": (
                "https://github.com/MilanKorda/KoopmanMPC/",
                "ext_link",
            ),
            "": (":cite:t:`korda-2018` (Sect. 8.2)", "ref"),
        },
    )

    add_tutorial(
        "14_online_dmd.ipynb",
        "This tutorial showcases the online dynamic mode decomposition (``OnlineDMD``) at the "
        "example of a simple 2D time-varying system. The performance of the online DMD is "
        "compared with batch DMD and the analytical solution of the system. Following the "
        "online update scheme the model is updated once new data becomes available, which is "
        "particularly useful in time-varying systems.",
        reference={
            "Original demo": (
                "https://github.com/haozhg/odmd/blob/master/demo/demo_online.ipynb",
                "ext_link",
            ),
            "": (":cite:t:`zhang-2019`", "ref"),
        },
    )
    #  The notebook is taken from the original work by "Zhang and Rowley, 2019; see the
    #  notebook for reference.

    # add_tutorial(
    #     "11_koopman_mpc/koopman_mpc.ipynb",
    #     "Walkthrough for doing Model Predictive Control (MPC) based on the Koopman "
    #     "operator. We apply MPC using an EDMD predictor to a toy model: the "
    #     "inverted pendulum, sometimes referred to as a cartpole.",
    #     archive=True,
    #     nblink_kwargs={
    #         "extra-media": [f"{Tutorial.FOLDER}/11_koopman_mpc/cartpole2.png"]
    #     },
    # )


def remove_existing_nblinks_and_indexfile(tutorial_index_filename):
    for file in os.listdir(PATH2DOCSOURCE):
        if file.endswith(".nblink"):
            os.remove(file)
    try:
        os.remove(tutorial_index_filename)
    except FileNotFoundError:
        pass  # don't worry


def generate_nblink_files():
    for tutorial in ALL_TUTORIALS.values():
        filename_nblink = get_nblink(tutorial.fullpath)

        data = {
            "path": os.path.normpath(tutorial.fullpath).replace("\\", "/"),
            **(tutorial.nblink_kwargs or {}),
        }

        fname = f"{filename_nblink}.nblink"
        with open(fname, "w") as nblinkfile:
            json.dump(data, nblinkfile)


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

    for filename, tutorial in ALL_TUTORIALS.items():
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

    # generate links to Jupyter .ipynb files
    generate_nblink_files()

    # generate and write content to rst file
    tutorial_page_content_docs = generate_docs_str(target="docs")
    with open(tutorial_index_filename, "w") as indexfile:
        indexfile.write(tutorial_page_content_docs)

    # PART 2: README.rst file in tutorials

    # clean
    tutorial_readme_filename = os.path.join(Tutorial.FOLDER, "README.rst")
    try:
        os.remove(tutorial_readme_filename)
    except FileNotFoundError:
        pass  # don't worry

    tutorial_page_content_readme = generate_docs_str(target="readme")

    # write content to rst file
    with open(tutorial_readme_filename, "w") as indexfile:
        indexfile.write(tutorial_page_content_readme)


def execute_tutorials(extra_arguments):
    print("###########################")
    print("### Executing notebooks ###")
    print("###########################")
    for tutorial in ALL_TUTORIALS.values():
        print(f"Executing tutorial `{tutorial.name}`")
        nbpath = tutorial.fullpath
        ex_path = os.path.dirname(nbpath)
        with open(nbpath, "r") as f:
            nb = nbformat.read(f, as_version=nbformat.NO_CONVERT)
            ep = ExecutePreprocessor(extra_arguments=extra_arguments)
            ep.preprocess(nb, {"metadata": {"path": ex_path}})

        with open(nbpath, "w", encoding="utf-8") as f:
            nbformat.write(nb, f)


if __name__ == "__main__":
    setup_tutorials()
