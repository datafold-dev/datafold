#!/usr/bin/env python3

import json
import pathlib
import warnings

import nbformat
import requests  # type: ignore
from nbconvert.preprocessors import ExecutePreprocessor

# path to current file location
PATH2DOCSOURCE = pathlib.Path(__file__).parent.resolve()
PATH2ROOT = PATH2DOCSOURCE.parent.parent.absolute()

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

or use the target in the Makefile (if the repository was downloaded):

.. code-block:: bash

    make tutorial

"""  # NOTE: make sure to include a single new line at the end

ALL_TUTORIALS = dict()
# prefix required as a file pattern in .gitignore (change also there!)
PREFIX_DOC_FILES = "tutorial_"
# Indentation to easier format .rst files
INDENT = "    "


def get_nblink(nb_filename) -> str:
    filename_tutorial = nb_filename.replace(".ipynb", "")
    return f"{PREFIX_DOC_FILES}{filename_tutorial}"


class Tutorial:
    FOLDER = PATH2ROOT / "tutorials"

    def __init__(
        self,
        filename,
        description,
        reference=None,
        warning=None,
        is_archive=False,
        nblink_kwargs=None,
    ):
        self.filename = filename
        self.description = description.rstrip()
        self.reference = reference
        self.warning = warning.rstrip() if warning is not None else None
        self.is_archive = is_archive
        self.nblink_kwargs = nblink_kwargs
        assert self.fullpath

    @property
    def fullpath(self) -> pathlib.Path:
        fullpath = Tutorial.FOLDER / self.filename
        if not fullpath.exists():
            raise FileNotFoundError(f"Tutorial '{fullpath=}' does not exist")

        if not fullpath.is_file() and not fullpath.suffix != ".ipynb":
            raise ValueError(
                "The path must point to a Jupster notebook file "
                f"(i.e. ending with '.ipynb'). Got {fullpath=}"
            )

        return fullpath

    @property
    def nblink_filename(self):
        return get_nblink(self.filename)

    @property
    def web_link(self):
        return f"{BASE_URL}/{self.nblink_filename}.html"

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
                f"tutorials/{self.filename}?inline=false"
            )

    @property
    def nblink(self):
        return get_nblink(self.filename)

    @property
    def name(self):
        return self.filename.replace(".ipynb", "")


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
            f"the tutorial will be published soon and that the link is correct.",
            stacklevel=3,
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
                "* ``{filename}`` (`download <{download_link}>`__, `doc <{web_link}>`__)\n",
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
            "filename": tutorial.filename,
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

            subs["reference"] = subs["reference"].rstrip(" | ")  # noqa: B005
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
        warning="The tutorial generates a large dataset with 10 Mio. samples. "
        "This number may have to be reduced when running locally, depending on the available "
        "computer memory.",
    )

    add_tutorial(
        "03_dmap_scurve.ipynb",
        "We use the ``DiffusionMaps`` model to compute lower dimensional embeddings of an "
        "S-curved point cloud manifold. We also select the best combination of intrinsic "
        "parameters automatically with an optimization routine.",
        reference={
            "scikit-learn tutorial": (
                "https://scikit-learn.org/stable/auto_examples/manifold/plot_compare_methods.html#sphx-glr-auto-examples-manifold-plot-compare-methods-py",
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
                "https://scikit-learn.org/stable/auto_examples/manifold/plot_compare_methods.html#sphx-glr-auto-examples-manifold-plot-compare-methods-py",
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
        warning="The implementation of Mahalanobis kernel in datafold is still experimental "
        "and should be used with care. Contributions (mainly testing / documentation) "
        "are welcome!",
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
        "``GeometricHarmonicsInterpolator`` for out-of-sample and "
        "``LaplacianPyramidsInterpolator`` for the pre-image mapping respectively.",
    )

    add_tutorial(
        "09_dmd_mixed_spatial_signals.ipynb",
        "We utilize dynamic mode decomposition (DMD) on a linear spatiotemporal system. This "
        "system is formed by combining two mixed spatiotemporal signals. The example is from "
        "the DMD book by Kutz et al.",
        reference={
            "pykoopman tutorial": (
                "https://github.com/dynamicslab/pykoopman/blob/master/docs/tutorial_dmd_separating_two_mixed_signals_400d_system.ipynb",
                "ext_link",
            ),
            "": (":cite:t:`kutz-2016`", "ref"),
        },
    )

    add_tutorial(
        "10_edmd_limitcycle.ipynb",
        "We generate data from the Hopf system (an ODE system) and compare different "
        "dictionaries of the Extended Dynamic Mode Decomposition (``EDMD``). We also showcase "
        "out-of-sample predictions with a time horizon that exceeds the sampled time series "
        "in the training.",
    )

    add_tutorial(
        "11_dmd_control.ipynb",
        "We introduce the dynamic mode decomposition with control. In this tutorial origins "
        "from the PyDMD package. Here we use it to compare the interface and highlight that "
        "the results are identical.",
        reference={
            "Original PyDMD tutorial": (
                "https://github.com/mathLab/PyDMD/blob/master/tutorials/tutorial7/tutorial-7-dmdc.ipynb",
                "ext_link",
            )
        },
    )

    add_tutorial(
        "12_edmd_control.ipynb",
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
        "13_EDMD_with_dictionary_learning.ipynb",
        "We demonstrate the EDMD-DL method, with which it is possible to learn the dictionary "
        "(i.e. set of observable functions) from the data. In the tutorial we use an "
        "feedforward artificial network and demonstrate the method for the Duffing system.",
        reference={"": (":cite:t:`li-2017`", "ref")},
        warning="The implementation of EDMD-DL is still experimental and should be used with "
        "care. Contributions (testing / documentation / enhanced functionality) are welcome! "
        "The notebook also requires the Python package `torch` to be installed separately to "
        "the datafold's dependencies",
    )

    add_tutorial(
        "14_kmpc_flowcontrol.ipynb",
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
        "15_kmpc_motor_engine.ipynb",
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
        "16_online_dmd.ipynb",
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
    for f in PATH2DOCSOURCE.glob("*.nblink"):
        f.unlink()

    tutorial_index_filename.unlink(missing_ok=True)


def generate_nblink_files():
    for tutorial in ALL_TUTORIALS.values():
        filename_nblink = get_nblink(tutorial.fullpath.name)

        data = {
            "path": str(tutorial.fullpath.as_posix()),
            **(tutorial.nblink_kwargs or {}),
        }

        with pathlib.Path(f"{filename_nblink}.nblink").open(mode="w") as nblinkfile:
            json.dump(data, nblinkfile)


def generate_docs_str(target):
    assert target in ["docs", "readme"]

    tutorial_page_content = (
        ".. NOTE: this file was automatically generated with "
        f"'{pathlib.Path(__file__)}'. Navigate to this file, if you wish to change the "
        "content of this page.\n\n"
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
        filename_nblink = get_nblink(tutorial.fullpath.name)

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
    tutorial_index_filename = pathlib.Path("tutorial_index.rst")

    # clean
    remove_existing_nblinks_and_indexfile(tutorial_index_filename)

    # generate links to Jupyter .ipynb files
    generate_nblink_files()

    # generate and write content to rst file
    tutorial_page_content_docs = generate_docs_str(target="docs")
    with tutorial_index_filename.open(mode="w") as indexfile:
        indexfile.write(tutorial_page_content_docs)

    # PART 2: README.rst file in tutorials

    # clean
    tutorial_readme_filename = Tutorial.FOLDER / "README.rst"
    tutorial_readme_filename.unlink(missing_ok=True)

    tutorial_page_content_readme = generate_docs_str(target="readme")

    # write content to rst file
    with tutorial_readme_filename.open(mode="w") as indexfile:
        indexfile.write(tutorial_page_content_readme)


def execute_tutorials(extra_arguments):
    print("###########################")
    print("### Executing notebooks ###")
    print("###########################")
    for tutorial in ALL_TUTORIALS.values():
        print(f"Executing tutorial `{tutorial.name}`")
        nbpath = tutorial.fullpath
        ex_path = nbpath.parent
        with nbpath.open(mode="r") as f:
            nb = nbformat.read(f, as_version=nbformat.NO_CONVERT)
            ep = ExecutePreprocessor(extra_arguments=extra_arguments)
            ep.preprocess(nb, {"metadata": {"path": ex_path}})

        with nbpath.open(mode="w", encoding="utf-8") as f:
            nbformat.write(nb, f)


if __name__ == "__main__":
    # init_tutorials()
    # execute_tutorials(None)
    setup_tutorials()
