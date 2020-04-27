#!/usr/bin/env python3

import os

PATH2DOC = os.path.abspath(".")
PATH2ROOT = os.path.abspath(os.path.join(".", "..", ".."))
PATH2TUTORIAL = os.path.abspath(os.path.join(PATH2ROOT, "tutorials"))

# in the same order they will be listed in the documentation
include_tutorial_filenames = [
    "basic_datastructures.ipynb",
    "basic_pcm_subsampling.ipynb",
    "basic_dmap_scurve.ipynb",
    "basic_dmap_digitclustering.ipynb",
    "basic_gh_oos.ipynb",
    "basic_edmd_limitcycle.ipynb",
]


def get_all_tutorial_files():
    abs_path_tutorial_files = []
    for file in include_tutorial_filenames:
        abs_filepath = os.path.join(PATH2TUTORIAL, file)

        if not os.path.exists(abs_filepath):
            raise FileNotFoundError(f"Tutorial file not found {abs_filepath}")

        assert abs_filepath.endswith(".ipynb")
        abs_path_tutorial_files.append(abs_filepath)
    return abs_path_tutorial_files


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
    tutorial_index_content += "\n"
    tutorial_index_content += ".. toctree::\n"
    tutorial_index_content += f"{ws}:maxdepth: 1\n"
    tutorial_index_content += "\n"

    prefix = "tutorial_"

    abs_path_tutorial_files = get_all_tutorial_files()

    for filepath in abs_path_tutorial_files:
        filename_tutorial = os.path.basename(filepath).replace(".ipynb", "")
        filename_nblink = f"{prefix}{filename_tutorial}"
        with open(f"{filename_nblink}.nblink", "w") as nblinkfile:
            nblinkfile.write(nblink_content.replace("??INSERT??", filepath))

        tutorial_index_content += f"{ws}{filename_nblink}\n"

    tutorial_index_content += "\n"

    # write content to rst file
    with open(tutorial_index_filename, "w") as indexfile:
        indexfile.write(tutorial_index_content)
