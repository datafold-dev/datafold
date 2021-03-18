import os
import subprocess
import tempfile
import unittest
from typing import List

import nbformat
import pytest

IGNORE_NOTEBOOKS: List[str] = []


def _notebook_run(path):
    """Execute a notebook via nbconvert and collect output.
    returns the parsed notebook object and the execution errors.

    Source:
    https://blog.thedataincubator.com/2016/06/testing-jupyter-notebooks/
    """

    dirname, _ = os.path.split(path)
    os.chdir(dirname)
    with tempfile.NamedTemporaryFile(suffix=".ipynb") as fout:
        args = [
            "jupyter",
            "nbconvert",
            "--to",
            "notebook",
            "--execute",
            "--ExecutePreprocessor.timeout=400",
            "--output",
            fout.name,
            path,
        ]
        subprocess.check_call(args)

        fout.seek(0)
        nb = nbformat.read(fout, nbformat.current_nbformat)

    errors = [
        output
        for cell in nb.cells
        if "outputs" in cell
        for output in cell["outputs"]
        if output.output_type == "error"
    ]

    return nb, errors


def _find_all_notebooks_to_run():
    from datafold import __path__ as datafold_path

    assert len(datafold_path) == 1
    datafold_path = datafold_path[0]

    examples_path = os.path.join(datafold_path, "..", "tutorials")
    examples_path = os.path.abspath(examples_path)

    example_notebooks = []
    for current_path, directories, files in os.walk(examples_path):
        for file in files:
            if file.endswith(".ipynb") and ".ipynb_checkpoints" not in current_path:

                insert_path = os.path.join(current_path, file)
                assert os.path.exists(insert_path)

                if os.path.basename(insert_path) not in IGNORE_NOTEBOOKS:
                    example_notebooks.append(insert_path)

    return example_notebooks


@pytest.mark.parametrize("nb_path", _find_all_notebooks_to_run())
def test_notebooks(nb_path):
    _, errors = _notebook_run(nb_path)
    assert errors == []
