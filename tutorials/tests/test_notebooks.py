import os
import subprocess
import tempfile

import nbformat
import pytest

IGNORE_NOTEBOOKS: list[str] = [
    "12_ResDMD.ipynb",
    "12_koopman_mpc.ipynb",
    "koopman_mpc.ipynb",
]


def _notebook_run(path):
    """Execute a notebook via nbconvert and collect output. Returns the parsed notebook object
    and the execution errors.

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

    import pathlib

    datafold_path = pathlib.Path(datafold_path[0])

    tutorials_path = datafold_path.parent / "tutorials"

    assert tutorials_path.is_dir()

    example_notebooks = []

    for ipynb_filepath in tutorials_path.rglob("*.ipynb"):
        if (
            ".ipynb_checkpoints" not in str(ipynb_filepath)
            and ipynb_filepath.name not in IGNORE_NOTEBOOKS
        ):
            assert ipynb_filepath.is_file()
            example_notebooks.append(ipynb_filepath)

    return example_notebooks


@pytest.mark.parametrize("nb_path", _find_all_notebooks_to_run())
def test_notebooks(nb_path):
    _, errors = _notebook_run(nb_path)
    assert not errors
