import os
import subprocess
import tempfile
import unittest

import nbformat

IGNORED_TESTED_NOTEBOOKS = []


class TestNotebooks(unittest.TestCase):
    @staticmethod
    def _notebook_run(path):
        """Execute a notebook via nbconvert and collect output.
           :returns (parsed nb object, execution errors)

           Source:
           https://blog.thedataincubator.com/2016/06/testing-jupyter-notebooks/
        """

        dirname, __ = os.path.split(path)
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

    def _find_all_notebooks_to_run(self):
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
                    example_notebooks.append(insert_path)
        return example_notebooks

    def setUp(self) -> None:
        self.notebook_paths = self._find_all_notebooks_to_run()
        self.assertGreater(len(self.notebook_paths), 0)

    def check_errors_notebook(self, nb_path):
        _, errors = self._notebook_run(nb_path)
        self.assertEqual(errors, [])

    def test_notebooks(self):
        for nb_path in self.notebook_paths:
            if os.path.basename(nb_path) not in IGNORED_TESTED_NOTEBOOKS:
                self.check_errors_notebook(nb_path)


if __name__ == "__main__":
    tests = TestNotebooks()
    tests.setUp()
    tests.test_notebooks()
