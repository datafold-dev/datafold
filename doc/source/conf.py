# type: ignore

# Configuration file for the Sphinx documentation builder.

# -- Path setup --------------------------------------------------------------
# If extensions (or modules to document with autodoc) are in another directory,
# add these (absolute path) directories to sys.path here.

import importlib
import os
import pathlib
import shutil
import sys
from datetime import datetime

PATH2DOC = pathlib.Path(__file__).parent.resolve()
PATH2ROOT = PATH2DOC.parent.parent
PATH2SRC = PATH2ROOT.joinpath("datafold")

try:
    sys.path.insert(0, PATH2DOC.as_posix())
    sys.path.insert(0, PATH2ROOT.as_posix())
    sys.path.insert(0, PATH2SRC.as_posix())

    from datafold import __version__
except ImportError:
    raise ImportError(
        f"The path to the datafold root folder ({PATH2ROOT=}) is incorrect. "
        f"Check in conf.py file"
    )

# For a details on Sphinx configuration see documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
# -- Project information -----------------------------------------------------------------
project = "datafold"
copyright = f"2019-{datetime.now().year}, the datafold contributors"  # noqa: A001
author = "datafold contributors"
version = __version__
release = version  # no need to make separate from version
today_fmt = "%d %B %Y"

# -- General configuration ---------------------------------------------------------------

needs_sphinx = "4.5.0"

# document name of the “master” document, that is, the document that contains the root
# toctree directive
master_doc = "index"

# Sphinx extension modules names:
extensions = [
    # See makefile target docs to execute sphinx-apidoc which fetches
    # the documentation from the Python source code automatically.
    "sphinx.ext.autodoc",
    # generates function/method/attribute summary lists
    "sphinx.ext.autosummary",
    # See below for configuration of _todo extension
    "sphinx.ext.todo",
    # See below for configuration (required to render equations)
    "sphinx.ext.imgmath",
    # Include bibtex citations
    # see https://sphinxcontrib-bibtex.readthedocs.io/en/latest/index.html
    "sphinxcontrib.bibtex",
    # 'napoleon' supports NumPy and Google style documentation (no external Sphinx module
    #  required)
    #  -> https://www.sphinx-doc.org/en/master/usage/extensions/napoleon.html
    # numpydoc docstring guide
    #  -> https://numpydoc.readthedocs.io/en/latest/format.html
    "sphinx.ext.napoleon",
    # Provides automatic generation of API documentation pages for Python package
    # modules. https://sphinx-automodapi.readthedocs.io/en/latest/
    "sphinx_automodapi.automodapi",
    # Allows using type-hinting for documenting acceptable argument types and return
    # value types of functions.
    # https://github.com/agronholm/sphinx-autodoc-typehints
    # NOTE: sphinx_autodoc_typehints must be included AFTER "sphinx.ext.napoleon" module!!
    # https://github.com/agronholm/sphinx-autodoc-typehints/issues/15#issuecomment\-298224484
    "sphinx_autodoc_typehints",
    # Tries to find the source files where the objects are contained. When found,
    # a separate HTML page will be included in the docs for each class:
    # https://www.sphinx-doc.org/en/master/usage/extensions/viewcode.html
    "sphinx.ext.viewcode",
    # Generate automatic links to the documentation of Python objects in other projects.
    # see options below
    # https://www.sphinx-doc.org/en/master/usage/extensions/intersphinx.html
    "sphinx.ext.intersphinx",
    # https://nbsphinx.readthedocs.io/en/0.8.5/
    # provides a source parser for Jupyter notebooks (*.ipynb files)
    "nbsphinx",
    # Include notebook files from outside the sphinx source root (required for the tutorials)
    # https://github.com/vidartf/nbsphinx-link
    "nbsphinx_link",
    # Include design elements, such as a panel grid layout or drop-down menus
    # https://sphinx-design.readthedocs.io/en/furo-theme/
    "sphinx_design",
    # Include copy buttons in code blocks
    # https://sphinx-copybutton.readthedocs.io/en/latest/
    "sphinx_copybutton",
]

# If the API folder is not removed, classes that were renamed can produce errors
# because the old files are still around.
remove_api_folder = True

if remove_api_folder:
    try:
        shutil.rmtree(PATH2DOC / "api")
    except FileNotFoundError:
        pass  # no worries, the folder is not there

# ----------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------
# sphinx.ext._todo:
# See all options: http://www.sphinx-doc.org/en/master/ext/todo.html
# If this is True, _todo and todolist produce output, else they produce nothing. The
# default is False.
todo_include_todos = True

# If this is True, _todo emits a warning for each _TODO entry. The default is False.
todo_emit_warnings = False

# ----------------------------------------------------------------------------------------
# sphinx.ext.imgmath -- only the image version allows including full Latex functionality
# MathJax has other advantages (such as copying the equations in Latex format) but does
# only support basic functionality
# https://www.sphinx-doc.org/en/master/usage/extensions/math.html#module-sphinx.ext.imgmath

imgmath_image_format = "png"  # default "png", other option "svg"
# add the LaTeX code as an "alt" attribute for math images (like on Wikipedia equations)
imgmath_add_tooltips = True

imgmath_font_size = 12  # default=12

# command name with which to invoke LaTeX. The default is 'latex';
# you may need to set this to a full path if latex is not in the executable search path
imgmath_latex = "latex"
imgmath_latex_args = []
imgmath_latex_preamble = r"\usepackage{amsmath,amstext}"

# ----------------------------------------------------------------------------------------
# "sphinxcontrib.bibtex"
# see https://sphinxcontrib-bibtex.readthedocs.io/en/latest/usage.html#configuration

filepath_literature_file = PATH2DOC / "_static" / "literature.bib"
assert filepath_literature_file.is_file()
bibtex_reference_style = "author_year"

# currently supported "alpha" (default)  "plain", "unsrt", "unsrtalpha"
bibtex_default_style = "plain"
bibtex_bibfiles = [str(filepath_literature_file)]

# ----------------------------------------------------------------------------------------
# napoleon (see full list of available options:
# Full config explanations here:
# https://www.sphinx-doc.org/en/master/usage/extensions/napoleon.html#configuration

napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False

# include private members (like _membername)
napoleon_include_private_with_doc = False

# include special members (like __membername__)
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False

# use the :ivar: role for instance variables
# shows the "Attributes" section
napoleon_use_ivar = True

# True -> :param: role for each function parameter.
# False -> use a single :parameters: role for all the parameters.
napoleon_use_param = True
napoleon_use_keyword = True

# True to use the :rtype: role for the return type. False to output the return type inline
# with the description.
napoleon_use_rtype = True

# ----------------------------------------------------------------------------------------
# sphinx_automodapi.automodapi (see full list of available options:
# Full config explanations here:
# https://sphinx-automodapi.readthedocs.io/en/latest/

# Do not include inherited members by default
automodsumm_inherited_members = False

# ----------------------------------------------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/extensions/intersphinx.html
# generate automatic links to the documentation of objects in other projects.
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scikit-learn": ("https://scikit-learn.org/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/reference/", None),
    "pandas": ("http://pandas.pydata.org/pandas-docs/stable/", None),
}

# The maximum number of days to cache remote inventories.
intersphinx_cache_limit = 5  # default = 5

# The number of seconds for timeout.
intersphinx_timeout = 30

# ----------------------------------------------------------------------------------------
# nbsphinx - provides a source parser for *.ipynb files
# generate automatic links to the documentation of objects in other projects.
# https://nbsphinx.readthedocs.io/en/0.6.0/usage.html#nbsphinx-Configuration-Values

nbsphinx_allow_errors = False
nbsphinx_execute = "never"  # do not use nbsphinx to execute the tutorials (see MR !106)
nbsphinx_prolog = (
    "Visit the "
    "`tutorials page <https://datafold-dev.gitlab.io/datafold/tutorial_index.html>`__ "
    "to view all notebooks."
)

# allows setting expensive tutorial execution with environment variable
# the environment variable should be set if publishing the pages
nb_execute_env = os.environ.get("DATAFOLD_TUTORIALS_EXECUTE", "").lower()

if nb_execute_env == "true":
    nb_execute = True
elif nb_execute_env in ("false", "", None):
    nb_execute = False
else:
    raise ValueError(f"DATAFOLD_TUTORIALS_EXECUTE={nb_execute_env} not a valid choice")

# add datafold and tutorials folder to PYTHONPATH to run jupyter notebooks
os.environ["PYTHONPATH"] = f"{PATH2ROOT}:{PATH2ROOT / 'tutorials'}"

# next code lines were taken from https://stackoverflow.com/a/67692
spec = importlib.util.spec_from_file_location(
    "tutorials_script", PATH2DOC / "generate_tutorials_page.py"
)

tutorials_script = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tutorials_script)

tutorials_script.setup_tutorials()

nb_execute_arguments = [
    "--InlineBackend.figure_formats={'svg', 'pdf'}",
    "--InlineBackend.rc={'figure.dpi': 96}",
]

# execute tutorials
if nb_execute:
    tutorials_script.execute_tutorials(extra_arguments=nb_execute_arguments)

# ----------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------

# Add any path that contain templates here (relative to this directory)
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["README.rst", "setup.py"]


# -- Options for HTML output -------------------------------------------------

# The theme to use for the HTML documentation.

# html_theme = "sphinx_rtd_theme" # alternative theme
html_theme = (
    "pydata_sphinx_theme"  # https://pydata-sphinx-theme.readthedocs.io/en/stable/
)
html_logo = "_static/img/datafold_logo_pre.svg"

html_context = {"default_mode": "light"}  # auto, dark, light

html_theme_options = {
    "icon_links": [
        {
            "name": "GitLab",
            "url": "https://gitlab.com/datafold-dev/datafold/",
            "icon": "fab fa-gitlab",
            "type": "fontawesome",
        },
        {
            "name": "PyPI",
            "url": "https://pypi.org/project/datafold/",
            "icon": "fab fa-python",
            "type": "fontawesome",
        },
    ],
    "icon_links_label": "Quick Links",
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
