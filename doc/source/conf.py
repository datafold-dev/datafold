#!/usr/bin/env python3
# type: ignore

# Configuration file for the Sphinx documentation builder.

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute

import importlib
import os
import shutil
import sys
from datetime import datetime

PATH2DOC = os.path.abspath(".")
PATH2ROOT = os.path.abspath(os.path.join(PATH2DOC, "..", ".."))
PATH2SRC = os.path.abspath(os.path.join(PATH2ROOT, "datafold"))

try:
    sys.path.insert(0, PATH2DOC)
    sys.path.insert(0, PATH2ROOT)
    sys.path.insert(0, PATH2SRC)

    from datafold import __version__
except ImportError:
    raise ImportError(f"The path to datafold is not correct \npath:" f"{PATH2ROOT}")

# For a details on Sphinx configuration see documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
# -- Project information -----------------------------------------------------------------
project = "datafold"
copyright = f"2019-{datetime.now().year}, the datafold contributors"
author = "datafold contributors"
version = __version__
release = version  # no need to make it separate
today_fmt = "%d %B %Y"

# -- General configuration ---------------------------------------------------------------

needs_sphinx = "3.4.0"

# document name of the “master” document, that is, the document that contains the root
# toctree directive
master_doc = "index"

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom ones.
extensions = [
    # See build_full.sh file to execute sphinx-apidoc which fetches
    # the documentation from the Python source code automatically.
    "sphinx.ext.autodoc",
    # generates function/method/attribute summary lists
    "sphinx.ext.autosummary",
    # See below for configuration of _todo extension
    "sphinx.ext.todo",
    # See below for configuration
    "sphinx.ext.imgmath",
    # Include bibtex citations
    # see https://sphinxcontrib-bibtex.readthedocs.io/en/latest/quickstart.html#overview
    "sphinxcontrib.bibtex",
    # 'napoleon' allows NumPy and Google style documentation (no external Sphinx
    #  package required)
    #  -> https://www.sphinx-doc.org/en/master/usage/extensions/napoleon.html
    # numpydoc docstring guide
    #  -> https://numpydoc.readthedocs.io/en/latest/format.html
    "sphinx.ext.napoleon",
    # Provides automatic generation of API documentation pages for Python package
    # modules. https://sphinx-automodapi.readthedocs.io/en/latest/
    "sphinx_automodapi.automodapi",
    # Allows to use type-hinting for documenting acceptable argument types and return
    # value types of functions.
    # https://github.com/agronholm/sphinx-autodoc-typehints
    # NOTE: sphinx_autodoc_typehints must be AFTER the "sphinx.ext.napoleon" include!!
    # https://github.com/agronholm/sphinx-autodoc-typehints/issues/15#issuecomment\-298224484
    "sphinx_autodoc_typehints",
    # Tries to find the source files where the objects are contained. When found,
    # a separate HTML page will be output for each module with a highlighted version of
    # the source code.
    # https://www.sphinx-doc.org/en/master/usage/extensions/viewcode.html
    "sphinx.ext.viewcode",
    # Generate automatic links to the documentation of objects in other projects.
    # see options below
    # https://www.sphinx-doc.org/en/master/usage/extensions/intersphinx.html
    "sphinx.ext.intersphinx",
    # https://nbsphinx.readthedocs.io/en/0.8.5/
    # provides a source parser for *.ipynb files
    "nbsphinx",
    # Include notebook files from outside the sphinx source root.
    # https://github.com/vidartf/nbsphinx-link
    "nbsphinx_link",
    # Include panels in a grid layout or as drop-downs
    # https://sphinx-panels.readthedocs.io/en/latest/
    "sphinx_panels",
    # Include copy buttons in code blocks
    # https://sphinx-copybutton.readthedocs.io/en/latest/
    "sphinx_copybutton",
]

# If the API folder is not removed, classes that were renamed can produce errors
# because the old files are still around.
remove_api_folder = True
if remove_api_folder:
    try:
        shutil.rmtree(os.path.join(PATH2DOC, "api"))
    except FileNotFoundError:
        pass  # no worries the folder is already not there anymore

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
# sphinx.ext.imgmath -- only the image version allows to include full latex functionality
# MathJax has other advantages (such as copying the equations in latex format) but does
# only support basic functionality
# https://www.sphinx-doc.org/en/master/usage/extensions/math.html#module-sphinx.ext.imgmath

imgmath_image_format = "png"  # default "png", other option "svg"
imgmath_add_tooltips = (
    True  # add the LaTeX code as an “alt” attribute for math images --
)
# (like on Wikipedia equations)
imgmath_font_size = 12  # default=12

# command name with which to invoke LaTeX. The default is 'latex';
# you may need to set this to a full path if latex is not in the executable search path
imgmath_latex = "latex"
imgmath_latex_args = []  # TODO raise error if not found?
imgmath_latex_preamble = r"\usepackage{amsmath,amstext}"

# ----------------------------------------------------------------------------------------
# "sphinxcontrib.bibtex"
# Because exported BibTex files include file information to PDF -- remove in the
# following snippet.

filepath_literature_file = os.path.join(".", "_static", "literature.bib")
filepath_literature_file = os.path.abspath(filepath_literature_file)

# read content
with open(filepath_literature_file, "r") as file:
    content = file.read()

# leave out 'file' keys out
new_content = []
for line in content.splitlines(keepends=True):
    if not line.lstrip().startswith("file") and not line.lstrip().startswith("urldate"):
        new_content.append(line)

# write content back to file
with open(filepath_literature_file, "w") as file:
    file.write("".join(new_content))

bibtex_bibfiles = [filepath_literature_file]

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
}

# TODO: many pandas links are not resolved -- See:
#  https://github.com/agronholm/sphinx-autodoc-typehints/issues/47
#  in order to have not a mix between some links that work and many that don't
#  pandas is unfortunately excluded for now
#  a solution would be to make an own .inv file, that replaces the short links to
#  deep-links (see github issue)
#  "pandas": ("http://pandas.pydata.org/pandas-docs/dev", None)
#  ~
#  See also: https://sphobjinv.readthedocs.io/en/latest/customfile.html

# The maximum number of days to cache remote inventories.
intersphinx_cache_limit = 5  # default = 5

# The number of seconds for timeout.
intersphinx_timeout = 30

# ----------------------------------------------------------------------------------------
# nbsphinx - provides a source parser for *.ipynb files
# generate automatic links to the documentation of objects in other projects.
# https://nbsphinx.readthedocs.io/en/0.6.0/usage.html#nbsphinx-Configuration-Values

nbsphinx_allow_errors = False

try:
    # allows to set expensive tutorial execution with environment variable
    # the environment variable should be set if publishing the pages
    nbsphinx_execute = str(os.environ["DATAFOLD_NBSPHINX_EXECUTE"])
    print(nbsphinx_execute)
    assert nbsphinx_execute in ["auto", "always", "never"]
    print(
        f"INFO: found valid DATAFOLD_NBSPHINX_EXECUTE={nbsphinx_execute} environment "
        f"variable."
    )
except KeyError:
    # default
    print(
        "INFO: no environment variable DATFOLD_NBSPHINX_EXECUTE. Defaulting to not "
        "execute tutorial files."
    )
    nbsphinx_execute = "never"

nbsphinx_execute_arguments = [
    "--InlineBackend.figure_formats={'svg', 'pdf'}",
    "--InlineBackend.rc={'figure.dpi': 96}",
]

# add datafold and tutorials folder to PYTHONPATH to run jupyter notebooks
os.environ["PYTHONPATH"] = f"{PATH2ROOT}:{os.path.join(PATH2ROOT, 'tutorials')}"

# code parts were taken from here https://stackoverflow.com/a/67692
spec = importlib.util.spec_from_file_location(
    "tutorials_script", os.path.join(PATH2DOC, "generate_tutorials_page.py")
)
tutorials_script = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tutorials_script)

tutorials_script.setup_tutorials()

# ----------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["README.rst", "setup.py"]


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.

# html_theme = "sphinx_rtd_theme" # alternative theme
html_theme = "pydata_sphinx_theme"
html_logo = "_static/img/datafold_logo_pre.svg"

html_theme_options = {
    "icon_links": [
        {
            "name": "GitLab",
            "url": "https://gitlab.com/datafold-dev/datafold/",
            "icon": "fab fa-gitlab",
        },
        {
            "name": "PyPI",
            "url": "https://pypi.org/project/datafold/",
            "icon": "fab fa-python",
        },
    ],
    "icon_links_label": "Quick Links",
}


# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
