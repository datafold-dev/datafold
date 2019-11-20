# Configuration file for the Sphinx documentation builder.
#
# For a full list of Sphinx configuration see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute
#
import os
import sys

PATH2ROOT = os.path.abspath(os.path.join(".", "..", ".."))
PATH2SRC = os.path.abspath(os.path.join(PATH2ROOT, "datafold"))
sys.path.insert(0, PATH2ROOT)
sys.path.insert(0, PATH2SRC)

import sphinx_rtd_theme  # "Read the doc" theme -- https://sphinx-rtd-theme.readthedocs.io/en/stable/
from datafold import __version__


# -- Project information -----------------------------------------------------
project = "datafold"
copyright = "2019, datafold contributors"
author = "datafold contributors"
version = __version__

# -- General configuration ---------------------------------------------------

needs_sphinx = "2.0"

# document name of the “master” document, that is, the document that contains the root toctree directive
master_doc = "index"

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [  # See build_full.sh file to execture sphinx-apidoc which fetches the documentation automatically.
    "sphinx.ext.autodoc",
    # See below for configuration
    "sphinx.ext.todo",
    # See below for configuration
    "sphinx.ext.imgmath",
    # see https://sphinxcontrib-bibtex.readthedocs.io/en/latest/quickstart.html
    "sphinxcontrib.bibtex",
    # 'napoleon' allows NumPy and Google style documentation (no external Sphinx package required)
    #  -> https://www.sphinx-doc.org/en/master/usage/extensions/napoleon.html
    # numpydoc docstring guide
    #  -> https://numpydoc.readthedocs.io/en/latest/format.html
    "sphinx.ext.napoleon",
    "sphinx_rtd_theme",
]

# --------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------
# sphinx.ext.todo:
# See all options: http://www.sphinx-doc.org/en/master/ext/todo.html
# If this is True, todo and todolist produce output, else they produce nothing. The default is False.
todo_include_todos = True

# If this is True, todo emits a warning for each TODO entry. The default is False.
todo_emit_warnings = False


# ---------------------------------------------------------------------------------------------------------------
# sphinx.ext.imgmath -- only the image version allows to include full latex functionality
# MathJax has other advantages (such as copying the equations in latex format) but does only support basic functionality
# https://www.sphinx-doc.org/en/master/usage/extensions/math.html#module-sphinx.ext.imgmath

imgmath_image_format = "png"  # default "png", other option "svg"
imgmath_add_tooltips = True  # add the LaTeX code as an “alt” attribute for math images -- (like on Wikipedia equations)
imgmath_font_size = 12  # default=12

# command name with which to invoke LaTeX. The default is 'latex';   # TODO raise error if not found?
# you may need to set this to a full path if latex is not in the executable search path
imgmath_latex = "latex"
imgmath_latex_args = []
imgmath_latex_preamble = r"\usepackage{amsmath,amstext}"

# ---------------------------------------------------------------------------------------------------------------
# napoleon (see full list of available options:
# Full config explanations here: https://www.sphinx-doc.org/en/master/usage/extensions/napoleon.html#configuration

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
napoleon_use_ivar = False

# True -> :param: role for each function parameter.
# False -> use a single :parameters: role for all the parameters.
napoleon_use_param = True
napoleon_use_keyword = True

# True to use the :rtype: role for the return type. False to output the return type inline with the description.
napoleon_use_rtype = True

# ---------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
html_theme = "sphinx_rtd_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
