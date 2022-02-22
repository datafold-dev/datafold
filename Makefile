.DEFAULT_GOAL := help

#Internal variables:
CURRENT_PATH = $(shell pwd)/
VENV_DIR = .venv
OS ?= Linux  # Windows has a predefined variable "OS=Windows_NT"

HTML_DOC_PATH = $(CURRENT_PATH)/doc/build/html/

.PHONY: help venv print_variables install_devdeps install_docdeps versions docs docs_linkcheck unittest tutorialtest tutorial precommit ci build install uninstall test_install pypi pypi_test clean_docs clean_build clean_test clean_venv clean clean_all

#help: @ Lists available targets and short description in this makefile.
help:
	@echo 'Execute with "make [target] [arguments]"'.
	@echo 'Example: "make docs OPEN_BROWSER=false"'
	@echo ''
	@echo 'General arguments:'
	@echo 'PYTHON - Python interpreter to use to set up the virtual environment. Defaults to "python3".'
	@echo ''
	@echo 'Arguments associated to the target "docs".'
	@echo 'EXECUTE_TUTORIAL - Select whether to run tutorials (Jupter notebooks) in target "docs". {never [default], auto, always}'
	@echo 'OUTPUT_DOCS - Path to write the html output to. Defaults to "doc/build".'
	@echo 'SPHINXOPTS - Options passed to "sphinx-build".'
	@echo 'OPEN_BROWSER - Whether to open the browser (with file specified in "URL_DOC_OPEN") in target "docs".'
	@echo 'HTML_FILE_OPEN - If "OPEN_DOCS_BROWSER" is enabled, specify the html file that opens. Defaults to "index.html".'
	@echo ''
	@echo 'Arguments associated to the target "unittest".'
	@echo 'PYTESTOPTS - Options passed to "pytest" in target "unittest"'
	@echo ''
	@echo 'Arguments associated to the target "precommit".'
	@echo 'GITHOOK - Execute specific githook. Defaults to "--all".'
	@echo ''
	@echo 'Available targets:'
	@grep -E '[a-zA-Z\.\-]+:.*?@ .*$$' $(MAKEFILE_LIST)| tr -d '#'  | awk 'BEGIN {FS = ":.*?@ "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

PYTHON ?= python3

# Set "DOCKER_ENVIRONMENT" in Dockerfile as "ENV DOCKER_ENVIRONMENT" to indicate the Makefile to not use the additional Python virtualization
IS_DOCKER = $(DOCKER_ENVIRONMENT)

ifeq ($(IS_DOCKER),)
	# is not docker because variable IS_DOCKER is not available
	# activate Python virtual environment
	ifeq ($(OS),Windows_NT)
		ACTIVATE_VENV = . $(VENV_DIR)/Scripts/activate; which python
	else
		ACTIVATE_VENV = . $(VENV_DIR)/bin/activate; which python
	endif
else
	# If Makefile is executed in Docker (as a virtual environment)
	ACTIVATE_VENV = which python
endif

SPHINXOPTS    ?=
PYTESTOPTS    ?=

DATAFOLD_NBSPHINX_EXECUTE ?= never
OUTPUT_DOCS ?= doc/build/

_DOCDEPS = libjs-mathjax fonts-mathjax dvipng pandoc graphviz texlive-base texlive-latex-extra

ifeq ($(IS_DOCKER),)
	OPEN_BROWSER ?= true
else
# is_DOCKER=true -> do not open browser by default, because no browser is installed in the
# docker container
	OPEN_BROWSER ?= false
endif

HTML_FILE_OPEN ?= index.html
URL_DOC_OPEN = $(HTML_DOC_PATH)$(HTML_FILE_OPEN)

GITHOOK = --all

# Check that the used Python version fulfills the minimum requirement
VPYTHON_MIN_MAJOR = 3
VPYTHON_MIN_MINOR = 8

define PYTHON_CHECK_SCRIPT
import sys

errmsg = "Used Python version invalid: \n {} \n\nMinimum version required: $(VPYTHON_MIN_MAJOR).$(VPYTHON_MIN_MINOR)\n".format(sys.version)

if sys.version_info.major < int(sys.argv[1]):
	raise RuntimeError(errmsg)
elif sys.version_info.minor < int(sys.argv[2]):
	raise RuntimeError(errmsg)
else:
	print("Python version {}.{} valid".format(sys.version_info.major, sys.version_info.minor))
endef
export PYTHON_CHECK_SCRIPT

# Check that the Python version supports the minimum required version.
python_check:
	@$(PYTHON) -c "$$PYTHON_CHECK_SCRIPT" $(VPYTHON_MIN_MAJOR) $(VPYTHON_MIN_MINOR)

# used for debugging / information in CI pipelines (not documented)
print_variables:
	@echo OS = $(OS)
	@echo CURRENT_PATH = $(CURRENT_PATH)
	@echo IS_DOCKER = $(IS_DOCKER)
	@echo PYTHON = $(PYTHON)
	@echo VPYTHON_MIN_MAJOR = $(VPYTHON_MIN_MAJOR)
	@echo VPYTHON_MIN_MINOR = $(VPYTHON_MIN_MINOR)
	@echo IS_DOCKER = $(IS_DOCKER)
	@echo ACTIVATE_VENV = $(ACTIVATE_VENV)
	@echo SPHINXOPTS = $(SPHINXOPTS)
	@echo PYTESTOPTS = $(PYTESTOPTS)
	@echo EXECUTE_TUTORIAL = $(EXECUTE_TUTORIAL)
	@echo OUTPUT_DOCS = $(OUTPUT_DOCS)
	@echo OPEN_BROWSER = $(OPEN_BROWSER)
	@echo HTML_FILE_OPEN = $(HTML_FILE_OPEN)
	@echo GITHOOK = $(GITHOOK)

#venv: @ Create new Python virtual environment if it does not exist yet.
venv: python_check
ifeq ($(IS_DOCKER),)
# Create / use a Python virtual environment
# Do not create Python-venv in a docker environment, because docker is already a virtualization
	test -d $(VENV_DIR) || $(PYTHON) -m venv $(VENV_DIR);
endif

#docker_build: @ Build a docker image by processing the Dockerfile.
docker_build:
	docker build -t datafold .

#docker_run: @ Start and execute interactively sign into a new docker container (based on the available image). The datafold folder is mounted into the container as "datafold-mount".
docker_run:
	docker run -v `pwd`:/home/datafold-mount -w /home/datafold-mount/ -it --rm --net=host datafold bash

#docker_clean: @ Remove unused data from docker by removing all unused images (not just dangling ones)
docker_clean:
	docker system prune -a

#install_devdeps: @ Install (or update) development dependencies in virtual environment according to file "requirements-dev.txt".
install_devdeps: venv
	$(ACTIVATE_VENV); \
    python -m pip install --upgrade pip wheel setuptools twine; \
	python -m pip install -r requirements-dev.txt;

#install_doc_deps: @ Install dependencies to render datafold's documentation via apt-get (Note: this may require 'sudo').
install_docdeps:   #TODO: update for windows
ifeq ($(OS),Windows_NT)
	@echo "INFO: Make sure to execute with administrator rights"
	choco install pandoc miktex pandoc
ifeq ($(IS_DOCKER),) # no docker
	sudo apt-get install $(_DOCDEPS)
else # in docker "sudo" is not available and everything is executed with root
	apt-get -y install $(_DOCDEPS)
endif
endif

#versions: @ Print datafold version and essential dependency versions.
versions:
	$(ACTIVATE_VENV); \
	python datafold/_version.py

#docs: @ Build datafold documentation with Sphinx.
docs:
	$(ACTIVATE_VENV); \
	echo Execute tutorials environment variable: DATAFOLD_NBSPHINX_EXECUTE=$(DATAFOLD_NBSPHINX_EXECUTE); \
	python -m sphinx -M html doc/source/ $(OUTPUT_DOCS) $(SPHINXOPTS) $(O);
	@# Open the browser at the page specified in URL_DOC_OPEN
ifeq ($(OPEN_BROWSER),true)
ifeq ($(OS),Windows_NT)
	start "$(URL_DOC_OPEN)"
else  # Linux
	@ if which xdg-open > /dev/null; then \
		xdg-open $(URL_DOC_OPEN); \
	elif which gnome-open > /dev/null; then \
		gnome-open $(URL_DOC_OPEN); \
	fi
endif
endif

#docs_linkcheck: @ Check if all links in the documentation are valid.
docs_linkcheck:
	$(ACTIVATE_VENV); \
	cd doc/; \
	python -m sphinx -M linkcheck source/ build/ $(SPHINXOPTS) $(O)

#unittest: @ Run and report all unittests with pytest.
unittest:
	$(ACTIVATE_VENV); \
	python -m coverage run --branch -m pytest $(PYTESTOPTS) datafold/; \
	python -m coverage html -d ./coverage/; \
	python -m coverage report;

#tutorialtest: @ Run all tutorials with pytest to check for errors.
tutorialtest:
	$(ACTIVATE_VENV); \
	# TODO: in Windows the ":" causes troubles
	export PYTHONPATH=$(CURRENT_PATH):$$PYTHONPATH; \
	python -m pytest tutorials/;

test: unittest tutorialtest

#tutorial: @ Open tutorials in Jupyter notebook (this opens the browser).
tutorial:
	$(ACTIVATE_VENV); \
	export PYTHONPATH=$(CURRENT_PATH):$$PYTHONPATH; \
	python -m notebook $(CURRENT_PATH)/tutorials/

#precommit: @ Run git hooks to check and analyze the code.
precommit:
	$(ACTIVATE_VENV); \
	python -m pre_commit run $(GITHOOK);

#ci: @ Run continuous integration pipeline.
ci: install_devdeps test precommit test_install

#build: @ Build a source distribution and wheel.
build:
	$(ACTIVATE_VENV); \
	python setup.py sdist bdist_wheel; \

#install: @ Install datafold in virtual environment.
install:
	$(ACTIVATE_VENV); \
	python setup.py install

#uninstall: @ Uninstall datafold from virtual environment.
uninstall:
	$(ACTIVATE_VENV); \
	yes | pip uninstall datafold

#test_install: @ Install and subsequently uninstall datafold for testing purposes (all created files are removed).
test_install: install uninstall clean_build
	@echo 'Successful'

#pypi: @ Upload and release datafold to https://pypi.org/ (requires account and password).
pypi: build
	$(ACTIVATE_VENV); \
	python setup.py sdist bdist_wheel; \
	python -m twine check dist/*; \
	python -m twine upload --verbose dist/*

#pypi_test: @ Upload and release datafold to https://test.pypi.org/ for testing purposes (requires account and password).
test_pypi: build
	$(ACTIVATE_VENV); \
	python -m twine check dist/*; \
	python -m twine upload --verbose --repository testpypi dist/*

#clean_docs: @ Remove all files that are created for target "docs".
clean_docs:
	cd doc/; \
	rm -fr build/; \
	rm -f source/api/*.rst source/_apidoc/*.rst; \
	find . -name *.nblink -type f -delete; \
	# the README file in the "tutorials" folder is generated when building the docs
	rm -f tutorials/README.rst

#clean_build: @ Remove all files that are created for target "build".
clean_build:
	rm -fr build/;
	rm -fr dist/;
	rm -fr datafold.egg-info/;

#clean_test: @ Remove all files that are created for target "unittest".
clean_test:
	rm -fr .pytest_cache/;
	rm -f .coverage;
	rm -fr coverage/;

#clean_cache: @ Remove all cache files from the repository.
clean_cache:
	rm -rf .mypy_cache;
	rm -fr .pytest_cache/;
	find datafold/ -name __pycache__ -type d -delete;
	find . -name .ipynb_checkpoints -type d -delete;

#clean_venv: @ Remove all files that are created for target "venv".
clean_venv:
	rm -fr $(VENV_DIR);

#clean: @ Call targets "clean_docs", "clean_build", "clean_test" and "clean_cache".
clean: clean_docs clean_build clean_test clean_cache
