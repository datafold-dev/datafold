.DEFAULT_GOAL := help
.PHONY: help venv print_variables install_devdeps install_docdeps versions docs docs_linkcheck unittest tutorialtest tutorial precommit ci build install uninstall test_install pypi pypi_test clean_docs clean_build clean_test clean_venv clean clean_all

#Internal variables:
CURRENT_PATH = $(shell pwd)/
VENV_DIR = .venv

# Windows has a predefined variable "OS=Windows_NT"
OS ?= Linux
HTML_DOC_PATH = $(CURRENT_PATH)doc/build/html/

#help: @ List available targets and options in the Makefile with a short description.
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
	ifeq ($(OS),Linux)
		ACTIVATE_VENV = . $(VENV_DIR)/bin/activate; which python
	else
		ACTIVATE_VENV = . $(VENV_DIR)/Scripts/activate; which python
	endif
else
	# If Makefile is executed in Docker no activation of an virtual environment is required,
	# because Docker is already a virtual environment.
	ACTIVATE_VENV = which python
endif

SPHINXOPTS    ?=
PYTESTOPTS    ?=

DATAFOLD_NBSPHINX_EXECUTE ?= never
OUTPUT_DOCS ?= doc/build/

ifeq ($(OS),Linux)
	_DOCDEPS = pandoc texlive-base texlive-latex-extra graphviz libjs-mathjax fonts-mathjax dvipng
else # Windows_NT
	_DOCDEPS = pandoc miktex graphviz
endif

ifeq ($(IS_DOCKER),)
	OPEN_BROWSER ?= true
else # is in docker environment
    # do not open browser by default, because no browser is installed in the container
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
errmsg = "Used Python version (={}) invalid.\nMinimum Python version required: {}.{}\n".format(sys.version, sys.argv[1], sys.argv[2])
if sys.version_info.major < int(sys.argv[1]):
	raise RuntimeError(errmsg)
elif sys.version_info.minor < int(sys.argv[2]):
	raise RuntimeError(errmsg)
else:
	print("Python version {}.{} valid".format(sys.version_info.major, sys.version_info.minor))
endef
export PYTHON_CHECK_SCRIPT

# used for debugging / information in CI pipelines (not documented)
print_variables:
	@echo OS = $(OS)
	@echo CURRENT_PATH = $(CURRENT_PATH)
	@echo IS_DOCKER = $(IS_DOCKER)
	@echo PYTHON = $(PYTHON)
	@echo VPYTHON_MIN_MAJOR.VPYTHON_MIN_MINOR = $(VPYTHON_MIN_MAJOR).$(VPYTHON_MIN_MINOR)
	@echo IS_DOCKER = $(IS_DOCKER)
	@echo ACTIVATE_VENV = "$(ACTIVATE_VENV)"
	@echo SPHINXOPTS = $(SPHINXOPTS)
	@echo PYTESTOPTS = $(PYTESTOPTS)
	@echo EXECUTE_TUTORIAL = $(EXECUTE_TUTORIAL)
	@echo OUTPUT_DOCS = $(OUTPUT_DOCS)
	@echo OPEN_BROWSER = $(OPEN_BROWSER)
	@echo HTML_FILE_OPEN = $(HTML_FILE_OPEN)
	@echo GITHOOK = $(GITHOOK)

#venv: @ Create a new Python virtual environment if it does not exist yet (target is disabled in a docker container).
venv:
ifeq ($(IS_DOCKER),)
	@# Only create a Python-venv if not in a docker environment, because docker is already a virtualization
	@# if venv exists already, then check that the Python version meets the minimum version
	@# else create a new Python venv and check that the Python version (in PYTHON) meets the minimum version
	@if [ -d "$(CURRENT_PATH)$(VENV_DIR)" ]; then \
  		echo "Check Python set in virtual environment:"; \
  		$(ACTIVATE_VENV); \
  		python -c "$$PYTHON_CHECK_SCRIPT" $(VPYTHON_MIN_MAJOR) $(VPYTHON_MIN_MINOR); \
  	else \
		echo "Check Python set in variable PYTHON:"; \
		$(PYTHON) -c "$$PYTHON_CHECK_SCRIPT" $(VPYTHON_MIN_MAJOR) $(VPYTHON_MIN_MINOR); \
		$(PYTHON) -m venv $(VENV_DIR); \
	fi
else
	@# Check Python version in the docker container.
	@$(PYTHON) -c "$$PYTHON_CHECK_SCRIPT" $(VPYTHON_MIN_MAJOR) $(VPYTHON_MIN_MINOR);
endif

#docker_build: @ Build a new docker image from the Dockerfile (named "datafold").
docker_build:
	docker build -t datafold .

#docker_run: @ Start and execute a new docker container based on the available image (interactive session). The datafold repo is mounted into the container as "datafold-mount".
docker_run:
	docker run -v `pwd`:/home/datafold-mount -w /home/datafold-mount/ -it --rm --net=host datafold bash

#install_devdeps: @ Install (or update) development dependencies in virtual environment according to the file "requirements-dev.txt".
install_devdeps: venv
	$(ACTIVATE_VENV); \
    python -m pip install --upgrade pip wheel setuptools twine; \
	python -m pip install -r requirements-dev.txt

#install_docdeps: @ Install non-Python dependencies to build datafold's documentation (may require 'sudo' rights).
install_docdeps:
ifeq ($(OS),Linux)
ifeq ($(IS_DOCKER),) # no docker
	sudo apt install $(_DOCDEPS)
else # in docker "sudo" is not available and everything is executed with root
	apt-get update && \
	apt-get -y install $(_DOCDEPS)
endif
else # OS = Windows
	@echo "INFO: Make sure that chocolatery is installed ()"
	@echo "INFO: Make sure to execute with sudo / administrator rights."
	choco install $(_DOCDEPS)
endif

#devenv: @ Set up development environment (executing targets 'install_devdeps' and 'install_docdeps')
devenv: install_devdeps install_docdeps

#versions: @ Show current datafold version and of the essential dependencies.
versions:
	@$(ACTIVATE_VENV); \
	python datafold/_version.py

#docs: @ Build datafold documentation with Sphinx.
docs:
	@$(ACTIVATE_VENV); \
	echo Execute tutorials environment variable: DATAFOLD_NBSPHINX_EXECUTE=$(DATAFOLD_NBSPHINX_EXECUTE); \
	python -m sphinx -M html doc/source/ $(OUTPUT_DOCS) $(SPHINXOPTS) $(O);
	@# Open the default browser at the page specified in URL_DOC_OPEN
ifeq ($(OPEN_BROWSER),true)
ifeq ($(OS),Linux)
	@$(PYTHON) -m webbrowser "$(URL_DOC_OPEN)"
else  # Windows
	@start "$(URL_DOC_OPEN)"
endif
endif

#docs_linkcheck: @ Check if the URLs in the documentation are valid.
docs_linkcheck:
	@$(ACTIVATE_VENV); \
	cd doc/; \
	python -m sphinx -M linkcheck source/ build/ $(SPHINXOPTS) $(O)

#unittest: @ Run all unittests with pytest in datafold.
unittest:
	@$(ACTIVATE_VENV); \
	python -m coverage run --branch -m pytest $(PYTESTOPTS) datafold/; \
	python -m coverage html -d ./coverage/; \
	python -m coverage report;

#tutorialtest: @ Run all tutorials with pytest.
tutorialtest:
	@$(ACTIVATE_VENV); \
	export PYTHONPATH="$(CURRENT_PATH):$$PYTHONPATH"; \
	python -m pytest tutorials/;

test: unittest tutorialtest

#tutorial: @ Open tutorials in Jupyter notebook (opens a window in the default browser).
tutorial:
	@$(ACTIVATE_VENV); \
	export PYTHONPATH=$(CURRENT_PATH):$$PYTHONPATH; \
	python -m notebook $(CURRENT_PATH)/tutorials/

#precommit: @ Run git hooks to check and analyze the code. Manged by pre-commit.
precommit:
	@$(ACTIVATE_VENV); \
	python -m pre_commit run $(GITHOOK);

#ci: @ Run continuous integration pipeline.
ci: install_devdeps test precommit test_install

#build: @ Build a Python source distribution (sdist) and wheel (bdist_wheel).
build:
	@$(ACTIVATE_VENV); \
	python setup.py sdist bdist_wheel; \

#install: @ Install datafold in virtual environment.
install:
	@$(ACTIVATE_VENV); \
	python setup.py install

#test_install: @ Install and subsequently uninstall datafold for testing purposes (all created files are removed).
test_install: install clean_install
	@echo 'Successful'

#pypi: @ Upload and release datafold to https://pypi.org/ (requires account and password).
pypi: build
	@$(ACTIVATE_VENV); \
	python setup.py sdist bdist_wheel; \
	python -m twine check dist/*; \
	python -m twine upload --verbose dist/*

#pypi_test: @ Upload and release datafold to https://test.pypi.org/ for testing purposes (requires account and password).
test_pypi: build
	@$(ACTIVATE_VENV); \
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

#clean_precommit: @ Remove all files that are created for target "precommit".
clean_precommit:
	@$(ACTIVATE_VENV); \
	python -m pre_commit clean;

#clean_test: @ Remove all files that are created for target "unittest".
clean_test:
	rm -fr .pytest_cache/;
	rm -f .coverage;
	rm -fr coverage/;

#clean_cache: @ Remove all cache files from the repository.
clean_cache:
	rm -rf .mypy_cache;
	rm -fr .pytest_cache/;
	find . -path '*/__pycache__/*' -delete
	find . -path '*/.ipynb_checkpoints/*' -delete

#clean_docker: @ Remove all unused docker images in docker (not just dangling ones).
clean_docker:
	docker system prune -a

#uninstall: @ Remove all files for the target "install".
clean_install: clean_build
	@$(ACTIVATE_VENV); \
	yes | pip uninstall datafold

#clean_venv: @ Remove the virtual environment folder which is created for the target "venv".
clean_venv:
	rm -fr $(VENV_DIR);

#clean: @ Call targets "clean_docs", "clean_install", "clean_test", "clean_precommit" and "clean_cache".
clean: clean_docs clean_install clean_test clean_precommit clean_cache
