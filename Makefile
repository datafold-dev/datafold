.DEFAULT_GOAL := help
.PHONY: help venv print_variables install_deps install_devdeps install_docdeps versions docs docs_linkcheck unittest tutorialtest tutorial precommit ci build install uninstall test_install pypi pypi_test clean_docs clean_build clean_test clean_venv clean clean_all

#Internal variables:
CURRENT_PATH = $(shell pwd)/
VENV_DIR = .venv

# Windows has a predefined variable "OS=Windows_NT"
OS ?= Linux
HTML_DOC_PATH = $(CURRENT_PATH)doc/build/html/

#help: @ List the available targets and options in this Makefile with a short description.
help:
	@echo 'Execute with "make [target] [arguments]"'.
	@echo 'Example: "make docs OPEN_BROWSER=false"'
	@echo ''
	@echo 'General arguments:'
	@echo 'PYTHON - Python interpreter to use to set up the virtual environment. Defaults to "python3".'
	@echo ''
	@echo 'Arguments associated to the target "docs".'
	@echo 'DATAFOLD_TUTORIALS_EXECUTE - Select whether to run tutorials (Jupter notebooks) in target "docs". {false [default], true}'
	@echo 'OUTPUT_DOCS - Path to write the html output to. Defaults to "doc/build".'
	@echo 'SPHINXOPTS - Options passed to "sphinx-build".'
	@echo 'OPEN_BROWSER - Whether to open the browser (with file specified in "URL_DOC_OPEN") in target "docs".'
	@echo 'HTML_FILE_OPEN - If "OPEN_DOCS_BROWSER" is enabled, specify the html file that opens. Defaults to "index.html".'
	@echo ''
	@echo 'Arguments associated to the target "unittest".'
	@echo 'PYTESTOPTS - Options passed to "pytest" in target "unittest" and "tutorialtest" (defaults to --verbose)'
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
PYTESTOPTS    ?= --verbose

DATAFOLD_TUTORIALS_EXECUTE ?= false
OUTPUT_DOCS ?= doc/build/

ifeq ($(OS),Linux)
	# NOTE: a newer version of pandoc is installed separately in install_docdeps target for Linux
	_DOCDEPS = texlive-base texlive-lang-english texlive-latex-extra ffmpeg graphviz libjs-mathjax fonts-mathjax dvipng
else # Windows_NT
	_DOCDEPS = pandoc miktex ffmpeg graphviz
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
VPYTHON_MIN_MINOR = 9

define PYTHON_CHECK_SCRIPT
import sys
errmsg = "Python version (={}) invalid.\nMinimum Python version required: {}.{}\n".format(sys.version, sys.argv[1], sys.argv[2])
if sys.version_info.major < int(sys.argv[1]):
	raise RuntimeError(errmsg)
elif sys.version_info.minor < int(sys.argv[2]):
	raise RuntimeError(errmsg)
else:
	print("Valid Python version {}.{} detected (minimum version {}.{}).".format(sys.version_info.major, sys.version_info.minor, sys.argv[1], sys.argv[2]))
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
	@echo DATAFOLD_TUTORIALS_EXECUTE = $(DATAFOLD_TUTORIALS_EXECUTE)
	@echo OUTPUT_DOCS = $(OUTPUT_DOCS)
	@echo OPEN_BROWSER = $(OPEN_BROWSER)
	@echo HTML_FILE_OPEN = $(HTML_FILE_OPEN)
	@echo GITHOOK = $(GITHOOK)

#venv: @ Create a new Python virtual environment if it does not exist yet (target is disabled if DOCKERENVIRONMENT is set).
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


#install_deps: @ Install (or update) datafold dependencies in virtual environment specifed file "requirements.txt".
install_deps: venv
	$(ACTIVATE_VENV); \
	python -m pip install -r requirements.txt

#install_devdeps: @ Install (or update) development dependencies in virtual environment specifed in file "requirements-dev.txt".
install_devdeps: venv
	$(ACTIVATE_VENV); \
	python -m pip install --upgrade pip wheel setuptools twine; \
	python -m pip install -r requirements-dev.txt

#install_docdeps: @ Install non-Python dependencies to build datafold's documentation (requires admin rights or set, DOCKER_ENVIRONMENT=true).
install_docdeps:
ifeq ($(OS),Linux)
ifeq ($(IS_DOCKER),) # true if IS_DOCKER is empty
	@# TODO this could be improved to avoid statement duplication... only sudo changes
	sudo apt-get update \
	&& wget "https://github.com/jgm/pandoc/releases/download/3.1.5/pandoc-3.1.5-1-amd64.deb" \
	&& sudo apt -y --no-install-recommends install ./pandoc-3.1.5-1-amd64.deb \
	&& sudo apt -y --no-install-recommends install $(_DOCDEPS) \
	&& rm -f pandoc-3.1.5-1-amd64.deb;
else # in docker "sudo" is not available and everything is executed with root
	apt-get update \
	&& wget "https://github.com/jgm/pandoc/releases/download/3.1.5/pandoc-3.1.5-1-amd64.deb" \
	&& apt -y --no-install-recommends install ./pandoc-3.1.5-1-amd64.deb \
	&& apt -y --no-install-recommends install $(_DOCDEPS);
endif
else # OS = Windows
	@echo "INFO: Make sure that chocolatery is installed (https://community.chocolatey.org/)."
	@echo "INFO: Make sure to execute with administrator rights."
	choco install $(_DOCDEPS)
endif

#ci_build_image: @ Build a docker image (name: datafold/gitlab-runner) that is based on the gitlab-runner:latest image (see Dockerfile.ci file).
ci_build_image:
	sudo docker build -t datafold/gitlab-runner -f Dockerfile.ci .

#ci_run_container: @ Start the container from the datafold/gitlab-runner image (target ci_build_image).
ci_run_container:
	sudo docker run -it -d --restart always --name gitlab-runner \
	-v /var/run/docker.sock:/var/run/docker.sock \
	-v /srv/gitlab-runner/config:/etc/gitlab-runner \
	datafold/gitlab-runner:latest


#ci_start_gitlab_runner: @ Start the gitlab-runner within the container. Note that this requires a registered gitlab-runner/.
ci_start_gitlab_runner:
	sudo docker exec gitlab-runner gitlab-runner start
	sudo docker exec gitlab-runner gitlab-runner status


#ci_enter: @ Enter the running docker container with an interactive shell.
ci_enter:
	sudo docker exec -it gitlab-runner /bin/bash

#ci_show_runner_config: @ Show the current config file of the gitlab-runner (config file does not exist if the runner was never registered).
ci_show_runner_config:
	sudo docker exec gitlab-runner cat /etc/gitlab-runner/config.toml

#ci_build_and_run: @ Build docker image, start container and start gitlab-runner. If there is no config file (or empty) at '/srv/gitlab-runner/config', then the gitlab-runner needs still to be registered.
ci_build_and_run: ci_build_image ci_run_container ci_start_gitlab_runner

#ci_register_gitlab_runner: @ Register gitlab-runner (note the hints at the beginning). Since this Makefile is within the container, this should be run within the container (see target 'ci_enter').
ci_register_gitlab_runner:
	# TODO: do not register if there is already a config.toml file present?
	@echo "-------------------------------------------------------"
	@echo "-------------------------------------------------------"
	@echo "The gitlab instance is usually https://www.gitlab.com/"
	@echo "The token is obtained by going to the gitlab repo -> Settings -> CI/CD -> Runners -> New Project runner -> Linux -> Tick "Run untagged jobs" -> Create runner"
	@echo "Set 'bash' as executor"
	@echo "-------------------------------------------------------"
	@echo "-------------------------------------------------------"
	gitlab-runner register

#devenv: @ Setup full development environment by executing targets 'install_devdeps' and 'install_docdeps'.
devenv: install_devdeps install_docdeps precommit

#versions: @ Show current version of datafold and versions of essential dependencies.
versions:
	@$(ACTIVATE_VENV); \
	python datafold/_version.py

#docs: @ Build datafold documentation with Sphinx. The documentation is the located at "/doc/source/".
docs:
	@$(ACTIVATE_VENV); \
	echo Execute tutorials environment variable: DATAFOLD_TUTORIALS_EXECUTE=$(DATAFOLD_TUTORIALS_EXECUTE); \
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

#unittest: @ Run unittests with 'pytest' and 'coverage'. A html coverage report is saved to folder './coverage/'.
unittest:
	# run unittests with coverage first and then return exit code at end
    # (otherwise a successful generation of coverage report is returned)
	@$(ACTIVATE_VENV); \
	python -m coverage run --branch -m pytest $(PYTESTOPTS) datafold/;
	EXIT_CODE=$$?
	@$(ACTIVATE_VENV); \
	python -m coverage html -d ./coverage/; \
	python -m coverage report; \
	exit $(EXIT_CODE);

#unittest_last_failed: @ Run only unittests that failed last.
unittest_last_failed:
	# --lf, --last-failed   rerun only the tests that failed at the last run (or all if none failed)
	@$(ACTIVATE_VENV); \
	python -m pytest --lf datafold/;

#tutorialtest: @ Run all tutorials with pytest.
tutorialtest:
	@$(ACTIVATE_VENV); \
	export PYTHONPATH="$(CURRENT_PATH):$$PYTHONPATH"; \
	python -m pytest $(PYTESTOPTS) tutorials/;

test: unittest tutorialtest


#jupyter: @ Open Juypter notebook with correct virtual environment and PYTHONPATH set for datafold source code.
jupyter:
	@$(ACTIVATE_VENV); \
	export PYTHONPATH=$(CURRENT_PATH):$$PYTHONPATH; \
	jupyter notebook

#tutorial: @ Open tutorials in Jupyter notebook (opens in the default web browser).
tutorial:
	@$(ACTIVATE_VENV); \
	export PYTHONPATH=$(CURRENT_PATH):$$PYTHONPATH; \
	python -m notebook $(CURRENT_PATH)/tutorials/

#precommit: @ Run git hooks managed by "pre-commit" to analyze and automatically format the source code.
precommit:
	@$(ACTIVATE_VENV); \
	python -m pre_commit run --all $(GITHOOK);

#ruff_fix: @ Run ruff (installed within pre-commit) with '--fix' option to detect and fix issues (if possible).
ruff_fix:
	@$(ACTIVATE_VENV); \
	python -m pre_commit run ruff-with-fix --hook-stage manual;

#gitamend: @ Amend a commit to the last commit (already pushed).
gitamend:
	@$(ACTIVATE_VENV);\
	git commit --amend;\
	git push --force-with-lease

#ci: @ Run continuous integration pipeline.
ci: install_devdeps test precommit test_install

#build: @ Build a Python source distribution (sdist) and wheel (bdist_wheel).
build:
	@$(ACTIVATE_VENV); \
	python setup.py sdist bdist_wheel; \

#install: @ Install datafold in virtual environment.
install:
	@$(ACTIVATE_VENV); \
	python -m pip install .

#test_install: @ Install and subsequently uninstall datafold for testing. All created files during installation are removed.
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

#clean_install: @ Remove all files that are created for the target "install".
clean_install: clean_build
	@$(ACTIVATE_VENV); \
	yes | pip uninstall datafold

#clean_cache: @ Remove all cache files from the repository.
clean_cache:
	rm -rf .mypy_cache;
	rm -fr .pytest_cache/;
	find . -path '*/__pycache__/*' -delete
	find . -path '*/.ipynb_checkpoints/*' -delete

#clean_docker: @ Remove all unused docker images in docker (not just dangling ones).
clean_docker:
	docker system prune -a

#clean_docker_ci: Clean the container and image from the gitlab-runner. Note that this removes the registered runner.
clean_docker_ci:
	-sudo docker container stop gitlab-runner;
	-sudo docker container rm gitlab-runner;
	-sudo docker rmi datafold/gitlab-runner;

#clean_venv: @ Remove the virtual environment folder which is created in the target "venv".
clean_venv:
	rm -fr $(VENV_DIR);

#clean: @ Call targets "clean_docs", "clean_install", "clean_test", "clean_precommit" and "clean_cache".
clean: clean_docs clean_install clean_test clean_precommit clean_cache
