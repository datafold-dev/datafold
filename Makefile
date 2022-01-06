.DEFAULT_GOAL := help

#Internal variables:
VENV_DIR = .venv
ACTIVATE_VENV = . $(VENV_DIR)/bin/activate
HTML_DOC_PATH = ./doc/build/html/

.PHONY: help venv doc_deps versions docs docs_linkcheck unittests tutorials precommit ci build install uninstall test_install pypi test_pypi clean_docs clean_build clean_dev

#help: @ List available targets in this makefile
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
	@echo 'Arguments associated to the target "unittests".'
	@echo 'PYTESTOPTS - Options passed to "pytest" in target "unittests"'
	@echo ''
	@echo 'Arguments associated to the target "precommit".'
	@echo 'GITHOOK - Execute specific githook. Defaults to "--all".'
	@echo ''
	@echo 'Available targets:'
	@grep -E '[a-zA-Z\.\-]+:.*?@ .*$$' $(MAKEFILE_LIST)| tr -d '#'  | awk 'BEGIN {FS = ":.*?@ "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

PYTHON ?= python3
SPHINXOPTS    ?=
PYTESTOPTS    ?=
EXECUTE_TUTORIAL ?= never

OUTPUT_DOCS ?= doc/build/
OPEN_BROWSER ?= true
HTML_FILE_OPEN ?= index.html
URL_DOC_OPEN = $(HTML_DOC_PATH)$(HTML_FILE_OPEN)

GITHOOK = --all

#venv: @ Create virtual environment if it does not exist yet.
venv:
	test -d $(VENV_DIR) || $(PYTHON) -m venv $(VENV_DIR);

#doc_deps: @ Install required dependencies to render the documentation.
doc_deps:
	# TODO: Required latex packages are missing here!
ifeq ($(shell apt version libjs-mathjax),)
	sudo apt-get install libjs-mathjax;
endif
ifeq ($(shell apt version fonts-mathjax),)
	sudo apt-get install fonts-mathjax
endif
ifeq ($(shell apt version dvipng),)
	sudo apt-get install dvipng
endif
ifeq ($(shell apt version pandoc),)
	sudo apt-get install pandoc
endif
ifeq ($(shell apt version graphviz),)
	sudo apt-get install graphviz
endif

#develop @ Create development environment
develop: venv
	$(ACTIVATE_VENV); \
	python -m pip install -r requirements-dev.txt; \
	python -m pre_commit install --install-hooks;

#update_deps: @ Update dependencies in virtual environment according to file "requirements-dev.txt".
update_deps: develop
	$(ACTIVATE_VENV); \
    python -m pip install --upgrade pip wheel setuptools twine; \
	python -m pip install -r requirements-dev.txt;

#versions: @ Print datafold version and essential versions of dependencies.
versions:
	$(ACTIVATE_VENV); \
	python datafold/_version.py

#docs: @ Build datafold documentation with Sphinx.
docs:
	$(ACTIVATE_VENV); \
	python -m sphinx -M html doc/source/ $(OUTPUT_DOCS) $(SPHINXOPTS) $(O);
	@# Open the browser at the page specified in URL_DOC_OPEN
ifeq ($(OPEN_BROWSER),true)
	@ if which xdg-open > /dev/null; then \
		xdg-open $(URL_DOC_OPEN); \
	elif which gnome-open > /dev/null; then \
		gnome-open $(URL_DOC_OPEN); \
	fi
endif

#docs_linkcheck: @ Check if all links in the documentation are valid.
docs_linkcheck:
	. $(VENV_DIR)/bin/activate; \
	cd doc/; \
	python -m sphinx -M linkcheck source/ build/ $(SPHINXOPTS) $(O)

#unittests: @ Run and report all unittests with pytest.
unittests:
	$(ACTIVATE_VENV); \
	python -m coverage run -m pytest $(PYTESTOPTS) datafold/; \
	python -m coverage html -d ./coverage/; \
	python -m coverage report;

#tutorials: @ Run all tutorials with pytest.
tutorials:
	$(ACTIVATE_VENV); \
	export PYTHONPATH=`pwd`; \
	python -m pytest tutorials/;

test: unittests tutorials

#precommit: @ Run git hooks to check and analyze the code:
precommit:
	$(ACTIVATE_VENV); \
	python -m pre_commit run $(GITHOOK);

#ci: @ Run entire continuous integration pipeline.
ci: update_deps test precommit

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

#test_install: @ Install and subsequently uninstall datafold for testing purposes (all output is cleaned afterwards).
test_install: install uninstall clean_build
	@echo 'Successful'

#pypi: @ Upload and release to PyPI..
pypi: build
	$(ACTIVATE_VENV); \
	python setup.py sdist bdist_wheel; \
	python -m twine check dist/*; \
	python -m twine upload --verbose dist/*

#pypi_test: @ Make a test upload to PyPI.
test_pypi: build
	$(ACTIVATE_VENV); \
	python -m twine check dist/*; \
	python -m twine upload --verbose --repository testpypi dist/*

#clean_docs: @ Clean all files that are created for the documentation.
clean_docs:
	cd doc/; \
	rm -fr build/; \
	rm -f source/api/*.rst source/_apidoc/*.rst

#clean_build: @ Clean all files that are created for the build.
clean_build:
	rm -fr build/;
	rm -fr dist/;
	rm -fr datafold.egg-info/;

#clean: @ Clean all files associated to development.
clean: clean_docs clean_build
	rm -fr .venv/;
	rm -f .coverage;
	rm -fr coverage/;
	find datafold/ -name __pycache__ -type d -exec rm -rf {} \;
	find . -name .ipynb_checkpoints -type d -exec rm -rf {} \;
