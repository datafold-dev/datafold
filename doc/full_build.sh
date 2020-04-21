#!/bin/bash

source ../venv/bin/activate

# ./.. points to the datafold package and fetches then the datafold package with all modules contained
# documentation: https://www.sphinx-doc.org/en/master/man/sphinx-apidoc.html
# use -f to force build all modules
#sphinx-autogen -o ./source/_apidoc ./.. setup.py
#sphinx-apidoc -o ./source/_apidoc ./.. setup.py

export DATAFOLD_NBSPHINX_EXECUTE="never"

# create html of the documentation
make html

# Open the start page of generated documentation (Linux only!)
URL="./build/html/index.html"
if which xdg-open > /dev/null
then
  xdg-open $URL
elif which gnome-open > /dev/null
then
  gnome-open $URL
fi
