#!/bin/bash

# Build sdist and wheel using the system Python
python -m pip install -U pip
python -m pip install build
python -m build

# Check sdist install and imports using the same Python version
mkdir -p test-sdist
cd test-sdist
python -m venv venv-sdist
venv-sdist/bin/python -m pip install --upgrade pip
venv-sdist/bin/python -m pip install ../dist/langextract_outlines-*.tar.gz
venv-sdist/bin/python -c "import langextract_outlines"
cd ..

# Check wheel install and imports using the same Python version
mkdir -p test-wheel
cd test-wheel
python -m venv venv-wheel
venv-wheel/bin/python -m pip install --upgrade pip
venv-wheel/bin/python -m pip install ../dist/langextract_outlines-*.whl
venv-wheel/bin/python -c "import langextract_outlines"
cd ..