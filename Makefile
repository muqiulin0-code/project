CONDA_ENV ?= pytorch
PYTHON ?= conda run -n $(CONDA_ENV) python
CMAKE ?= cmake

.PHONY: setup env test cpp check train detect rotation

setup:
	$(PYTHON) -m pip install -e . --no-deps

env:
	$(PYTHON) scripts/check_environment.py

test:
	$(PYTHON) -m pytest

cpp:
	$(CMAKE) -S cpp -B build/cpp -DCMAKE_BUILD_TYPE=Release
	$(CMAKE) --build build/cpp --parallel
	ctest --test-dir build/cpp --output-on-failure

check: test cpp

train:
	$(PYTHON) -m cifar10_classifier.train --output-dir artifacts/cifar10

detect:
	$(PYTHON) -m opencv_detection --source 0 --mode face

rotation:
	$(PYTHON) -m rotation_conversions --axis 0 0 1 --angle 90
