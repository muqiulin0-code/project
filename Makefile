PYTHON ?= .venv/bin/python
CMAKE ?= cmake

.PHONY: setup test cpp check train detect rotation

setup:
	python3 -m venv .venv
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt
	$(PYTHON) -m pip install -e . --no-deps

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
