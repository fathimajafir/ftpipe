PY := python3
export PYTHONPATH := src

.PHONY: help install test demo validate build clean

help:
	@echo "make install   - editable install (optional; PYTHONPATH alone works too)"
	@echo "make test      - run the unit tests"
	@echo "make demo      - offline end-to-end run with the mock provider"
	@echo "make validate  - check datasets against the real config"
	@echo "make build     - export per-provider training files"
	@echo "make clean     - remove build/ runs/ caches"

install:
	$(PY) -m pip install -e .

test:
	$(PY) -m unittest discover -s tests -v

demo:
	$(PY) -m ftpipe run -c configs/demo.json

validate:
	$(PY) -m ftpipe validate -c configs/pipeline.json

build:
	$(PY) -m ftpipe build -c configs/pipeline.json --sample

clean:
	rm -rf build runs .pytest_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
