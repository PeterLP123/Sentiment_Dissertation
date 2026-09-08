FINAL_DIR := submission/news-sentiment-beyond-mean
BENCHMARK_RUN := uv run --project . --locked --extra dev --extra figures

.DEFAULT_GOAL := help

.PHONY: help setup validate manuscript benchmark-setup benchmark-check benchmark-test benchmark-lint benchmark-types

help:
	@echo "Project commands:"
	@echo "  make setup            Install the locked final-submission development environment"
	@echo "  make validate         Check the root project and validate/lint the final package"
	@echo "  make manuscript       Rebuild the final manuscript (manuscript/main.pdf)"
	@echo "  make benchmark-setup  Install the locked historical benchmark environment"
	@echo "  make benchmark-check  Validate benchmark data, lint, type-check and run the full root test suite"
	@echo "  make benchmark-test   Run root tests (optional model integrations may skip)"
	@echo "  make benchmark-lint   Lint src/, tests/ and scripts/"
	@echo "  make benchmark-types  Type-check the benchmark library"

setup:
	uv sync --project "$(FINAL_DIR)" --locked --extra dev

validate:
	python3 scripts/check_project.py
	uv run --project "$(FINAL_DIR)" --locked --extra dev make -C "$(FINAL_DIR)" validate lint

manuscript:
	@echo "Rebuilding $(FINAL_DIR)/manuscript/main.pdf; the named portfolio PDF is not overwritten."
	uv run --project "$(FINAL_DIR)" --locked --extra dev make -C "$(FINAL_DIR)" manuscript

benchmark-setup:
	uv sync --project . --locked --extra dev --extra figures

benchmark-check:
	$(BENCHMARK_RUN) sentiment-bench validate-data
	$(MAKE) benchmark-lint benchmark-types benchmark-test

benchmark-test:
	$(BENCHMARK_RUN) python -m pytest -q -ra

benchmark-lint:
	$(BENCHMARK_RUN) python -m ruff check src tests scripts

benchmark-types:
	$(BENCHMARK_RUN) python -m mypy src/sentiment_benchmark
