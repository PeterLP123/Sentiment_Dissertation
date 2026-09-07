FINAL_DIR := submission/news-sentiment-beyond-mean

.DEFAULT_GOAL := help

.PHONY: help setup validate manuscript benchmark-setup benchmark-check

help:
	@echo "Project commands:"
	@echo "  make setup            Install the locked final-submission development environment"
	@echo "  make validate         Check the root project and validate/lint the final package"
	@echo "  make manuscript       Rebuild the final manuscript (manuscript/main.pdf)"
	@echo "  make benchmark-setup  Install the locked historical benchmark environment"
	@echo "  make benchmark-check  Validate the public benchmark and run focused tests"

setup:
	uv sync --project "$(FINAL_DIR)" --locked --extra dev

validate:
	python3 scripts/check_project.py
	uv run --project "$(FINAL_DIR)" --locked --extra dev make -C "$(FINAL_DIR)" validate lint

manuscript:
	@echo "Rebuilding $(FINAL_DIR)/manuscript/main.pdf; the named submission PDF remains the preserved snapshot."
	uv run --project "$(FINAL_DIR)" --locked --extra dev make -C "$(FINAL_DIR)" manuscript

benchmark-setup:
	uv sync --project . --locked --extra dev

benchmark-check:
	uv run --project . --locked --extra dev sentiment-bench validate-data
	uv run --project . --locked --extra dev python -m pytest -q tests/test_dataset.py tests/test_statistics.py tests/test_project_validation.py
	uv run --project . --locked --extra dev python -m ruff check scripts/check_project.py tests/test_project_validation.py
