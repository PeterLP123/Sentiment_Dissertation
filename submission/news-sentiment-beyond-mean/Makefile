.PHONY: test lint prose artifacts artifact-reproducibility validate wordcount manuscript clean-manuscript

test:
	python -m pytest -q

lint:
	python -m ruff check experiments/lib tests scripts

prose:
	python scripts/check_prose_style.py

artifacts:
	env MPLCONFIGDIR=experiments/generated/.matplotlib python -m scripts.generate_dissertation_artifacts --output-dir manuscript/artifacts

artifact-reproducibility:
	python scripts/check_artifact_reproducibility.py

validate:
	python scripts/validate_repository.py
	python scripts/check_prose_style.py
	python -m pytest -q
	python scripts/check_artifact_reproducibility.py

wordcount:
	cd manuscript && texcount -inc -sum -brief main.tex

manuscript: artifacts prose
	python scripts/svg_to_pdf.py manuscript/ucl_logo.svg manuscript/ucl_logo.pdf
	cd manuscript && latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex

clean-manuscript:
	cd manuscript && latexmk -C main.tex
	rm -f manuscript/ucl_logo.pdf
