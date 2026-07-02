# Dissertation build helpers. Run from dissertation/ (or `make -C dissertation ...`).

WORD_BUDGET := 10000

.PHONY: pdf assets count clean distclean

pdf:            ## full build: pdflatex + biber + makeglossaries
	latexmk -pdf main.tex

assets:         ## pull registered-run tables/figures into generated/
	python ../scripts/refresh_dissertation_assets.py

count:          ## words per chapter and total vs the $(WORD_BUDGET)-word budget
	@texcount -brief -sum chapters/*.tex frontmatter/*.tex | sed 's/ File(s)//'
	@total=$$(texcount -brief -sum -total chapters/*.tex | awk -F: '{print $$1}' | tr -d ' '); \
	echo "----"; \
	echo "chapters total: $$total / $(WORD_BUDGET) words"

clean:          ## remove aux files, keep the PDF
	latexmk -c

distclean:      ## remove everything latexmk generated, including the PDF
	latexmk -C
