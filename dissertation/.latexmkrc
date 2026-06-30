# .latexmkrc — reproducible build for the dissertation
#
#   latexmk -pdf main.tex   # full build: pdflatex + biber + makeglossaries
#   latexmk -c              # clean aux files (keeps main.pdf)
#   latexmk -C              # clean everything including main.pdf
#
# main.tex uses biblatex/biber AND the glossaries package. A plain
# `latexmk -pdf` runs biber automatically but does NOT run makeglossaries,
# so the List of Acronyms would come out empty. The custom dependency
# below wires makeglossaries into the build.

$pdf_mode = 1;        # compile with pdflatex
$bibtex_use = 2;      # run biber/bibtex and clean its output on -c

# --- glossaries / acronyms support ---------------------------------
add_cus_dep('acn', 'acr', 0, 'run_makeglossaries');
add_cus_dep('glo', 'gls', 0, 'run_makeglossaries');

sub run_makeglossaries {
    my ($base_name, $path) = fileparse($_[0]);
    pushd $path;
    my $return = system "makeglossaries", $base_name;
    popd;
    return $return;
}

# Let `latexmk -c` clean glossary + biber leftovers too.
push @generated_exts, 'glo', 'gls', 'glg', 'acn', 'acr', 'alg';
$clean_ext .= ' %R.ist %R.xdy %R.bbl %R.run.xml %R.glsdefs';
