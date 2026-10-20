# IEEE Transactions manuscript — LaTeX source

Compile on Overleaf (upload this folder as a zip; compiler pdfLaTeX) or locally:

    pdflatex main && bibtex main && pdflatex main && pdflatex main

- `main.tex` — manuscript (IEEEtran journal class v1.8b, included)
- `refs.bib` — references, formatted by `IEEEtran.bst`
- `tables/` — result tables generated from the experiment outputs by `make_results.py`
- `figures/` — vector PDF figures (+ PNG copies); diagrams from `*.dot` (Graphviz), plots from `make_*.py`
- Fill in: author list/affiliations, journal name in `\markboth`, repository URL.
