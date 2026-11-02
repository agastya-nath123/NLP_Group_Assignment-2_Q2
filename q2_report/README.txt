Q2 report (LaTeX)

Compile:  pdflatex report.tex   (run it twice so the table/figure numbers resolve)
Overleaf: New Project -> Upload Project -> this zip; the default pdfLaTeX compiler works.

Files
  report.tex              the report
  explain_snippet.tex     coloured words for Figure 4 (generated, \input by report.tex)
  figures/                Figures 1-3 (generated)
  make_figures.py         redraws Figures 1-3 from outputs/*/history.json and metrics.json
  extract_explain_colors.py  rebuilds Figure 4's colours from results/explain/*.png
To regenerate the figures, put this folder inside the project root (next to outputs/ and results/).
