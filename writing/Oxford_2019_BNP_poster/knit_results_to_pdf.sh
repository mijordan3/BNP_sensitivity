#!/bin/bash

Rscript -e 'library(knitr); knit("bnp2019_poster.rnw")'
pdflatex bnp2019_poster.tex
