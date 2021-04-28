#!/bin/bash

# Rscript -e 'library(knitr); knit("figs.Rnw")'

Rscript -e 'library(knitr); knit("iris_figs.Rnw")'

Rscript -e 'library(knitr); knit("structure_figs.Rnw")'
