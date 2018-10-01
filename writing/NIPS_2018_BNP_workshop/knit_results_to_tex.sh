#!/bin/bash

Rscript -e 'library(knitr); knit("results_alpha.Rnw")'
Rscript -e 'library(knitr); knit("results_functions.Rnw")'
