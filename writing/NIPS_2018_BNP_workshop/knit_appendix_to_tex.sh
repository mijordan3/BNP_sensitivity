#!/bin/bash

Rscript -e 'library(knitr); knit("appendix_alpha.Rnw")'
Rscript -e 'library(knitr); knit("appendix_functions.Rnw")'

