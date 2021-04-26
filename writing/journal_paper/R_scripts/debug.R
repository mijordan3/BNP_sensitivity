git_repo_loc <- "/home/rgiordan/Documents/git_repos/BNP_sensitivity"
paper_directory <- file.path(git_repo_loc, "writing/journal_paper")
setwd(paper_directory)
knitr_debug <- TRUE # Set to true to see error output
simple_cache <- FALSE # Set to true to cache knitr output for this analysis.
source("R_scripts/initialize.R", echo=FALSE)
source("R_scripts/plotting_utils.R")

#source('./R_scripts/simple_examples/pathological_r2.R', echo=knitr_debug, print.eval=TRUE)

source('./R_scripts/simple_examples/positive_pert.R', echo=knitr_debug, print.eval=TRUE)
