git_repo_loc <- "/home/rgiordan/Documents/git_repos/BNP_sensitivity"
paper_directory <- file.path(git_repo_loc, "writing/journal_paper")
setwd(paper_directory)
knitr_debug <- TRUE # Set to true to see error output
simple_cache <- FALSE # Set to true to cache knitr output for this analysis.
r_script_lib <- file.path(paper_directory, "R_scripts")
r_script_dir <- file.path(paper_directory, "R_scripts")
source("R_scripts/initialize.R", echo=FALSE)
source("R_scripts/plotting_utils.R")

#source('./R_scripts/simple_examples/pathological_r2.R', echo=knitr_debug, print.eval=TRUE)
#source('./R_scripts/simple_examples/positive_pert.R', echo=knitr_debug, print.eval=TRUE)


source('./R_scripts/simple_examples/load_data.R', echo=knitr_debug, print.eval=TRUE)

#source('./R_scripts/simple_examples/function_paths.R', echo=knitr_debug, print.eval=TRUE)
#source('./R_scripts/simple_examples/function_paths_mult.R', echo=knitr_debug, print.eval=TRUE)
#source('./R_scripts/simple_examples/function_paths_lin.R', echo=knitr_debug, print.eval=TRUE)
#source('./R_scripts/simple_examples/function_ball.R', echo=knitr_debug, print.eval=TRUE)
source('./R_scripts/simple_examples/distant_distributions.R', echo=knitr_debug, print.eval=TRUE)
