library(tidyverse)
library(reshape2)
library(reticulate)
reticulate::use_python("/usr/bin/python3")
py_main <- reticulate::import_main()

reticulate::py_run_string("
import numpy as np
import json_tricks
")

fit_dir <- "/home/rgiordan/Documents/git_repos/BNP_sensitivity/RegressionClustering/fits/"
fit_file <- paste(
  "transformed_gene_regression_df4_degree3_genes700_num_components40_",
  " inflate0.0_shrunkTrue_alphascale0.22508_analysis.json", sep="")

fit_files <- system(sprintf("ls %s/*genes700_*inflate0.0*_analysis.json", fit_dir), intern=TRUE)

analysis_name <- "genes700_inflate0.0"
ResultDictToDF <- function(result_dict) {
  df  <- data.frame(analysis=analysis_name)
  for (key in names(result_dict)) {
    df[[key]] <- result_dict[key]
  }
  return(df)
}


results <- data.frame()
for (fit_file in fit_files) {
  cat("Running for ", fit_file, "\n")
  #py_main$fit_filename <- file.path(fit_dir, fit_file)
  py_main$fit_filename <- fit_file
  reticulate::py_run_string("
with open(fit_filename, 'r') as infile:
  this_result = json_tricks.loads(infile.read())
")
  
  for (i in 1:length(py_main$this_result)) {
    results <- rbind(results, ResultDictToDF(py_main$this_result[[i]]))
  }
}

results <- results %>%
  select(-refit_filename)

ggplot(results) +
  geom_point(aes(x=alpha1, y=e_num1, color="True")) +
  geom_point(aes(x=alpha1, y=e_num_pred, color="predicted")) +
  facet_grid(predictive ~ threshold)
  