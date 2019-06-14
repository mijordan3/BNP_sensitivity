library(tidyverse)
library(reshape2)
library(reticulate)
library(gridExtra)

reticulate::use_python("/usr/bin/python3")
py_main <- reticulate::import_main()

reticulate::py_run_string("
import numpy as np
import json_tricks
")

fit_dir <- "/home/rgiordan/Documents/git_repos/BNP_sensitivity/RegressionClustering/fits/"

genes <- "700"

results <- data.frame()

for (inflate in c("0.0", "1.0")) {
  fit_files <- system(sprintf("ls %s/*genes%s_*inflate%s*_analysis.json", fit_dir, genes, inflate), intern=TRUE)
  analysis_name <- sprintf("genes%s_inflate%s", genes, inflate)
  
  ResultDictToDF <- function(result_dict) {
    df  <- data.frame(analysis=analysis_name)
    for (key in names(result_dict)) {
      df[[key]] <- result_dict[key]
    }
    return(df)
  }
  
  for (fit_file in fit_files) {
    cat("Running for ", fit_file, "\n")
    #py_main$fit_filename <- file.path(fit_dir, fit_file)
    py_main$fit_filename <- fit_file
    reticulate::py_run_string("
with open(fit_filename, 'r') as infile:
  this_result = json_tricks.loads(infile.read())
")
    
    for (i in 1:length(py_main$this_result)) {
      results <- rbind(
        results,
        ResultDictToDF(py_main$this_result[[i]]) %>%
          mutate(inflate=inflate != "0.0"))
    }
  }
}

results <- results %>%
  select(-refit_filename) %>%
  mutate(alpha_increase=alpha1 > alpha0)

table(results$inflate)

MakePlot <- function(use_alpha_increase, use_predictive, use_inflate) {
  ggplot(filter(results,
                alpha_increase==use_alpha_increase,
                predictive==use_predictive,
                inflate==use_inflate)) +
    geom_point(aes(x=alpha1, y=e_num1, color="Refitting")) +
    geom_point(aes(x=alpha1, y=e_num_pred, color="Approximation")) +
    geom_line(aes(x=alpha1, y=e_num1, color="Refitting")) +
    geom_line(aes(x=alpha1, y=e_num_pred, color="Approximation")) +
    geom_vline(aes(xintercept=alpha0)) +
    facet_grid(threshold ~ ., scales="free", labeller = label_context) +
    ggtitle(sprintf("%s\n%s\n%s",
            ifelse(use_inflate, "Noise added", "No noise added"),
            ifelse(use_alpha_increase, "Increasing alpha", "Decreasing alpha"),
            ifelse(use_predictive, "Predictive", "In-sample"))) +
    xlab("Alpha") + ylab("Expected number of clusters")
}

use_predictive <- TRUE
grid.arrange(
  MakePlot(FALSE, use_predictive, FALSE),
  MakePlot(FALSE, use_predictive, TRUE),
  MakePlot(TRUE, use_predictive, FALSE),
  MakePlot(TRUE, use_predictive, TRUE),
  ncol=4
)
