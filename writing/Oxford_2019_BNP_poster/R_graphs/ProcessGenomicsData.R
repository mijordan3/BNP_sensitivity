# This file is produced by RegressionClustering/scripts/combine_analyses.R.
genomics_data <- LoadIntoEnvironment(
    file.path(data_path, "genomics_data/combined_results_genes7000.Rdata"))

# Process the data to work with the Iris plotting functions.

use_predictive <- TRUE
use_taylor_order <- 1
use_threshold <- 0

use_inflate <- FALSE

ProcessGenomicsData <- function(results) {
    processed_results <-
      results %>%
      filter(threshold==use_threshold,
             taylor_order==use_taylor_order,
             predictive==use_predictive) %>%
      mutate(alpha=case_when(!is.na(epsilon) ~ epsilon, TRUE ~ alpha1),
             approx=e_num_pred, refitted=e_num1) %>%
      select(alpha, functional, alpha_increase, inflate, approx, refitted) %>%
      gather(key="method", value="e_num_clusters",
             -alpha, -functional, -alpha_increase, -inflate)
    return(processed_results)
}

genomics_data$processed_results <-
    ProcessGenomicsData(
        filter(genomics_data$results, inflate == use_inflate))

genomics_data$processed_pert_df <-
    genomics_data$pert_df %>%
        mutate(p0=exp(log_p0), p1=exp(log_p1)) %>%
        select(v_grid, p0, p1) %>%
        rename(nu_k=v_grid) %>%
        gather("which_prior", "p", -nu_k)
