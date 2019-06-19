# This file is produced by RegressionClustering/scripts/combine_analyses.R.
genomics_data <- LoadIntoEnvironment(
    file.path(data_path, "genomics_data/combined_results_genes7000.Rdata"))

# Process the data to work with the Iris plotting functions.

use_predictive <- TRUE
use_taylor_order <- 1
use_threshold <- 0

genomics_data$processed_results <-
  genomics_data$results %>%
  filter(predictive==use_predictive,
         threshold==use_threshold,
         taylor_order==use_taylor_order) %>%
  mutate(alpha=case_when(!is.na(epsilon) ~ epsilon, TRUE ~ alpha1),
         linear_approx=e_num_pred, refitted=e_num1) %>%
  select(alpha, functional, alpha_increase, inflate, linear_approx, refitted) %>%
  gather(key="method", value="e_num_clusters",
         -alpha, -functional, -alpha_increase, -inflate)

genomics_data$processed_pert_df <-
    genomics_data$pert_df %>%
        mutate(p0=exp(log_p0), p1=exp(log_p1)) %>%
        select(v_grid, p0, p1) %>%
        rename(nu_k=v_grid) %>%
        gather("which_prior", "p", -nu_k)
