setwd("/home/rgiordan/Documents/git_repos/BNP_sensitivity/writing/Oxford_2019_BNP_poster/")
knitr_debug <- FALSE
source("R_graphs/Initialize.R")
source("R_graphs/CommonGraphs.R")

genomics_data <- LoadIntoEnvironment(file.path(data_path, "genomics_data/combined_results_genes7000.Rdata"))

# Parametric graph
plot_parametric_sensitivity(
  genomics_data$results %>%
    filter(!inflate, predictive, !alpha_increase, threshold==0, taylor_order==1, !functional) %>%
    mutate(alpha=alpha1, linear_approx=e_num_pred, refitted=e_num1) %>%
    select(alpha, linear_approx, refitted) %>%
    gather(key="method", value="e_num_clusters", -alpha),
  alpha_0=2.0
)

# Functional perturbation graph
plot_parametric_sensitivity(
  genomics_data$results %>%
    filter(!inflate, predictive, threshold==0, taylor_order==1, functional) %>%
    mutate(alpha=epsilon, linear_approx=e_num_pred, refitted=e_num1) %>%
    select(alpha, linear_approx, refitted) %>%
    gather(key="method", value="e_num_clusters", -alpha),
  alpha_0=-1., xlabel=TeX("$\\delta$")
)


head(genomics_data$pert_df)
# Prior perturbation graph
plot_prior_perturbation(
  genomics_data$pert_df %>%
    mutate(p0=exp(log_p0), p1=exp(log_p1)) %>%
    select(v_grid, p0, p1) %>%
    rename(nu_k=v_grid) %>%
    gather("which_prior", "p", -nu_k))




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
  select(alpha, functional, alpha_increase, linear_approx, refitted) %>%
  gather(key="method", value="e_num_clusters", -alpha, -functional, -alpha_increase)
