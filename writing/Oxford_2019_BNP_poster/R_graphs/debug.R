setwd("/home/rgiordan/Documents/git_repos/BNP_sensitivity/writing/Oxford_2019_BNP_poster/")
knitr_debug <- FALSE
source("R_graphs/Initialize.R")
source("R_graphs/CommonGraphs.R")

source("R_graphs/ProcessGenomicsData.R")

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




gene_pred_pert <- filter(genomics_data$processed_results, functional, !inflate)


plot_parametric_sensitivity(gene_pred_pert, xlabel=TeX("$\\delta$")) +
  geom_text(aes(x=0.1, y=59.68, label="lines are overplotted"))
