library(tidyverse)
library(latex2exp)

setwd(paste("/home/rgiordan/Documents/git_repos/BNP_sensitivity/",
            "writing/Oxford_2019_BNP_poster/R_graphs/",
            "data/iris_data/",
            sep=""))

# Load the perturbations

GetPriorPertDF <- function(csv_filename) {
  prior_pert <- 
    read.csv(csv_filename, sep = ',', header = FALSE) %>%
    t() %>%
    as.data.frame()
  colnames(prior_pert) <- c('nu_k', 'p0', 'p1')
  prior_pert$filename <- csv_filename
  return(prior_pert)
}

prior_pert_df <-
  bind_rows(GetPriorPertDF('prior_pert1.csv'),
            GetPriorPertDF('prior_pert2.csv')) %>%
  gather(which_prior, p, -nu_k, -filename)

if (FALSE) {
  # Check that the old graphs work.
  head(prior_pert_df)
  filter(prior_pert_df, filename == "prior_pert1.csv") %>%
    ggplot() +
    geom_line(aes(x = nu_k, y = p, color = which_prior)) +
    theme(legend.position = c(0.85, 0.80), legend.title=element_blank()) +
    xlab(TeX("$\\nu_k$")) + ylab(TeX("$p(\\nu_k)$")) + 
    scale_color_manual(values=c("red", "blue"))

}

# Load the results

GetResultsDF <- function(csv_filename, pred, pert) {
  results_df <-
    read.csv(csv_filename, header = FALSE) %>%
    t() %>%
    as.data.frame()
    colnames(results_df) <- c('alpha', 'refitted', 'linear approx')
    results_df <-
      results_df %>%
      gather(method, e_num_clusters, -alpha) %>%
      mutate(pred=pred, pert=pert)
  return(results_df)
}

results_df <- bind_rows(
  GetResultsDF('prior_pert1_enum_clust_results_thresh0.csv',      pred=FALSE, pert="1"),
  GetResultsDF('prior_pert2_enum_clust_results_thresh0.csv',      pred=FALSE, pert="2"),
  
  GetResultsDF('prior_pert1_enum_clust_results_pred_thresh0.csv', pred=TRUE, pert="1"),
  GetResultsDF('prior_pert2_enum_clust_results_pred_thresh0.csv', pred=TRUE, pert="2"),
  
  GetResultsDF('param_sens_init_alpha3_thresh0_e_num_clusters.csv', pred=FALSE, pert="alpha3"),
  GetResultsDF('param_sens_init_alpha8_thresh0_e_num_clusters.csv', pred=FALSE, pert="alpha8"),
  GetResultsDF('param_sens_init_alpha13_thresh0_e_num_clusters.csv', pred=FALSE, pert="alpha13"),

  GetResultsDF('param_sens_init_alpha3_thresh0_e_num_clusters.csv', pred=TRUE, pert="alpha3"),
  GetResultsDF('param_sens_init_alpha8_thresh0_e_num_clusters.csv', pred=TRUE, pert="alpha8"),
  GetResultsDF('param_sens_init_alpha13_thresh0_e_num_clusters.csv', pred=TRUE, pert="alpha13")
)



if (FALSE) {
  # Make sure the graphs work
  alpha_0 <- 3.0
  xlabel <- TeX('$\\alpha$')
  results_df_long <- results_df %>% filter(!pred, pert=="alpha3")
  ggplot(results_df_long) +
    geom_point(aes(x = alpha, y = e_num_clusters, color = method)) +
    geom_line(aes(x = alpha, y = e_num_clusters, color = method)) +
    xlab(xlabel) + ylab('E[# clusters]') +
    theme(legend.position = c(0.75, 0.2)) +
    geom_vline(xintercept = alpha_0, color = 'blue', linetype = 'dashed')
}


save(prior_pert_df, results_df, file="iris_data_for_knitr.Rdata")
