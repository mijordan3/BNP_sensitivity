# Makes figure 2: functional sensitivity results

###################################
# Plots of prior perturbations

# the first one
prior_pert1 <- read.csv('./data_for_figures/prior_pert1.csv',
                sep = ',', header = FALSE)
prior_pert1_df <- as.data.frame(t(prior_pert1))
colnames(prior_pert1_df) <- c('nu_k', 'p0', 'pc')

prior_pert1_plot <-
  prior_pert1_df %>%
  gather(which_prior, p, -nu_k) %>%
  ggplot() + geom_line(aes(x = nu_k, y = p, color = which_prior)) +
    theme(legend.position = c(0.75, 0.75), legend.title=element_blank()) +
    xlab(TeX("$\\nu_k$")) + ylab(TeX("$p(\\nu_k)$")) + 
  scale_color_manual(values=c("red", "blue"))
# prior_pert1_plot

# the second one
prior_pert2 <- read.csv('./data_for_figures/prior_pert2.csv',
                        sep = ',', header = FALSE)
prior_pert2_df <- as.data.frame(t(prior_pert2))
colnames(prior_pert2_df) <- c('nu_k', 'p0', 'pc')

prior_pert2_plot <-
  prior_pert2_df %>% gather(which_prior, p, -nu_k) %>%
  ggplot() + geom_line(aes(x = nu_k, y = p, color = which_prior)) +
      theme(legend.position = c(0.75, 0.75), legend.title=element_blank()) +
      xlab(TeX("$\\nu_k$")) + ylab(TeX("$p(\\nu_k)$")) + 
      scale_color_manual(values=c("red", "blue"))
# prior_pert2_plot

##########################
# Plot results

# plot results from first perturbation
results_matrix_prior_pert1 <-
  read.csv('./data_for_figures/prior_pert1_enum_clust_results.csv', header = FALSE)
results_df_prior_pert1 <- as.data.frame(t(results_matrix_prior_pert1))
colnames(results_df_prior_pert1) <- c('alpha', 'refitted', 'linear approx')

prior_pert1_results_plot <-
  plot_parametric_sensitivity(
    results_df_prior_pert1, alpha_0 = -1, xlabel=TeX("$\\delta$")) +
  theme(legend.position = c(0.8, 0.75), legend.title=element_blank())
# prior_pert1_results_plot

# plot results from second perturbation
results_matrix_prior_pert2 <-
  read.csv('./data_for_figures/prior_pert2_enum_clust_results.csv', header = FALSE)
results_df_prior_pert2 <- as.data.frame(t(results_matrix_prior_pert2))
colnames(results_df_prior_pert2) <- c('alpha', 'refitted', 'linear approx')

prior_pert2_results_plot <-
  plot_parametric_sensitivity(
    results_df_prior_pert2, alpha_0 = -1, xlabel=TeX("$\\delta$")) +
  theme(legend.position = c(0.15, 0.75), legend.title=element_blank())
# prior_pert2_results_plot

grid.arrange(
  # prior_pert1_plot,
  # prior_pert1_results_plot,
  prior_pert2_plot,
  prior_pert2_results_plot,
  ncol  = 2)

