# create plots of figure 1: parametric results

# load data

# results where alpha_0 was 3.0
results_matrix_1 <- read.csv('./data_for_figures/param_sens_init_alpha3_thresh0_e_num_clusters.csv', header = FALSE)
rownames(results_matrix_1) <- c('alpha', 'refitted', 'linear approx')
results_df_1 <- as.data.frame(t(results_matrix_1))


# results where alpha_0 was 8.0
results_matrix_2 <- read.csv('./data_for_figures/param_sens_init_alpha8_thresh0_e_num_clusters.csv', header = FALSE)
rownames(results_matrix_2) <- c('alpha', 'refitted', 'linear approx')
results_df_2 <- as.data.frame(t(results_matrix_2))

# results where alpha_0 was 13
results_matrix_3 <- read.csv('./data_for_figures/param_sens_init_alpha13_thresh0_e_num_clusters.csv', header = FALSE)
rownames(results_matrix_3) <- c('alpha', 'refitted', 'linear approx')
results_df_3 <- as.data.frame(t(results_matrix_3))

w <- 1.1
grid.arrange(
  plot_parametric_sensitivity(results_df_1, alpha_0 = 3.0), 
  plot_parametric_sensitivity(results_df_2, alpha_0 = 8.0), 
  plot_parametric_sensitivity(results_df_3, alpha_0 = 13.0),
  ncol  = 3)

