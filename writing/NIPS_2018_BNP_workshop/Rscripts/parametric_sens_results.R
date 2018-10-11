# create plots of figure 1: parametric results

# load data

# results where alpha_0 was 3.0
results_matrix_20 <- read.csv('./data_for_figures/init_alpha3_thresh0_param_sens.csv', header = FALSE)
rownames(results_matrix_20) <- c('alpha', 'refitted', 'linear approx')
results_df_20 <- as.data.frame(t(results_matrix_20))


# results where alpha_0 was 8.0
results_matrix_35 <- read.csv('./data_for_figures/init_alpha8_thresh0_param_sens.csv', header = FALSE)
rownames(results_matrix_35) <- c('alpha', 'refitted', 'linear approx')
results_df_35 <- as.data.frame(t(results_matrix_35))

# results where alpha_0 was 13
results_matrix_50 <- read.csv('./data_for_figures/init_alpha13_thresh0_param_sens.csv', header = FALSE)
rownames(results_matrix_50) <- c('alpha', 'refitted', 'linear approx')
results_df_50 <- as.data.frame(t(results_matrix_50))


w <- 1.1
grid.arrange(
  plot_parametric_sensitivity(results_df_20, alpha_0 = 3.0), 
  plot_parametric_sensitivity(results_df_35, alpha_0 = 8.0), 
  plot_parametric_sensitivity(results_df_50, alpha_0 = 13.0),
  ncol  = 3)

