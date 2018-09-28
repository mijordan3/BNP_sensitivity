# create plots of figure 1: parametric results

# load data

# results where alpha_0 was 2.0
results_matrix_20 <- read.csv('./data_for_figures/init_20_param_sens.csv', header = FALSE)
rownames(results_matrix_20) <- c('alpha', 'refitted', 'linear approx')
results_df_20 <- as.data.frame(t(results_matrix_20))


# results where alpha_0 was 3.5
results_matrix_35 <- read.csv('./data_for_figures/init_35_param_sens.csv', header = FALSE)
rownames(results_matrix_35) <- c('alpha', 'refitted', 'linear approx')
results_df_35 <- as.data.frame(t(results_matrix_35))

# results where alpha_0 was 5.0
results_matrix_50 <- read.csv('./data_for_figures/init_50_param_sens.csv', header = FALSE)
rownames(results_matrix_50) <- c('alpha', 'refitted', 'linear approx')
results_df_50 <- as.data.frame(t(results_matrix_50))


w <- 1.1
grid.arrange(
  plot_parametric_sensitivity(results_df_20, alpha_0 = 2.0), 
  plot_parametric_sensitivity(results_df_35, alpha_0 = 3.5), 
  plot_parametric_sensitivity(results_df_50, alpha_0 = 5.0),
  ncol  = 3)

