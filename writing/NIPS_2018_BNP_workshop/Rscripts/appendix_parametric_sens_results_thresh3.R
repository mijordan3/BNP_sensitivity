# parametric sensitivity results for threshold = 3

####################
# IN-SAMPLE PLOTS
####################

# results where alpha_0 was 3.0
results_matrix_1 <- read.csv('./data_for_figures/param_sens_init_alpha3_thresh3_e_num_clusters.csv', header = FALSE)
rownames(results_matrix_1) <- c('alpha', 'refitted', 'linear approx')
results_df_1 <- as.data.frame(t(results_matrix_1))


# results where alpha_0 was 8.0
results_matrix_2 <- read.csv('./data_for_figures/param_sens_init_alpha8_thresh3_e_num_clusters.csv', header = FALSE)
rownames(results_matrix_2) <- c('alpha', 'refitted', 'linear approx')
results_df_2 <- as.data.frame(t(results_matrix_2))

# results where alpha_0 was 13
results_matrix_3 <- read.csv('./data_for_figures/param_sens_init_alpha13_thresh3_e_num_clusters.csv', header = FALSE)
rownames(results_matrix_3) <- c('alpha', 'refitted', 'linear approx')
results_df_3 <- as.data.frame(t(results_matrix_3))

w <- 1.1
grid.arrange(
  plot_parametric_sensitivity(results_df_1, alpha_0 = 3.0) + 
    ggtitle('In-sample quantity, t = 3') + 
    theme(legend.position = c(0.25, 0.8), legend.title=element_blank()), 
  plot_parametric_sensitivity(results_df_2, alpha_0 = 8.0) + 
    ggtitle(' ') + 
    theme(legend.position="none"), 
  plot_parametric_sensitivity(results_df_3, alpha_0 = 13.0) + 
    ggtitle(' ') + 
    theme(legend.position="none"),
  ncol  = 3)

####################
# PREDICTIVE PLOTS
####################
# results where alpha_0 was 3.0
results_matrix_1 <- read.csv('./data_for_figures/param_sens_init_alpha3_thresh3_e_num_clusters_pred.csv', 
                             header = FALSE)
rownames(results_matrix_1) <- c('alpha', 'refitted', 'linear approx')
results_df_1 <- as.data.frame(t(results_matrix_1))


# results where alpha_0 was 8.0
results_matrix_2 <- read.csv('./data_for_figures/param_sens_init_alpha8_thresh3_e_num_clusters_pred.csv', 
                             header = FALSE)
rownames(results_matrix_2) <- c('alpha', 'refitted', 'linear approx')
results_df_2 <- as.data.frame(t(results_matrix_2))

# results where alpha_0 was 13
results_matrix_3 <- read.csv('./data_for_figures/param_sens_init_alpha13_thresh3_e_num_clusters_pred.csv', 
                             header = FALSE)
rownames(results_matrix_3) <- c('alpha', 'refitted', 'linear approx')
results_df_3 <- as.data.frame(t(results_matrix_3))

w <- 1.1
grid.arrange(
  plot_parametric_sensitivity(results_df_1, alpha_0 = 3.0) + 
    ggtitle('Preditive quantity, t = 3') + 
    theme(legend.position = c(0.25, 0.8), legend.title=element_blank()), 
  plot_parametric_sensitivity(results_df_2, alpha_0 = 8.0) + 
    ggtitle(' ') + 
    theme(legend.position="none"), 
  plot_parametric_sensitivity(results_df_3, alpha_0 = 13.0) + 
    ggtitle(' ') + 
    theme(legend.position="none"),
  ncol  = 3)

