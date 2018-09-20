# load data
library(tidyverse)
source('./Rscripts/multiplot.R')

# results where alpha_0 was 2.0
results_matrix_20 <- read.csv('../data_for_figures/init_20_param_sens.csv', header = FALSE)
rownames(results_matrix_20) <- c('alpha', 'refitted', 'linear approx')
results_df_20 <- as.data.frame(t(results_matrix_20))


# results where alpha_0 was 3.5
results_matrix_35 <- read.csv('../data_for_figures/init_35_param_sens.csv', header = FALSE)
rownames(results_matrix_35) <- c('alpha', 'refitted', 'linear approx')
results_df_35 <- as.data.frame(t(results_matrix_35))

# results where alpha_0 was 5.0
results_matrix_50 <- read.csv('../data_for_figures/init_50_param_sens.csv', header = FALSE)
rownames(results_matrix_50) <- c('alpha', 'refitted', 'linear approx')
results_df_50 <- as.data.frame(t(results_matrix_50))


plot_parametric_sensitivity <- function(results_df, alpha_0){
  results_df %>% 
    gather(method, e_num_clusters, -alpha) 
  
  results_df_long <-   results_df %>% 
    gather(method, e_num_clusters, -alpha) 
  
  plot <- 
    ggplot(results_df_long) + geom_point(aes(x = alpha, y = e_num_clusters, color = method)) + 
      geom_line(data = subset(results_df_long, method == 'linear approx'),
                aes(x = alpha, y = e_num_clusters, color = method)) + 
      geom_vline(xintercept = alpha_0, color = 'blue', linetype = 'dashed') + 
      xlab('alpha') + ylab('expected number of clusters') + theme(legend.position = c(0.75, 0.2))
  
  return(plot)
}

plot_parametric_sensitivity(results_df_20, alpha_0 = 2.0)

w <- 1.1
multiplot(
  plot_parametric_sensitivity(results_df_20, alpha_0 = 2.0), 
  plot_parametric_sensitivity(results_df_35, alpha_0 = 3.5), 
  plot_parametric_sensitivity(results_df_50, alpha_0 = 5.0),
  cols = 3)

