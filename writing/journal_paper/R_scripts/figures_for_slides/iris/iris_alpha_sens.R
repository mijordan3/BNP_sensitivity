# load data for iris
load('../data_processed/iris.RData')


#####################
# in-sample plots 
#####################

# in-sample results
insample_df <- 
  data.frame(t = alpha_sens_df$alpha, 
             refit = alpha_sens_df$n_clusters_refit, 
             lin = alpha_sens_df$n_clusters_lr, 
             quantity = 'In-sample')

# predictive results
predictive_df <- 
  data.frame(t = alpha_sens_df_pred$alpha, 
             refit = alpha_sens_df_pred$n_clusters_refit, 
             lin = alpha_sens_df_pred$n_clusters_lr, 
             quantity = 'Predictive')

##################
# wrapper to plot
##################
plot_alpha_trace_plot <- function(results_df){
  p <- results_df %>% 
    plot_post_stat_trace_plot + 
    facet_wrap(~quantity, nrow = 1) + 
    ylab('E[# clusters]') + 
    xlab(TeX('DP parameter $\\alpha$')) + 
    # add vertical line
    geom_vline(xintercept = 6, 
               color = 'red', 
               linetype = 'dashed')
  
  return(p)
}

set_condition_to_nan <- function(x, bool_vec){
  x[bool_vec] <- NA
  return(x)
}

# refit, in-sample only 
rbind(insample_df, 
      predictive_df) %>% 
  mutate(lin = lin * NA) %>% 
  mutate(refit = set_condition_to_nan(refit, quantity == 'Predictive')) %>% 
  plot_alpha_trace_plot + 
  # keep axes conistent 
  geom_point(aes(x = min(insample_df$alpha), 
                 y = max(insample_df$lin)), 
             alpha = 0) + 
  geom_point(aes(x =min(insample_df$alpha), 
                 y = min(insample_df$lin)), 
             alpha = 0)

save_last_fig('iris_alpha_sens0.png',  
              aspect_ratio = 0.45)

# only the refit
rbind(insample_df, 
      predictive_df) %>% 
  mutate(refit = set_condition_to_nan(refit, quantity == 'Predictive')) %>% 
  mutate(lin = set_condition_to_nan(lin, quantity == 'Predictive')) %>% 
  plot_alpha_trace_plot

save_last_fig('iris_alpha_sens1.png',  
              aspect_ratio = 0.45)

# refit + lr
rbind(insample_df, 
      predictive_df) %>% 
  plot_alpha_trace_plot

save_last_fig('iris_alpha_sens2.png',  
              aspect_ratio = 0.45)
