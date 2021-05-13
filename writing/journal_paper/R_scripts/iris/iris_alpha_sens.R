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

# plot
rbind(insample_df, 
      predictive_df) %>% 
  plot_post_stat_trace_plot + 
  facet_wrap(~quantity, nrow = 1) + 
  ylab('E[# clusters]') + 
  xlab(TeX('GEM parameter $\\alpha$')) + 
  # add vertical line
  geom_vline(xintercept = alpha0, 
             color = 'red', 
             linetype = 'dashed')

