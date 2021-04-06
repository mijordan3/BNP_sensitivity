#####################
# in-sample plots 
#####################
p1 <- plot_post_stat_trace_plot(alpha_sens_df_pred$alpha,
                                alpha_sens_df_pred$n_clusters_refit,
                                alpha_sens_df_pred$n_clusters_lr) + 
  geom_vline(xintercept = alpha0, color = 'red', linetype = 'dashed') + 
  ylab('E[# clusters]') + 
  theme(legend.position = 'top', 
        legend.title = element_blank(), 
        legend.justification = 'left') 

p2 <- plot_post_stat_trace_plot(alpha_sens_df_pred$alpha,
                                alpha_sens_df_pred$n_clusters_thresh_refit,
                                alpha_sens_df_pred$n_clusters_thresh_lr) + 
  geom_vline(xintercept = alpha0, color = 'red', linetype = 'dashed') + 
  ylab('E[# clusters thresh.]') + 
  theme(legend.position = 'none') 

p1 + p2