#####################
# in-sample plots 
#####################
ymin = 3
ymax = 8.2

p1 <- plot_post_stat_trace_plot(alpha_sens_df$alpha,
                                alpha_sens_df$n_clusters_refit,
                                alpha_sens_df$n_clusters_lr) + 
  geom_vline(xintercept = alpha0, color = 'red', linetype = 'dashed') + 
  ylab('E[# clusters]') + 
  ylim(ymin, ymax) + 
  theme(legend.position = 'top', 
        legend.title = element_blank(), 
        legend.justification = 'left') 

p2 <- plot_post_stat_trace_plot(alpha_sens_df_pred$alpha,
                                alpha_sens_df_pred$n_clusters_refit,
                                alpha_sens_df_pred$n_clusters_lr) + 
  geom_vline(xintercept = alpha0, color = 'red', linetype = 'dashed') + 
  ylab('E[# pred. clusters]') + 
  ylim(ymin, ymax) + 
  theme(legend.position = 'none') 

p1 + p2
