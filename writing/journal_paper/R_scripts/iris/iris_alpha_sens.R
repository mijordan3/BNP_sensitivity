#####################
# in-sample plots 
#####################
p1 <- plot_post_stat_trace_plot(alpha_sens_df$alpha,
                                alpha_sens_df$n_clusters_refit,
                                alpha_sens_df$n_clusters_lr) + 
  geom_vline(xintercept = alpha0, color = 'red', linetype = 'dashed') + 
  ylab('E[# clusters]') + 
  ggtitle('in-sample (thresh = 0)') + 
  theme(legend.position = 'none', 
        axis.ticks.x = element_blank(), 
        axis.text.x = element_blank(),
        axis.title.x = element_blank()) 

p2 <- plot_post_stat_trace_plot(alpha_sens_df$alpha,
                                alpha_sens_df$n_clusters_thresh_refit,
                                alpha_sens_df$n_clusters_thresh_lr) + 
  geom_vline(xintercept = alpha0, color = 'red', linetype = 'dashed') + 
  ylab('E[# clusters thresh.]') + 
  ggtitle('in-sample (thresh = 1)') + 
  theme(legend.position = 'none', 
        axis.ticks.x = element_blank(), 
        axis.text.x = element_blank(),
        axis.title.x = element_blank()) 

#####################
# predictive plots 
#####################
p3 <- plot_post_stat_trace_plot(alpha_sens_df_pred$alpha,
                                alpha_sens_df_pred$n_clusters_refit,
                                alpha_sens_df_pred$n_clusters_lr) + 
  geom_vline(xintercept = alpha0, color = 'red', linetype = 'dashed') + 
  ylab('E[# clusters]') + 
  ggtitle('predictive (thresh = 0)') + 
  theme(legend.position = 'none') 

p4 <- plot_post_stat_trace_plot(alpha_sens_df_pred$alpha,
                                alpha_sens_df_pred$n_clusters_thresh_refit,
                                alpha_sens_df_pred$n_clusters_thresh_lr) + 
  geom_vline(xintercept = alpha0, color = 'red', linetype = 'dashed') + 
  ylab('E[# clusters thresh.]') + 
  ggtitle('predictive (thresh = 1)') + 
  theme(legend.position = 'bottom', 
        legend.title = element_blank(), 
        legend.justification = 'right') 


((p1 + p2) / (p3 + p4)) + 
  plot_layout(guides = "collect") &
  theme(legend.position = 'bottom',
        legend.title = element_blank(), 
        legend.text = element_text(size = axis_title_size))

