# data frame concerning number of clusters

p1 <- plot_post_stat_trace_plot(alpha_sens_df$alpha,
                                alpha_sens_df$n_clusters_refit,
                                alpha_sens_df$n_clusters_lr) + 
  geom_vline(xintercept = alpha0, color = 'red', linetype = 'dashed') + 
  ylab('E[# clusters]') + 
  theme(legend.position = 'top',
        legend.justification = 'left',
        legend.title = element_blank()) 


p2 <- plot_post_stat_trace_plot(alpha_sens_df$alpha,
                                alpha_sens_df$n_clusters_thresh_refit,
                                alpha_sens_df$n_clusters_thresh_lr) + 
  geom_vline(xintercept = alpha0, color = 'red', linetype = 'dashed') + 
  ylab('E[# clusters thresh.]') + 
  theme(legend.position = 'none') 
  
p1 + p2
save_last_fig('./figures/stucture_alpha_sens.png', 
              aspect_ratio = 0.45)