# data frame concerning number of clusters

p1 <- plot_post_stat_trace_plot(alpha_sens_df$alpha,
                                alpha_sens_df$n_clusters_refit,
                                alpha_sens_df$n_clusters_lr) + 
  geom_vline(xintercept = alpha0, color = 'red', linetype = 'dashed') + 
  ylab('E[# pop.]') + 
  ggtitle(paste0('thresh = ', threshold0)) + 
  theme(legend.position = 'none') 

ymax <- 4.75

p2 <- plot_post_stat_trace_plot(alpha_sens_df$alpha,
                                alpha_sens_df$n_clusters_thresh_refit1,
                                alpha_sens_df$n_clusters_thresh_lr1) + 
  geom_vline(xintercept = alpha0, color = 'red', linetype = 'dashed') + 
  ylab('E[# pop.]') + 
  ggtitle(paste0('thresh = ', threshold1)) + 
  theme(legend.position = 'none', 
        axis.title.y = element_blank()) +
  ylim(2, ymax)

p3 <- plot_post_stat_trace_plot(alpha_sens_df$alpha,
                                alpha_sens_df$n_clusters_thresh_refit2,
                                alpha_sens_df$n_clusters_thresh_lr2) + 
  geom_vline(xintercept = alpha0, color = 'red', linetype = 'dashed') + 
  ylab('E[# pop.]') + 
  ggtitle(paste0('thresh = ', threshold2)) + 
  theme(axis.title.y = element_blank(), 
        legend.position = 'bottom', 
        legend.title = element_blank(),
        legend.justification = 'right') + 
  ylim(2, ymax)


p1 + p2 + p3
