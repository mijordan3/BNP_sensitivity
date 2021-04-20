# data frame concerning number of clusters

p1 <- plot_post_stat_trace_plot(alpha_sens_df$alpha,
                                alpha_sens_df$n_clusters_refit,
                                alpha_sens_df$n_clusters_lr) + 
  geom_vline(xintercept = alpha0, color = 'red', linetype = 'dashed') + 
  ylab('E[# pop.]') + 
  ggtitle(paste0('thresh = ', 0)) + 
  theme(legend.position = 'none', 
        axis.title.x = element_blank(), 
        axis.text.x = element_blank()) 

ymax <- 4.75

p2 <- plot_post_stat_trace_plot(alpha_sens_df$alpha,
                                alpha_sens_df$n_clusters_thresh_refit,
                                alpha_sens_df$n_clusters_thresh_lr) + 
  geom_vline(xintercept = alpha0, color = 'red', linetype = 'dashed') + 
  ylab('E[# pop.]') + 
  ggtitle(paste0('thresh = ', threshold)) + 
  theme(legend.position = 'none', 
        axis.title.x = element_blank(), 
        axis.text.x = element_blank(), 
        axis.title.y = element_blank()) +
  ylim(2, ymax)

p3 <- plot_post_stat_trace_plot(alpha_sens_df$alpha,
                                alpha_sens_df$n_clusters_thresh_refit2,
                                alpha_sens_df$n_clusters_thresh_lr2) + 
  geom_vline(xintercept = alpha0, color = 'red', linetype = 'dashed') + 
  ylab('E[# pop.]') + 
  ggtitle(paste0('thresh = ', threshold2)) + 
  theme(legend.position = 'none') + 
  ylim(2, ymax)


p4 <- plot_post_stat_trace_plot(alpha_sens_df$alpha,
                                alpha_sens_df$n_clusters_thresh_refit3,
                                alpha_sens_df$n_clusters_thresh_lr3) + 
  geom_vline(xintercept = alpha0, color = 'red', linetype = 'dashed') + 
  ylab('E[# pop.]') + 
  ggtitle(paste0('thresh = ', threshold3)) + 
  theme(legend.position = 'bottom', 
        legend.justification = 'right', 
        legend.title = element_blank(), 
        axis.title.y = element_blank()) +
  ylim(2, ymax)


(p1 + p2) / (p3 + p4)
