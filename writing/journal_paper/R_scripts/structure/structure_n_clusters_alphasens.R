# data frame concerning number of clusters

alpha0 <- alpha_sens_file['alpha0']

p1 <- plot_post_stat_trace_plot(alpha_sens_file['alpha_list'][1:10],
                                alpha_sens_file['n_clusters_refit'][1:10],
                                alpha_sens_file['n_clusters_lr'][1:10]) + 
  geom_vline(xintercept = alpha0, color = 'red', linetype = 'dashed') + 
  ylab('E[# clusters]') + 
  theme(legend.position = 'top',
        legend.justification = 'left',
        legend.title = element_blank()) 


p2 <- plot_post_stat_trace_plot(alpha_sens_file['alpha_list'][1:10],
                                alpha_sens_file['n_clusters_thresh_refit'][1:10],
                                alpha_sens_file['n_clusters_thresh_lr'][1:10]) + 
  geom_vline(xintercept = alpha0, color = 'red', linetype = 'dashed') + 
  ylab('E[# clusters thresh.]') + 
  theme(legend.position = 'none') 
  
p1 + p2
