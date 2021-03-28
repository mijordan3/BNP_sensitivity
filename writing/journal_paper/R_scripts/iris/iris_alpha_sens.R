alpha_sens_file <- np$load("./R_scripts/iris/data/iris_alpha_sens.npz")

alpha0 <- 6 # alpha_sens_file['alpha0']

p1 <- plot_post_stat_trace_plot(alpha_sens_file['alpha_list'],
                                alpha_sens_file['n_clusters_refit'],
                                alpha_sens_file['n_clusters_lr']) + 
  geom_vline(xintercept = alpha0, color = 'red', linetype = 'dashed') + 
  ylab('E[# clusters]') + 
  theme(legend.key.size = unit(0.5, 'cm'), 
        legend.position = c(0.2, 0.7), 
        legend.title = element_blank()) 

p2 <- plot_post_stat_trace_plot(alpha_sens_file['alpha_list'],
                                alpha_sens_file['n_clusters_thresh_refit'],
                                alpha_sens_file['n_clusters_thresh_lr']) + 
  geom_vline(xintercept = alpha0, color = 'red', linetype = 'dashed') + 
  ylab('E[# clusters thresh.]') + 
  theme(legend.position = 'none') 


grid.arrange(p1, p2, nrow = 1)
