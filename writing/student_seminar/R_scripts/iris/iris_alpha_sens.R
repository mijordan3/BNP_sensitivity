#####################
# in-sample plots 
#####################

# only the refit
p0 <- plot_post_stat_trace_plot(alpha_sens_df$alpha,
                                alpha_sens_df$n_clusters_refit,
                                alpha_sens_df$n_clusters_lr * NaN) + 
  geom_vline(xintercept = alpha0, color = 'red', linetype = 'dashed') + 
  ylab('E[# clusters]') + 
  theme(legend.position = 'top', 
        legend.title = element_blank(), 
        legend.justification = 'left') 

p0 + plot_spacer()
save_last_fig('./figures/iris_alpha_sens0.png',  
              aspect_ratio = 0.45)

# refit + lr
p1 <- plot_post_stat_trace_plot(alpha_sens_df$alpha,
                                alpha_sens_df$n_clusters_refit,
                                alpha_sens_df$n_clusters_lr) + 
  geom_vline(xintercept = alpha0, color = 'red', linetype = 'dashed') + 
  ylab('E[# clusters]') + 
  theme(legend.position = 'top', 
        legend.title = element_blank(), 
        legend.justification = 'left') 

p1 + plot_spacer()
save_last_fig('./figures/iris_alpha_sens1.png',  
              aspect_ratio = 0.45)

#####################
# predictive plots 
#####################

p2 <- plot_post_stat_trace_plot(alpha_sens_df_pred$alpha,
                                alpha_sens_df_pred$n_clusters_refit,
                                alpha_sens_df_pred$n_clusters_lr) +
  geom_vline(xintercept = alpha0, color = 'red', linetype = 'dashed') +
  ylab('E[# clusters pred.]') +
  theme(legend.position = 'none')


p1 + p2
save_last_fig('./figures/iris_alpha_sens2.png',  
              aspect_ratio = 0.45)
