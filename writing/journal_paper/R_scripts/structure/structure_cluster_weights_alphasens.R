weights_df %>% 
  spread(key = method, value  = weight) %>% 
  rename(t = alpha) %>% 
  plot_post_stat_trace_plot() + 
  facet_wrap(~cluster, nrow = 2, scales = 'free_y') + 
  geom_vline(xintercept = alpha0, color = 'red', linetype = 'dashed') +
  ylab('E[# loci]') + 
  xlab('epsilon')

