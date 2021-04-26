# data frame concerning number of clusters

p <- alpha_sens_df %>% 
  rename(t = alpha) %>% 
  mutate(threshold = paste0('Threshold = ', threshold)) %>%
  plot_post_stat_trace_plot() + 
  facet_wrap(~threshold, nrow = 1, scales = 'free_y') + 
  # add a dummy point so that scales align
  geom_point(aes(x = 1, 4.5), alpha = 0.) + 
  ylab('E[# pop.]') + 
  xlab(TeX('GEM parameter $\\alpha$')) + 
  geom_vline(xintercept = 3, 
             color = 'red', 
             linetype = 'dashed')

p