weights_keep <- 9

weights_refit_df <-
  data.frame(alpha_sens_file['cluster_weights_refit'][, 1:weights_keep]) %>% 
  mutate(alpha = alpha_sens_file['alpha_list'], 
         method = 'refit')

weights_lr_df <- 
  data.frame(alpha_sens_file['cluster_weights_lr'][, 1:weights_keep]) %>% 
  mutate(alpha = alpha_sens_file['alpha_list'], 
         method = 'lr')

weights_df <- rbind(weights_refit_df, weights_lr_df) %>% 
  gather(key = cluster, value = weight, -c('alpha', 'method')) %>% 
  mutate(cluster = sub('X', 'cluster ', cluster))

# for these clusters, we add a horizontal line corresponding to the threshold
thresh_df <- 
  data.frame(cluster = paste0('cluster ', 4:weights_keep), 
             thresh = alpha_sens_file['threshold'])

weights_df %>% 
  left_join(thresh_df, by ='cluster') %>% 
  ggplot() + 
  # add line for threshold: only for some clusters ... 
  geom_line(aes(x = alpha, y = thresh), 
            color = 'blue', alpha = 0.3) + 
  # the actual results
  geom_point(aes(x = alpha, y = weight, color = method), size = 0.8) + 
  geom_line(aes(x = alpha, y = weight, color = method)) + 
  scale_color_brewer(palette = 'Dark2') + 
  geom_vline(xintercept = alpha0, color = 'red', linetype = 'dashed') + 
  facet_wrap(~cluster, nrow = 3, scales = 'free_y') + 
  ylab('E[# loci]') + 
  fontsize_theme + 
  theme(legend.position = 'bottom', 
        legend.title = element_blank(),
        strip.text = element_text(size = axis_ticksize, 
                                  margin = margin(.05, 0, .05, 0, "cm"))) 
