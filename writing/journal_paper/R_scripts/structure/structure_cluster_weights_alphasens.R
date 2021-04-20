# for these clusters, we add a horizontal line corresponding to the threshold
thresh_df <- 
  data.frame(cluster = paste0('cluster ', 3:weights_keep), 
             thresh1 = threshold1)

thresh_df2 <- 
  data.frame(cluster = paste0('cluster ', 3), 
             thresh2 = threshold2)

weights_df %>% 
  left_join(thresh_df, by ='cluster') %>% 
  left_join(thresh_df2, by ='cluster') %>% 
  ggplot() + 
  # add line for threshold: only for some clusters ... 
  geom_line(aes(x = alpha, y = thresh1), 
            color = 'blue', alpha = 0.3) + 
  # geom_line(aes(x = alpha, y = thresh2), 
  #           color = 'blue', alpha = 0.3) + 
  # the actual results
  geom_point(aes(x = alpha, y = weight, color = method, shape = method)) + 
  geom_line(aes(x = alpha, y = weight, color = method)) + 
  scale_color_brewer(palette = 'Dark2') + 
  geom_vline(xintercept = alpha0, color = 'red', linetype = 'dashed') + 
  facet_wrap(~cluster, nrow = 2, scales = 'free_y') + 
  ylab('E[# loci]') + 
  get_fontsizes() + 
  theme(legend.position = 'bottom', 
        legend.justification = 'right',
        legend.box.margin = unit(c(0, 0.5, 0, 0), units = 'cm'),
        legend.title = element_blank(), 
        strip.background = element_rect(color = 'white', 
                                        fill = 'white'), 
        strip.text = element_text(hjust = 0, 
                                  size = title_size)) 
