# for these clusters, we add a horizontal line corresponding to the threshold
thresh_df <- 
  data.frame(cluster = paste0('cluster ', 3:weights_keep), 
             thresh = threshold)

weights_df %>% 
  left_join(thresh_df, by ='cluster') %>% 
  ggplot() + 
  # add line for threshold: only for some clusters ... 
  geom_line(aes(x = alpha, y = thresh), 
            color = 'blue', alpha = 0.3) + 
  # the actual results
  geom_point(aes(x = alpha, y = weight, color = method, shape = method)) + 
  geom_line(aes(x = alpha, y = weight, color = method)) + 
  scale_color_brewer(palette = 'Dark2') + 
  geom_vline(xintercept = alpha0, color = 'red', linetype = 'dashed') + 
  facet_wrap(~cluster, nrow = 3, scales = 'free_y') + 
  ylab('E[# loci]') + 
  get_fontsizes() + 
  theme(legend.position = 'bottom', 
        legend.title = element_blank(),
        strip.text = element_text(size = axis_title_size, 
                                  margin = margin(.05, 0, .05, 0, "cm"))) 
