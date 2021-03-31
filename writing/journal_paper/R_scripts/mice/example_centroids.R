expression_data %>%
  # join cluster memberships
  inner_join(cluster_memberships, by = 'gene_id') %>%
  # join centroids
  inner_join(centroids_df, by = c('cluster_id', 'time')) %>% 
  # now filter to those top clusters
  inner_join(cluster_weights, by = 'cluster_id') %>%
  # re-order by cluster weights
  mutate(cluster_id = fct_reorder(as.factor(cluster_id),
                                  counts, .desc = TRUE)) %>% 
  # plot
  ggplot() + 
  geom_line(aes(x = time, y = expression, group = gene_id),
            alpha = 0.5) +
  geom_line(aes(x = time, y = centroid_value), 
            color = 'blue', size = 1.5) +
  facet_wrap(~cluster_id, scales = 'free_y') + 
  theme_bw() + 
  theme(strip.background = element_blank(),
        strip.text.x = element_blank()) + 
  xlab('time (hours)') + 
  ylab('gene expr. (shifted)') + 
  get_fontsizes()

