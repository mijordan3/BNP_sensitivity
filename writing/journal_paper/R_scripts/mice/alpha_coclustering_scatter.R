min_keep <- 1e-3
breaks = c(1e3, 1e5)

coclust_diff %>% 
  # spread method and diff
  select(gene1, gene2, alpha, diff, method) %>% 
  spread(method, diff) %>%
  # for plotting get rid of this high density patch at zero
  filter(abs(lr) > min_keep | abs(refit) > min_keep) %>%
  # rename alpha for plotting 
  mutate(alpha = paste0('alpha = ', alpha)) %>% 
  ggplot(aes(x = refit, y = lr)) + 
  # the area we excluded
  geom_rect(xmin = -min_keep, xmax = min_keep, 
            ymin = -min_keep, ymax = min_keep, 
            fill = 'grey', color = 'black') + 
  # identity line
  geom_abline(slope = 1, intercept = 0) +
  # the points
  geom_point(alpha = 0.1) +
  # 2d density
  geom_density_2d(breaks = breaks) +
  scale_fill_brewer(palette = 'PuBu') + 
  facet_wrap(~alpha) + 
  ylab("lr - init") + 
  xlab("refit - init")
