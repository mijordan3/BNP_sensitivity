###############
# plot logit sticks parameter
###############
p0 <- logit_stick_df %>%
  mutate(population = paste0('population ', population)) %>% 
  ggplot() + 
  geom_point(aes(x = epsilon, y = logit_stick_location,
                 color = method, shape = method)) + 
  geom_line(aes(x = epsilon, y = logit_stick_location,
                 color = method)) + 
  facet_wrap(~population, scales = 'free_y', nrow = 1) + 
  ylab('logit-stick location') +
  scale_color_brewer(palette = 'Dark2') + 
  get_fontsizes() + 
  theme(legend.position = 'none', 
        axis.text.x = element_blank(), 
        axis.title.x = element_blank(), 
        strip.text = element_text(size = title_size), 
        strip.background = element_rect(fill = 'white', 
                                        color = 'white'))


###############
# plot admixture
###############
p1 <- admix_df %>%
  mutate(population = paste0('population ', population)) %>% 
  ggplot() + 
  geom_point(aes(x = epsilon, y = admix,
                 color = method, shape = method)) + 
  geom_line(aes(x = epsilon, y = admix,
                color = method)) + 
  facet_wrap(~population, scales = 'free_y', nrow = 1) + 
  ylab('admixture') +
  scale_color_brewer(palette = 'Dark2') + 
  get_fontsizes() + 
  theme(legend.position = 'bottom',
        legend.title = element_blank(), 
        legend.justification = 'right',
        strip.text = element_blank())
p0 / p1
