# plot example gene
p1 <- example_data %>% 
  mutate(y_demean = y - mean(y)) %>% 
  ggplot() +
  geom_line(aes(x = time, y = fitted), 
            color = 'grey') + 
  geom_point(aes(x = time, y = y_demean), 
             color = 'blue', 
             shape = 'x', 
             size = 2) + 
  ylab('gene expr. (de-meaned)') + 
  xlab('time (hours)') + 
  theme_bw() + 
  get_fontsizes()

# plot regressor matrix
p2 <- 
  regr_df %>%
  distinct() %>% 
  gather(key = 'basis_id', 
         value = 'regressor', 
         -time) %>% 
  ggplot() + 
  geom_point(aes(x = time, y = regressor, color = basis_id), size = 0.5) + 
  geom_line(aes(x = time, y = regressor, color = basis_id)) + 
  ylab('B-spline basis value') + 
  xlab('time (hours)') + 
  theme_bw() + 
  theme(legend.position = 'none') + 
  get_fontsizes()

p1 + p2
