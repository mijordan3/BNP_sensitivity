###############
# plot logit sticks parameter
###############
plot_stick_params <- function(param_df, title = 'population'){
  p <- param_df %>%
    mutate(population = paste(title, population, sep = ' ')) %>% 
    ggplot() + 
    geom_point(aes(x = epsilon, y = y,
                   color = method, shape = method)) + 
    geom_line(aes(x = epsilon, y = y,
                  color = method)) + 
    facet_wrap(~population, scales = 'free_y', nrow = 1) + 
    scale_color_brewer(palette = 'Dark2') + 
    get_fontsizes() + 
    theme(strip.background = element_rect(fill = 'white',
                                          color = 'white'))
  
  return(p)
}

p0 <- logit_stick_df %>% 
  plot_stick_params(title = 'stick') + 
  ylab('logit-stick location') + 
  theme(legend.position = 'none', 
        axis.text.x = element_blank(), 
        axis.title.x = element_blank())

p1 <- admix_df %>% 
  plot_stick_params() + 
  ylab('admixture') + 
  theme(legend.position = 'bottom',
        legend.title = element_blank(), 
        legend.justification = 'right')

p0 / p1
