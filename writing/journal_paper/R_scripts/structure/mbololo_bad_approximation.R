###############
# plot logit sticks parameter
###############
plot_stick_params <- function(param_df, title = 'population'){
  
  wide_df <- 
    param_df %>%
    spread(key = method, value = y) %>% 
    mutate(population = paste(title, population, sep = ' ')) %>% 
    rename(t = epsilon) 
  
  p <-
    wide_df %>%
    plot_post_stat_trace_plot() +
    facet_wrap(~population, nrow = 1, scales = 'free_y') +
    geom_rect(data = subset(wide_df, population = 'stick 1'),
              aes(xmin=-Inf, xmax=Inf, ymin=-Inf, ymax=Inf), 
              color = 'red')

  return(p)
}

logit_stick_df %>% 
  plot_stick_params(title = 'stick')

p0 <- logit_stick_df %>% 
  plot_stick_params(title = 'stick') + 
  ylab('logit-stick location') + 
  theme(legend.position = 'none', 
        axis.text.x = element_blank(), 
        axis.title.x = element_blank())

p0

p1 <- admix_df %>% 
  plot_stick_params() + 
  ylab('admixture') 

p0 / p1
