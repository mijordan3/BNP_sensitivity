###############
# plot logit sticks parameter
###############
plot_stick_params <- function(param_df, 
                              title = 'Population', 
                              color_populations = TRUE){
  
  wide_df <- 
    param_df %>%
    spread(key = method, value = y) %>% 
    mutate(population = paste(title, population, sep = ' ')) %>% 
    rename(t = epsilon) 
  
  p <-
    wide_df %>%
    plot_post_stat_trace_plot() +
    facet_wrap(~population, nrow = 1, scales = 'free_y') 
  
  if(color_populations){
    p <- p + 
      geom_rect(aes(xmin=-Inf, xmax=Inf, ymin=-Inf, ymax=Inf, 
                  fill = population), 
              alpha = pop_box_alpha,
              color = 'white',
              show.legend = FALSE) + 
      scale_fill_brewer(palette = 'Set2', type = 'qualitative')
    
    p <- move_layers(p, "GeomRect", position = "bottom")
  }
    
  return(p)
}


p0 <- logit_stick_df %>% 
  plot_stick_params(title = 'Stick', 
                    color_populations = FALSE) + 
  ylab('logit-stick location') + 
  theme(legend.position = 'none', 
        axis.text.x = element_blank(), 
        axis.title.x = element_blank())

p1 <- admix_df %>% 
  plot_stick_params() + 
  ylab('admixture') 

p0 / p1

