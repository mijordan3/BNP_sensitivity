# data frame concerning number of clusters

set_to_na <- function(x, bool_vec){
  x[bool_vec] <- NA
  return(x)
}

# the initial figure with threshold = 0
alpha_sens_df %>% 
  rename(t = alpha) %>% 
  mutate(refit = set_to_na(refit, bool_vec = (threshold != 0))) %>% 
  mutate(lin = set_to_na(lin, bool_vec = (threshold != 0))) %>% 
  mutate(threshold = paste0('Threshold = ', threshold)) %>%
  plot_post_stat_trace_plot() + 
  facet_wrap(~threshold, nrow = 1, scales = 'free_y') + 
  # add a dummy point so that scales align
  geom_point(aes(x = 1, 4.5), alpha = 0.) + 
  ylab('E[# pop.]') + 
  xlab(TeX('DP parameter $\\alpha$')) + 
  geom_vline(xintercept = 3, 
             color = 'red', 
             linetype = 'dashed')
save_last_fig('stucture_alpha_sens0.png', 
              aspect_ratio = 0.45)



# we just use the same figure that was in our paper

source('../structure/structure_n_clusters_alphasens.R', 
       print.eval = FALSE)

p + xlab(TeX('DP parameter $\\alpha$')) 
save_last_fig('stucture_alpha_sens.png', 
              aspect_ratio = 0.45)
