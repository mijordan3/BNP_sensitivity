beta_prior <- function(x, alpha){
  return(dbeta(x, shape1 = 1, shape = alpha))
}

alpha_vec <- c(0.1, 1, 2, 4)

alpha_density_df <- 
  data.frame(x = seq(0, 1, length.out = 100))

for(alpha in alpha_vec){
  alpha_density_df[[paste0('alpha = ', alpha)]] <- 
    beta_prior(alpha_density_df$x, alpha = alpha)
}

alpha_density_df %>%
  gather(key = alpha, value = pdf, -x) %>% 
  # re-order alphas ... 
  mutate(alpha_num = sub('alpha = ', '', alpha)) %>% 
  mutate(alpha_num = as.numeric(alpha_num)) %>% 
  mutate(alpha = fct_reorder(as.factor(alpha),
                             alpha_num)) %>% 
  ggplot() + 
  geom_line(aes(x = x, y = pdf)) + 
  facet_wrap(~alpha, nrow = 1, scales = 'free_y') + 
  ylab('p.d.f.') + 
  xlab('stick propn') +
  get_fontsizes()
