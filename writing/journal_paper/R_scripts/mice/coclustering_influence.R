infl_data <- 
  np$load('./R_scripts/mice/data/coclustering_worstcase.npz')

p_infl <- 
  influence_df %>% 
  ggplot() + 
  geom_line(aes(x = logit_v, 
                y = influence)) + 
  geom_hline(yintercept = 0., alpha = 0.5) + 
  ylab('influence') + 
  xlab('logit-stick') + 
  ggtitle('influnce function') + 
  get_fontsizes()

p_prior <-
  influence_df %>% 
  ggplot() + 
  geom_line(aes(x = logit_v,
                y = p0_logit)) + 
  geom_hline(yintercept = 0., alpha = 0.5) + 
  ylab('p') + 
  xlab('logit-stick') + 
  ggtitle("prior in logit space") + 
  get_fontsizes()

p_infl_x_prior <- 
  influence_df %>% 
  ggplot() + 
  geom_line(aes(x = logit_v, 
                y = influence_x_prior)) + 
  geom_hline(yintercept = 0., alpha = 0.5) + 
  ylab('influence x p0') + 
  xlab('logit-stick') + 
  ggtitle('influence x prior') + 
  get_fontsizes()

p_infl + p_prior + p_infl_x_prior
