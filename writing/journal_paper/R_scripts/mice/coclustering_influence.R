infl_data <- 
  np$load('./R_scripts/mice/data/coclustering_worstcase.npz')

p_infl <- 
  ggplot() + 
  geom_line(aes(x = infl_data['logit_v_grid'], 
                y = infl_data['influence_grid'])) + 
  geom_hline(yintercept = 0., alpha = 0.5) + 
  ylab('influence') + 
  xlab('logit-stick') + 
  ggtitle('influnce function') + 
  get_fontsizes()

p_prior <-
  ggplot() + 
  geom_line(aes(x = infl_data['logit_v_grid'], 
                y = infl_data['p0_logit'])) + 
  geom_hline(yintercept = 0., alpha = 0.5) + 
  ylab('p') + 
  xlab('logit-stick') + 
  ggtitle(expression(paste("priors in logit space (", alpha, " = 3)"))) + 
  get_fontsizes()

p_infl_x_prior <- 
  ggplot() + 
  geom_line(aes(x = infl_data['logit_v_grid'], 
                y = infl_data['influence_grid_x_prior'])) + 
  geom_hline(yintercept = 0., alpha = 0.5) + 
  ylab('influence x p0') + 
  xlab('logit-stick') + 
  ggtitle('influence x prior') + 
  get_fontsizes()

p_infl + p_prior + p_infl_x_prior

# ggsave('./R_scripts/mice/figures_tmp/coclustering_influence.png', 
#        g,
#        width = 6, height = 2)
