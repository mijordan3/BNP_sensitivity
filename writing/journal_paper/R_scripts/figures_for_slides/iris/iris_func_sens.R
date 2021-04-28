# borrow these functions from the paper
source('../iris/iris_func_sens.R')

#############
# Plot influence function
#############
p_influence <- 
  influence_df %>%
  ggplot() + 
  geom_line(aes(x = logit_v, y = influence), 
            color = 'red') + 
  xlab('logit(stick length)') + 
  ylab(TeX('$\\Psi/P0$')) + 
  get_fontsizes()

#############
# Plot priors
#############
p_prior_logit <- 
  wc_results$priors_df %>%
  mutate(p0_logit = p0 * v * (1 - v)) %>% 
  ggplot() + 
  geom_line(aes(x = logit_v, y = p0_logit), 
            color = 'lightblue') + 
  xlab('logit(stick length)') + 
  ylab('p0') + 
  get_fontsizes()

p_prior <- 
  wc_results$priors_df %>%
  ggplot() + 
  geom_line(aes(x = v, y = p0), 
            color = 'lightblue') + 
  xlab('stick length') + 
  ylab('p0') + 
  get_fontsizes()

p_infl_x_prior <- 
  influence_df %>%
  ggplot() + 
  geom_line(aes(x = logit_v, y = influence_x_prior), 
            color = 'purple') + 
  xlab('logit(stick length)') + 
  ylab(TeX("$\\Phi$")) + 
  get_fontsizes()

p_prior + p_prior_logit + p_infl_x_prior

save_last_fig('iris_influence_function.png', 
              aspect_ratio = 0.3)

###################
# the first figure
###################
g00 <- plot_func_pert_results(fpert2_results,
                             remove_legend = FALSE,
                             remove_xlab = FALSE,
                             remove_title = FALSE,
                             ymax = ymax)
g00
save_last_fig('iris_func_sens0.png', 
              base_factor = 0.8,
              aspect_ratio = 0.4)


###################
# the rest of the functional perturbations
###################
g0 / g1 / g2
save_last_fig('iris_func_sens1.png', 
              base_factor = 0.8,
              aspect_ratio = 0.75)


###################
# the worst-case functional perturbation
###################
ymax_new <- 0.07
g0_new <- plot_func_pert_results(fpert1_results, 
                                 remove_legend = TRUE, 
                                 remove_xlab = TRUE, 
                                 remove_title = TRUE, 
                                 ymax = ymax_new)
  
g2_new <- plot_func_pert_results(fpert2_results, 
                          remove_legend = TRUE, 
                          remove_xlab = TRUE, 
                          remove_title = TRUE, 
                          ymax = ymax_new)

g3_new <- plot_func_pert_results(wc_results, 
                                 remove_title = TRUE, 
                                 ymax = ymax_new)


g0_new / g2_new / g3_new
save_last_fig('iris_worst_case.png', 
              base_factor = 0.8,
              aspect_ratio = 0.75)
