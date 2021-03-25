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
  fontsize_theme

p_infl_x_prior <- 
  ggplot() + 
  geom_line(aes(x = infl_data['logit_v_grid'], 
                y = infl_data['influence_grid_x_prior'])) + 
  geom_hline(yintercept = 0., alpha = 0.5) + 
  ylab('influence x p0') + 
  xlab('logit-stick') + 
  ggtitle(' ') + 
  fontsize_theme
g <- arrangeGrob(p_infl, p_infl_x_prior,
                 nrow = 1)

# p_wc <-
#   ggplot(data = NULL, 
#          aes(x = infl_data['logit_v_grid'], 
#              y = infl_data['worst_case_grid'])) + 
#   geom_area(fill = 'grey', color = 'black') + 
#   geom_hline(yintercept = 0, alpha = 0.5) + 
#   ggtitle('worst-case log-phi') + 
#   xlab('logit-stick') + 
#   ylab('log phi') + 
#   fontsize_theme
# 
# p_priors <- 
#   data.frame(logit_v_grid = infl_data['logit_v_grid'], 
#              p0 = infl_data['p0_logit'], 
#              p1 = infl_data['pc_logit']) %>%
#   gather(key = prior, value = p, -logit_v_grid) %>% 
#   ggplot() + 
#   geom_line(aes(x = logit_v_grid, 
#                 y = p, 
#                 color = prior)) + 
#   scale_color_manual(values = c('lightblue', 'blue')) + 
#   xlab('logit stick') + 
#   ggtitle('priors in logit space') + 
#   fontsize_theme + 
#   theme(legend.position = 'none')
# 
# 
# p_priors_contr <- 
#   data.frame(logit_v_grid = sigmoid(infl_data['logit_v_grid']), 
#              p0 = infl_data['p0_constrained'], 
#              p1 = infl_data['pc_constrained']) %>%
#   gather(key = prior, value = p, -logit_v_grid) %>% 
#   ggplot() + 
#   geom_line(aes(x = logit_v_grid, 
#                 y = p, 
#                 color = prior)) + 
#   scale_color_manual(values = c('lightblue', 'blue')) + 
#   xlab('stick') + 
#   ggtitle('priors in constrained space') + 
#   fontsize_theme + 
#   theme(legend.title = element_blank(), 
#         legend.position = c(0.85, 0.75))

# g <- arrangeGrob(p_infl, p_wc, p_priors, p_priors_contr, 
#                  nrow = 2)

ggsave('./R_scripts/mice/figures_tmp/coclustering_influence.png', 
       g,
       width = 6, height = 2)
