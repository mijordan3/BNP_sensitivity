###############
# plot logit sticks parameter
###############
p0 <- logit_stick_flin_df %>% 
  plot_stick_params(title = 'stick') + 
  ylab('logit-stick location') + 
  theme(legend.position = 'none', 
        axis.text.x = element_blank(), 
        axis.title.x = element_blank())

p1 <- admix_flin_df %>% 
  plot_stick_params() + 
  ylab('admixture') + 
  # plot the fully-linearized quantity
  geom_line(aes(x = epsilon, y = fully_lin), 
            color = 'blue') + 
  theme(legend.position = 'bottom',
        legend.title = element_blank(), 
        legend.justification = 'right')

p0 / p1
