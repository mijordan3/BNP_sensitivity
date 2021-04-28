###############
# plot logit sticks parameter
###############
p0 <- logit_stick_flin_df %>% 
  plot_stick_params(title = 'Stick', 
                    color_populations = FALSE) + 
  ylab('logit-stick location') + 
  theme(legend.position = 'none', 
        axis.text.x = element_blank(),
        axis.title.x = element_blank())


# a copy of the data to draw fully linearized lines 
dummy_df <- 
  admix_flin_df %>% 
  mutate(population = paste0('Population ', population))

p1 <- admix_flin_df %>% 
  plot_stick_params() + 
  ylab('admixture') + 
  # plot the fully-linearized quantity
  geom_line(data = dummy_df,
            aes(x = epsilon, y = fully_lin),
            color = 'red', 
            linetype = 'dashed') + 
  guides(color = FALSE, 
         shape = FALSE)

p0 / p1 + plot_layout(guides = "collect") & theme(legend.position = 'bottom')
