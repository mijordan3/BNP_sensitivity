out <- plot_initial_fit(add_geographic_labels = FALSE)

# we use Set2  for population colors
pop1_color <- '#66c2a5'
pop2_color <- '#fc8d62'
pop3_color <- '#8da0cb'

source('../structure/structure_fsens.R')

admix_plot <- plot_initial_fit()$p

admix_plot + 
  mbololo_box + 
  ngangao_box + 
  chawia_box + 
  # add letter labels 
  geom_text(aes(x = min(mbololo_outliers$obs_id) - 5, 
                y = 0.2, 
                label = 'A'), 
            size = text_size) + 
  geom_text(aes(x = min(ngangao_outliers$obs_id) - 5, 
                y = 0.5,
                label = 'B'), 
            size = text_size) +
  geom_text(aes(x = median(chawia_outliers$obs_id),
                y = 0.75, 
                label = 'C'), 
            size = text_size) + 
  theme(legend.position = 'top')

save_last_fig('structure_example_migration.png', 
              aspect_ratio = 0.5)

################
# Mbololo outliers 
################

# admixture plot with only mbololo labeled
p_admix_mbololo <- out$p + 
  mbololo_box + 
  # add letter labels 
  geom_text(aes(x = min(mbololo_outliers$obs_id) - 5, 
                y = 0.2, 
                label = 'A'), 
            size = text_size) + 
  theme(axis.text.x = element_blank(), 
        axis.ticks.x = element_blank(), 
        legend.position = 'top')

mbololo_plots <- 
  plot_struct_fsens_results(mbololo_fsens_results, 
                            pop_color = pop2_color)

mbololo_plots$p_logphi <- 
  mbololo_plots$p_logphi + 
  ggtitle('worst-case pert. of A') + 
  get_fontsizes()

mbololo_plots$p_priors <- 
  mbololo_plots$p_priors + 
  get_fontsizes() 

mbololo_plots$p_sens <-
  mbololo_plots$p_sens + 
  ylab('propn. pop2')


mbololo_plots_sum <- 
  mbololo_plots$p_logphi + 
  mbololo_plots$p_priors + 
  mbololo_plots$p_sens

p_admix_mbololo / mbololo_plots_sum

save_last_fig('structure_mbololo_sens.png', 
              aspect_ratio = 0.6)


################
# Ngangao outliers 
################
# admixture plot with only ngangao labeled
p_admix_ngangao <- out$p + 
  ngangao_box + 
  # add letter labels 
  geom_text(aes(x = min(ngangao_outliers$obs_id) - 5, 
                y = 0.5,
                label = 'B'), 
            size = text_size) +
  theme(axis.text.x = element_blank(), 
        axis.ticks.x = element_blank(), 
        legend.position = 'top')

ngangao_plots <- 
  plot_struct_fsens_results(ngangao_fsens_results, 
                            pop_color = pop1_color)

ngangao_plots$p_logphi <- 
  ngangao_plots$p_logphi + 
  ggtitle('worst-case pert. of B') + 
  get_fontsizes()

ngangao_plots$p_priors <- 
  ngangao_plots$p_priors + 
  get_fontsizes() + 
  title_remover  

ngangao_plots$p_sens <-
  ngangao_plots$p_sens + 
  ylab('propn. pop1') + 
  title_remover

ngangao_plots_sum <- 
  ngangao_plots$p_logphi + 
  ngangao_plots$p_priors + 
  ngangao_plots$p_sens

p_admix_ngangao / ngangao_plots_sum
save_last_fig('structure_ngangao_sens.png', 
              aspect_ratio = 0.6)

##########
# plots for chawia outliers
##########
p_admix_chawia <- out$p + 
  chawia_box + 
  # add letter labels 
  geom_text(aes(x = median(chawia_outliers$obs_id),
                y = 0.75, 
                label = 'C'), 
            size = text_size) + 
  theme(axis.text.x = element_blank(), 
        axis.ticks.x = element_blank(), 
        legend.position = 'top')

p_admix_chawia / chawia_plots_sum
save_last_fig('structure_chawia_sens.png', 
              aspect_ratio = 0.6)
