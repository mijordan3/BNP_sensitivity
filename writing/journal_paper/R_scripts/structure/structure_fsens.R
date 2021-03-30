out <- plot_initial_fit()

#################
# label outliers
#################
mbololo_outliers <- 
  out$ind_admix_df %>%
  filter(label == 'Mbololo') %>% 
  filter(cluster == 'X3') %>% 
  filter(admix > 0.15)

ngangao_outliers <- 
  out$ind_admix_df %>%
  filter(label == 'Ngangao') %>% 
  filter(cluster == 'X1') %>% 
  filter(admix > 0.4)

chawia_outliers <- 
  out$ind_admix_df %>%
  filter(label == 'Chawia')


intercepts <- c(mbololo_outliers$obs_id - 1,
                mbololo_outliers$obs_id + 1, 
                min(ngangao_outliers$obs_id) - 1, 
                max(ngangao_outliers$obs_id) + 1,
                min(chawia_outliers$obs_id) - 1, 
                max(chawia_outliers$obs_id) - 0.5)

rect_alpha = 0.1
linesize = 0.5
text_height = 0.1
text_size = 3
p_admix <- out$p + 
  # grey out everything we don't want
  geom_rect(aes(xmin = 0, xmax = intercepts[1], 
                ymin = 0, ymax = 1), 
            fill = 'grey', alpha = rect_alpha) + 
  geom_rect(aes(xmin = intercepts[2], xmax = intercepts[3], 
                ymin = 0, ymax = 1), 
            fill = 'grey', alpha = rect_alpha) + 
  geom_rect(aes(xmin = intercepts[4], xmax = intercepts[5], 
                ymin = 0, ymax = 1), 
            fill = 'grey', alpha = rect_alpha) + 
  # add vertical lines
  # label mbololo outliers
  geom_vline(xintercept = c(intercepts[1], intercepts[2]),
             size = linesize) +
  # label ngangao outliers
  geom_vline(xintercept = c(intercepts[3], intercepts[4]),
             size = linesize) +
  # label chawia
  geom_vline(xintercept = c(intercepts[5], intercepts[6]),
             size = linesize) +
  # add letter labels 
  geom_text(aes(x = intercepts[1] - 5, y = text_height, label = 'A'), 
            size = text_size) + 
  geom_text(aes(x = intercepts[3] - 5, y = text_height, label = 'B'), 
            size = text_size) + 
  geom_text(aes(x = (intercepts[5] + intercepts[6]) / 2,
                y = text_height, label = 'C'), 
            size = text_size) + 
  theme(axis.ticks = element_blank(), 
        axis.title.x = element_blank(),
        axis.text.x = element_blank())

#################
# plot results
#################
plot_struct_fsens_results <- function(input_file){
  
  fsens_results <- np$load(input_file)
  
  logit_v <- fsens_results['logit_v_grid']
  infl_function <- fsens_results['influence_grid_x_prior']
  log_phi <- fsens_results['log_phi']
  
  scale <- max(abs(log_phi)) / max(abs(infl_function))
  
  p_logphi <- plot_influence_and_logphi(logit_v, 
                                  infl_function, 
                                  log_phi) 
  p_priors <- plot_priors(sigmoid(logit_v), 
                    exp(fsens_results['log_p0']), 
                    exp(fsens_results['log_pc'])) + 
    xlab('stick') + 
    ggtitle('Priors') + 
    theme(legend.title = element_blank(), 
          legend.position = 'bottom')
  
  
  p_sens <- plot_post_stat_trace_plot(fsens_results['epsilon_vec'], 
                                  fsens_results['refit_vec'], 
                                  fsens_results['lr_vec']) + 
    ggtitle(' ') + 
    xlab('epsilon') + 
    theme(legend.title = element_blank(), 
          legend.position = 'bottom')
  
  return(list(p_logphi = p_logphi, 
              p_priors = p_priors, 
              p_sens = p_sens))
}

##########
# plots for mbololo outliers
##########
x_axis_remover <- 
  theme(axis.title.x = element_blank(),
        axis.text.x = element_blank(), 
        legend.position = 'none') 

title_remover <- ggtitle(NULL)

mbololo_plots <- 
  plot_struct_fsens_results('./R_scripts/structure/data/stru_fsens_mbololo.npz')

mbololo_plots$p_logphi <- 
  mbololo_plots$p_logphi + 
  ggtitle('Worst-case pert. of A') + 
  get_fontsizes() + 
  x_axis_remover

mbololo_plots$p_priors <- 
  mbololo_plots$p_priors + 
  get_fontsizes() + 
  x_axis_remover 

mbololo_plots$p_sens <-
  mbololo_plots$p_sens + 
  ylab('propn. purple') + 
  x_axis_remover 

mbololo_plots_sum <- 
  mbololo_plots$p_logphi + 
  mbololo_plots$p_priors + 
  mbololo_plots$p_sens


##########
# plots for ngangao outliers
##########
ngangao_plots <- 
  plot_struct_fsens_results('./R_scripts/structure/data/stru_fsens_ngangao.npz')

ngangao_plots$p_logphi <- 
  ngangao_plots$p_logphi + 
  ggtitle('Worst-case pert. of B') + 
  get_fontsizes() + 
  x_axis_remover

ngangao_plots$p_priors <- 
  ngangao_plots$p_priors + 
  get_fontsizes() + 
  title_remover + 
  x_axis_remover 

ngangao_plots$p_sens <-
  ngangao_plots$p_sens + 
  ylab('propn. green') + 
  title_remover + 
  x_axis_remover 

ngangao_plots_sum <- 
  ngangao_plots$p_logphi + 
  ngangao_plots$p_priors + 
  ngangao_plots$p_sens


##########
# plots for chawia outliers
##########
chawia_plots <- 
  plot_struct_fsens_results('./R_scripts/structure/data/stru_fsens_chawia.npz')

chawia_plots$p_logphi <- chawia_plots$p_logphi + 
  ggtitle('Worst-case pert. of C')

chawia_plots$p_priors <- chawia_plots$p_priors + 
  title_remover

chawia_plots$p_sens <- chawia_plots$p_sens + 
  title_remover + 
  ylab('propn. purple')

chawia_plots_sum <- 
  chawia_plots$p_logphi + 
  chawia_plots$p_priors + 
  chawia_plots$p_sens


p_admix / mbololo_plots_sum / ngangao_plots_sum / chawia_plots_sum

# layout_matrix <- matrix(c(1, 1, 1, 2:10), 
#                         nrow = 4, 
#                         byrow = TRUE)
# grid.arrange(p_admix, 
#              mbololo_plots$p1, mbololo_plots$p2, mbololo_plots$p3,
#              ngangao_plots$p1, ngangao_plots$p2, ngangao_plots$p3,
#              chawia_plots$p1, chawia_plots$p2, chawia_plots$p3,
#              layout_matrix = layout_matrix)

# g <- arrangeGrob(p_admix, 
#                  mbololo_plots$p1, mbololo_plots$p2, mbololo_plots$p3,
#                  ngangao_plots$p1, ngangao_plots$p2, ngangao_plots$p3,
#                  chawia_plots$p1, chawia_plots$p2, chawia_plots$p3,
#                  layout_matrix = layout_matrix)
# 
# ggsave('./R_scripts/structure/figures_tmp/fsens_structure.png', g,
#        width = 7, height = 6)
