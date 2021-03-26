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
linesize = 1
text_height = 0.08
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
  geom_text(aes(x = intercepts[1] - 5, y = text_height, label = 'A')) + 
  geom_text(aes(x = intercepts[3] - 5, y = text_height, label = 'B')) + 
  geom_text(aes(x = (intercepts[5] + intercepts[6]) / 2,
                y = text_height, label = 'C')) + 
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
  
  p1 <- plot_post_stat_trace_plot(fsens_results['epsilon_vec'], 
                                  fsens_results['refit_vec'], 
                                  fsens_results['lr_vec']) + 
    ggtitle(' ') + 
    xlab('epsilon') + 
    theme(legend.title = element_blank(), 
          legend.position = c(0.8, 0.8), 
          legend.key.size = unit(0.25, 'cm'))
  
  p2 <- plot_influence_and_logphi(logit_v, 
                                  infl_function, 
                                  log_phi) 
  p3 <- plot_priors(sigmoid(logit_v), 
                    exp(fsens_results['log_p0']), 
                    exp(fsens_results['log_pc'])) + 
    xlab('stick') + 
    ggtitle('Priors') + 
    theme(legend.title = element_blank(), 
          legend.position = c(0.8, 0.8), 
          legend.key.size = unit(0.25, 'cm'))
  
  return(list(p1 = p1, 
              p2 = p2, 
              p3 = p3))
}

##########
# plots for mbololo outliers
##########
x_axis_remover <- 
  theme(axis.title.x = element_blank(),
        axis.text.x = element_blank())


mbololo_plots <- 
  plot_struct_fsens_results('./R_scripts/structure/data/stru_fsens_mbololo.npz')

mbololo_plots$p1 <-
  mbololo_plots$p1 + 
  ylab('Propn. purple') + 
  x_axis_remover + 
  ggtitle('Sensitivity of A') 

mbololo_plots$p2 <- mbololo_plots$p2 + x_axis_remover

mbololo_plots$p3 <- 
  mbololo_plots$p3 + x_axis_remover 

##########
# plots for ngangao outliers
##########
ngangao_plots <- 
  plot_struct_fsens_results('./R_scripts/structure/data/stru_fsens_ngangao.npz')
ngangao_plots$p1 <- ngangao_plots$p1 + 
  x_axis_remover + 
  ylab('Propn. green') + 
  theme(legend.position = 'none') + 
  ggtitle('Sensitivity of B') 

ngangao_plots$p2 <- ngangao_plots$p2 + 
  ggtitle('') + 
  x_axis_remover 

ngangao_plots$p3 <- ngangao_plots$p3 +
  ggtitle('') + 
  x_axis_remover + 
  theme(legend.position = 'none') 

##########
# plots for chawia outliers
##########
chawia_plots <- 
  plot_struct_fsens_results('./R_scripts/structure/data/stru_fsens_chawia.npz')
chawia_plots$p1 <- chawia_plots$p1 + 
  ylab('Propn. purple') + 
  theme(legend.position = 'none') + 
  ggtitle('Sensitivity of C')

chawia_plots$p2 <- chawia_plots$p2 + 
  ggtitle('') 

chawia_plots$p3 <- chawia_plots$p3 + 
  ggtitle('') + 
  theme(legend.position = 'none')

layout_matrix <- matrix(c(1, 1, 1, 2:10), 
                        nrow = 4, 
                        byrow = TRUE)
# grid.arrange(p_admix, 
#              mbololo_plots$p1, mbololo_plots$p2, mbololo_plots$p3,
#              ngangao_plots$p1, ngangao_plots$p2, ngangao_plots$p3,
#              chawia_plots$p1, chawia_plots$p2, chawia_plots$p3,
#              layout_matrix = layout_matrix)

g <- arrangeGrob(p_admix, 
                 mbololo_plots$p1, mbololo_plots$p2, mbololo_plots$p3,
                 ngangao_plots$p1, ngangao_plots$p2, ngangao_plots$p3,
                 chawia_plots$p1, chawia_plots$p2, chawia_plots$p3,
                 layout_matrix = layout_matrix)

ggsave('./R_scripts/structure/figures_tmp/fsens_structure.png', g,
       width = 7, height = 6)
