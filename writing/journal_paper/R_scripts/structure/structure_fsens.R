out <- plot_initial_fit()

regions_df <- data.frame(obs_id = unique(out$init_ind_admix_df$obs_id), 
                         region = geographic_labels)
out$init_ind_admix_df <- 
  out$init_ind_admix_df %>% 
  inner_join(regions_df, by = 'obs_id')

#################
# label outliers
#################
mbololo_outliers <- 
  out$init_ind_admix_df %>%
  filter(region == 'Mbololo') %>% 
  filter(cluster == 'X2') %>% 
  filter(admix > 0.21)

ngangao_outliers <- 
  out$init_ind_admix_df %>%
  filter(region == 'Ngangao') %>% 
  filter(cluster == 'X1') %>% 
  filter(admix > 0.4)

chawia_outliers <- 
  out$init_ind_admix_df %>%
  filter(region == 'Chawia')


intercepts <- c(min(mbololo_outliers$obs_id) - 1,
                max(mbololo_outliers$obs_id) + 1, 
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

plot_struct_fsens_results <- function(results_list){
  
  p_logphi <- plot_influence_and_logphi(results_list$infl_df$logit_v, 
                                        results_list$infl_df$infl_x_prior, 
                                        results_list$pert_df$log_phi, 
                                        results_list$pert_df$logit_v) + 
    theme(axis.ticks.y.right = element_blank(), 
          axis.text.y.right = element_blank())
  
  p_priors <- plot_priors(sigmoid(results_list$pert_df$logit_v),
                          results_list$pert_df$p0,
                          results_list$pert_df$pc) + 
    xlab('stick') + 
    ggtitle('priors') + 
    theme(legend.title = element_blank(), 
          legend.position = 'bottom')
  
  
  p_sens <- plot_post_stat_trace_plot(results_list$sensitivity_df$epsilon, 
                                      results_list$sensitivity_df$refit, 
                                      results_list$sensitivity_df$lr) + 
    ggtitle('sensitivity') + 
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
  plot_struct_fsens_results(mbololo_fsens_results)

mbololo_plots$p_logphi <- 
  mbololo_plots$p_logphi + 
  ggtitle('worst-case pert. of A') + 
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
  plot_struct_fsens_results(ngangao_fsens_results)

ngangao_plots$p_logphi <- 
  ngangao_plots$p_logphi + 
  ggtitle('worst-case pert. of B') + 
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
  plot_struct_fsens_results(chawia_fsens_results)

chawia_plots$p_logphi <- chawia_plots$p_logphi + 
  ggtitle('worst-case pert. of C')

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
