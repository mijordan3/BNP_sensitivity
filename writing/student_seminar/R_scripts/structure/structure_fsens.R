out <- plot_initial_fit(add_geographic_labels = FALSE)

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


rect_alpha = 0.1
linesize = 0.5
text_height = 0.1
text_size = 3

mbololo_box <- geom_rect(aes(xmin = min(mbololo_outliers$obs_id) - 1.5,
                             xmax = max(mbololo_outliers$obs_id) + 1.5,
                             ymin = 0.15, 
                             ymax = 0.5), 
                         color = 'black', 
                         alpha = 0., 
                         size = linesize)

ngangao_box <- geom_rect(aes(xmin = min(ngangao_outliers$obs_id) - 1.5,
                             xmax = max(ngangao_outliers$obs_id) + 1.5,
                             ymin = 0.4, 
                             ymax = 1.1), 
                         color = 'black', 
                         alpha = 0., 
                         size = linesize)

chawia_box <- geom_rect(aes(xmin = min(chawia_outliers$obs_id) - 1.5,
                             xmax = max(chawia_outliers$obs_id),
                             ymin = 0.1, 
                             ymax = 0.65), 
                         color = 'black', 
                         alpha = 0., 
                         size = linesize)

p_admix_mbololo <- out$p + 
  mbololo_box + 
  # add letter labels 
  geom_text(aes(x = min(mbololo_outliers$obs_id) - 5, 
                y = 0.2, 
                label = 'A'), 
            size = text_size) + 
  theme(axis.text.x = element_blank(), 
        axis.ticks.x = element_blank())

p_admix_ngangao <- out$p + 
  ngangao_box + 
  geom_text(aes(x = min(ngangao_outliers$obs_id) - 5, 
                y = 0.5,
                label = 'B'), 
            size = text_size) + 
  theme(axis.text.x = element_blank(), 
        axis.ticks.x = element_blank())

p_admix_chawia <- out$p + 
  chawia_box + 
  geom_text(aes(x = median(chawia_outliers$obs_id),
                y = 0.75, 
                label = 'C'), 
            size = text_size) + 
  theme(axis.text.x = element_blank(), 
        axis.ticks.x = element_blank())


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
mbololo_plots <- 
  plot_struct_fsens_results(mbololo_fsens_results)

mbololo_plots$p_logphi <- 
  mbololo_plots$p_logphi + 
  ggtitle('worst-case pert. of A') + 
  get_fontsizes() 

mbololo_plots$p_priors <- 
  mbololo_plots$p_priors + 
  get_fontsizes() 

mbololo_plots$p_sens <-
  mbololo_plots$p_sens + 
  ylab('propn. orange') 

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
  get_fontsizes() 

ngangao_plots$p_priors <- 
  ngangao_plots$p_priors + 
  get_fontsizes() 

ngangao_plots$p_sens <-
  ngangao_plots$p_sens + 
  ylab('propn. green') 

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

chawia_plots$p_priors <- chawia_plots$p_priors 


chawia_plots$p_sens <- chawia_plots$p_sens + 
  ylab('propn. purple')

chawia_plots_sum <- 
  chawia_plots$p_logphi + 
  chawia_plots$p_priors + 
  chawia_plots$p_sens


p_admix_mbololo / mbololo_plots_sum
save_last_fig('./figures/structure_mbololo_sens.png', 
              aspect_ratio = 0.6)


p_admix_ngangao / ngangao_plots_sum
save_last_fig('./figures/structure_ngangao_sens.png', 
              aspect_ratio = 0.6)

p_admix_chawia / chawia_plots_sum
save_last_fig('./figures/structure_chawia_sens.png', 
              aspect_ratio = 0.6)
