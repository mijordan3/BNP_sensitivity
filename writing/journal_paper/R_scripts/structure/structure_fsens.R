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

p_admix <- out$p + 
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
  theme(axis.text.x = element_blank(), 
        axis.ticks.x = element_blank())


#################
# plot results
#################

# alpha for population coloring box
pop_box_alpha <- 0.05

plot_struct_fsens_results <- function(results_list, 
                                      pop_color){
  
  p_logphi <- plot_influence_and_logphi(results_list$infl_df$logit_v, 
                                        results_list$infl_df$infl_x_prior, 
                                        results_list$pert_df$log_phi, 
                                        results_list$pert_df$logit_v)
  
  p_priors <- plot_priors(sigmoid(results_list$pert_df$logit_v),
                          results_list$pert_df$p0,
                          results_list$pert_df$pc) 
  
  
  results_df <- 
    data.frame(t = results_list$sensitivity_df$epsilon, 
               refit = results_list$sensitivity_df$refit, 
               lin = results_list$sensitivity_df$lr)
  
  p_sens <- plot_post_stat_trace_plot(results_df, 
                                      abbreviate_legend = TRUE) + 
    ggtitle('Sensitivity') + 
    xlab('t') + 
    # add coloring for population 
    geom_rect(aes(xmin=-Inf, xmax=Inf, ymin=-Inf, ymax=Inf), 
              alpha = pop_box_alpha,
              fill = pop_color,
              color = 'white',
              show.legend = FALSE) 
  
  p_sens <- move_layers(p_sens, 'GeomRect', position = 'bottom')
    
  
  return(list(p_logphi = p_logphi, 
              p_priors = p_priors, 
              p_sens = p_sens))
}

##########
# plots for mbololo outliers
##########
panel_size <- 1.5

x_axis_remover <- 
  theme(axis.title.x = element_blank(),
        axis.text.x = element_blank(), 
        legend.position = 'none') 

title_remover <- ggtitle(NULL)

mbololo_plots <- 
  plot_struct_fsens_results(mbololo_fsens_results, 
                            pop_color = pop2_color)

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
  ylab('propn. pop2') + 
  x_axis_remover 


mbololo_plots_sum <- 
  mbololo_plots$p_logphi + 
  mbololo_plots$p_priors + 
  mbololo_plots$p_sens


##########
# plots for ngangao outliers
##########
ngangao_plots <- 
  plot_struct_fsens_results(ngangao_fsens_results, 
                            pop_color = pop1_color)

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
  ylab('propn. pop1') + 
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
  plot_struct_fsens_results(chawia_fsens_results, 
                            pop_color = pop3_color)

chawia_plots$p_logphi <- chawia_plots$p_logphi + 
  ggtitle('worst-case pert. of C')

chawia_plots$p_priors <- chawia_plots$p_priors + 
  title_remover

chawia_plots$p_sens <- chawia_plots$p_sens + 
  title_remover + 
  ylab('propn. pop3')

chawia_plots_sum <- 
  chawia_plots$p_logphi + 
  chawia_plots$p_priors + 
  chawia_plots$p_sens


p_admix <- p_admix + theme(legend.position = 'top')

p_admix / mbololo_plots_sum / ngangao_plots_sum / chawia_plots_sum + 
  plot_layout(heights = c(1.25, 1, 1, 1))
