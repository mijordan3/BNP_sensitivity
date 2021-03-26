plot_initial_fit <- function(){
  stru_init_file <- np$load('./R_scripts/structure/data/init_fit.npz')
  
  # we plot the top 7 clusters
  clusters_keep <- 7
  
  # get inferred admixtures
  ind_admix_matr <- stru_init_file['e_ind_admix'][, 1:clusters_keep]
  
  # number of observations
  n_obs <- dim(ind_admix_matr)[1]
  
  
  # create a data frame
  ind_admix_df <- 
    data.frame(ind_admix_matr) %>% 
    mutate(obs_id = 1:n(), 
           label = stru_init_file['labels'], 
           # the is the weight of "other" clusters
           # start w z so its the last factor ... 
           z_other = 1 - rowSums(ind_admix_matr)) %>% 
    gather(key = cluster, value = admix, -c('obs_id', 'label'))
  
  
  # these are manually entered
  # these separate out the true populations 
  intercepts <- c(80, 134, 138)
  ticks_loc <- c(intercepts[1] / 2,
                 (intercepts[1] + intercepts[2]) / 2, 
                 (intercepts[2] + intercepts[3]) / 2, 
                 (intercepts[3] + n_obs) / 2)
  labels <- c('Mbololo', 'Ngangao', 'Yale', 'Chawia')  
  
  # plot
  ind_admix_df %>% 
    ggplot() + 
    geom_col(aes(x = obs_id,
                 y = admix, 
                 fill = cluster, 
                 color = cluster)) + 
    scale_fill_brewer(palette = 'Set2') + 
    scale_color_brewer(palette = 'Set2') + 
    # flip y-axis and get rid of grey space
    ylim(c(1, 0)) + 
    coord_cartesian(xlim = c(0, n_obs), 
                    ylim = c(1, 0), 
                    expand = FALSE) + 
    # add separators for the ture populations
    geom_vline(xintercept = intercepts, 
               linetype = 'dashed') +
    # add labels
    scale_x_continuous(breaks=ticks_loc,
                       labels = labels) + 
    scale_y_continuous(breaks=NULL) + 
    theme(axis.title = element_blank(), 
          axis.text.y = element_blank(), 
          legend.position = 'none', 
          axis.text.x = element_text(angle = 45, 
                                     hjust = 1, 
                                     size = axis_ticksize)) 
  
}



