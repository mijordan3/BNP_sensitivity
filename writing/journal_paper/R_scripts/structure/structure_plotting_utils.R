plot_structure_fit <- function(ind_admix_matr){
  
  n_obs <- dim(ind_admix_matr)[1]
  
  # create a data frame
  ind_admix_df <- 
    data.frame(ind_admix_matr) %>% 
    mutate(obs_id = 1:n(), 
           # the is the weight of "other" clusters
           # start w z so its the last factor ... 
           z_other = 1 - rowSums(ind_admix_matr)) %>% 
    gather(key = cluster, value = admix, -c('obs_id')) %>% 
    mutate(population = sub('X', '', cluster))
  
  
  p <- ind_admix_df %>%
    ggplot() + 
    # plot the admixture bars
    geom_col(aes(x = obs_id,
                 y = admix, 
                 fill = population, 
                 color = population)) + 
    scale_fill_brewer(palette = 'Set2', 
                      breaks = c('1', '2', '3')) + 
    scale_color_brewer(palette = 'Set2', 
                       breaks = c()) + 
    # flip y-axis and get rid of grey space
    coord_cartesian(xlim = c(0.5, n_obs+0.5), 
                    ylim = c(1, 0), 
                    expand = FALSE) + 
    scale_y_continuous(breaks=NULL) +
    theme(axis.title = element_blank(), 
          axis.text.y = element_blank(), 
          legend.title = element_text(size = axis_title_size),
          legend.text = element_text(size = axis_title_size),
          legend.position = 'bottom',
          legend.key.size = unit(0.4, 'cm'),
          legend.margin=margin(-6,-6,-6,-6),
          axis.text.x = element_text(angle = 45, 
                                     hjust = 1, 
                                     size = axis_title_size))
  
  return(list(ind_admix_df = ind_admix_df, 
              p = p))
}

plot_initial_fit <- function(add_geographic_labels = TRUE){

  # we plot the top 7 clusters
  clusters_keep <- 7
  
  n_obs <- dim(e_ind_admix_init)[1]
  out <- plot_structure_fit(e_ind_admix_init[, 1:clusters_keep])

  p <- out$p
  init_ind_admix_df <- out$ind_admix_df
  
  # these are manually entered
  # these separate out the true populations 
  intercepts <- c(80, 134, 138)
  ticks_loc <- c(intercepts[1] / 2,
                 (intercepts[1] + intercepts[2]) / 2, 
                 (intercepts[2] + intercepts[3]) / 2, 
                 (intercepts[3] + n_obs) / 2)
  labels <- c('Mbololo', 'Ngangao', 'Yale', 'Chawia')  
  
  # plot
  p <- p + 
    scale_y_continuous(breaks=NULL) 
  
  if(add_geographic_labels){
    # add separators for the ture populations
    
    p <- p + 
      geom_vline(xintercept = intercepts, 
                   linetype = 'dashed') +
      # add labels
      scale_x_continuous(breaks=ticks_loc,
                         labels = labels)
  }else{
    p <- p + 
      scale_x_continuous(breaks = NULL)
  }
  
  return(list(p = p, 
              init_ind_admix_df = init_ind_admix_df, 
              intercepts = intercepts))
}