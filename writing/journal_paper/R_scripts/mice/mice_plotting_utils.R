################
# Plotting co-clustering matrix
################
plot_coclustering <- function(coclustering_df, 
                              x_name = 'gene1', 
                              y_name = 'gene2', 
                              value = 'coclustering'){
  # expects the coclustering to be already in long format
  # with columns x_name, yname, and value
  
  # change to be consistent names
  colnames(coclustering_df)[colnames(coclustering_df) == x_name] <- 'x'
  colnames(coclustering_df)[colnames(coclustering_df) == y_name] <- 'y'
  colnames(coclustering_df)[colnames(coclustering_df) == value] <- 'value'
  
  n_obs1 <- length(unique(coclustering_df$x))
  n_obs2 <- length(unique(coclustering_df$y))
    
  p <- 
    ggplot(coclustering_df, 
           aes(x, y, fill= value)) +
    # raster might be faster than geom_tile?
    geom_raster() + 
    coord_cartesian(xlim = c(0, n_obs1), 
                    ylim = c(0, n_obs2), 
                    expand = FALSE) + 
    ylab('gene') + 
    xlab('gene') + 
    theme_bw() + 
    get_fontsizes() + 
    theme(legend.title = element_blank())
  
  return(p)
}

################
# Plotting differences in co-clustering
################
get_coclust_diff <- function(coclust1, coclust2){
  
  coclust_diff <- 
    inner_join(coclust1, coclust2, by = c('gene1', 'gene2')) %>% 
    mutate(diff = coclustering.x - coclustering.y) 
  
  return(coclust_diff)
}


plot_coclust_diff <- function(coclust_diff, vmin){
  
  breaks <- c(-Inf, -vmin, vmin, Inf)
  labels <- c('-', '0', '+')
  
  p <- coclust_diff %>%
    mutate(diff_bin = cut(diff, breaks, labels = labels)) %>% 
    plot_coclustering(value = 'diff_bin') + 
    scale_fill_brewer(breaks = c('-', '+'), 
                      palette = 'RdBu', 
                      direction = -1, 
                      name = TeX("sign($\\Delta$)")) + 
    theme(legend.title = element_text(size = axis_title_size))
  
  return(p)
}



compare_coclust_lr_and_refit_scatter <-
  function(coclust_refit, 
           coclust_lr, 
           coclust_init, 
           min_keep = 1e-3){
  
  # compute diffs for the refit
  coclust_refit_diff <- get_coclust_diff(coclust_refit, 
                                         coclust_init) %>%
    rename(refit_diff = diff)
  
  # then, the linear response
  coclust_lr_diff <- get_coclust_diff(coclust_lr, 
                                      coclust_init) %>%
    rename(lr_diff = diff)
  
  
  # join
  diffs <- inner_join(coclust_refit_diff, 
                      coclust_lr_diff, 
                      by = c('gene1', 'gene2')) %>%
    filter(abs(refit_diff) > min_keep | abs(lr_diff) > min_keep)
  
  limit_min <- min(c(diffs$refit_diff, diffs$lr_diff))
  limit_max <- max(c(diffs$refit_diff, diffs$lr_diff))
  
  p <- ggplot(data = diffs, 
              aes(x = refit_diff, y = lr_diff)) +
    # the area we excluded
    geom_rect(xmin = -min_keep, xmax = min_keep, 
              ymin = -min_keep, ymax = min_keep, 
              fill = 'grey', color = 'black') + 
    # identity line
    geom_abline(slope = 1, intercept = 0, color = 'red') +
    # the points
    geom_point(alpha = 0.1, shape = 16, size = 1) +
    # stuff for axes
    ylab('lin. - init') + 
    xlab('refit - init') +
    ylim(c(limit_min, limit_max)) + 
    xlim(c(limit_min, limit_max)) + 
    get_fontsizes()
  
  return(p)
  }

compare_coclust_lr_and_refit <- function(coclust_refit, 
                                 coclust_lr, 
                                 coclust_init,
                                 vmin, 
                                 min_keep = 1e-3){
  
  # make scatter plot
  p_scatter <- compare_coclust_lr_and_refit_scatter(coclust_refit, 
                                                    coclust_lr,
                                                    coclust_init, 
                                                    min_keep) 
  
  # get heatmaps 
  # compute diffs
  coclust_diff_refit <- get_coclust_diff(coclust_refit, coclust_init) 
  coclust_diff_lr <- get_coclust_diff(coclust_lr, coclust_init)
  
  # make coclustering matrix 
  p_coclust_refit <-
    coclust_diff_refit %>% 
    plot_coclust_diff(vmin) + 
    ggtitle('refit - init') + 
    theme(legend.position = 'bottom', 
          legend.key.height = unit(0.2, 'cm'),
          legend.key.width = unit(0.4, 'cm'))
  
  p_coclust_lr <-
    coclust_diff_lr %>% 
    plot_coclust_diff(vmin) + 
    ggtitle('lin. - init') + 
    theme(axis.title.y = element_blank(), 
          axis.ticks.y = element_blank(), 
          axis.text.y = element_blank(),
          legend.position = 'none') 
  
  return(list(p_scatter = p_scatter,
              p_coclust_refit = p_coclust_refit, 
              p_coclust_lr = p_coclust_lr))
}
