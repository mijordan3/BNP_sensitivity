load_coclust_file <- function(file, key){
  # expects a coclustering matrix
  # converts the matrix to a long data frame
  
  coclust_wide <- as.data.frame(file[[key]])
  
  coclust <- coclust_wide %>% 
    mutate(gene1 = colnames(coclust_wide)) %>% 
    # make it a long data frame
    gather(key = gene2, value = coclustering, -gene1) %>%
    # clean up the gene names
    mutate(gene1 = sub('.', '', gene1), 
           gene2 = sub('.', '', gene2)) %>% 
    mutate(gene1 = as.numeric(gene1), 
           gene2 = as.numeric(gene2))
  return(coclust)
}

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
    theme_bw() + 
    theme(legend.title = element_blank(), 
          legend.key.width = unit(0.2,"cm"))
  
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


plot_coclust_diff <- function(coclust_diff, 
                             limits, 
                             limit_labels = NULL){
  
  # not enough discrete bins in color-brewer
  stopifnot(length(limits) < 5)
  # pass in only positive limits. we will symmetrize
  stopifnot(all(limits > 0))
  
  limits <- sort(c(-limits, limits))
  
  # get differenices
  coclust_diff <- coclust_diff %>% 
    # and create bins: we will be plotting bins, 
    # not the raw value
    mutate(diff_bins = cut(diff, limits, labels = limit_labels))
  
  p <- coclust_diff %>%
    plot_coclustering(value = 'diff_bins') + 
    scale_fill_brewer(type = "div", palette = 'RdBu', direction = -1)
  
  return(p)
}


# compare_coclust_lr_and_refit_matr <- function(coclust_refit, 
#                                          coclust_lr, 
#                                          coclust_init, 
#                                          limits, 
#                                          limit_labels = NULL){
#   
#   # Get differences in coclustering
#   
#   # first, the refit
#   coclust_refit_diff <- get_coclust_diff(coclust_refit, 
#                                          coclust_init) %>%
#     mutate(method = 'refit - init')
#   
#   # then, the linear response
#   coclust_lr_diff <- get_coclust_diff(coclust_lr, 
#                                       coclust_init) %>% 
#     mutate(method = 'lr - init')
#   
#   # combine: 
#   coclust_diff <- rbind(coclust_refit_diff, 
#                         coclust_lr_diff)
#   
#   # plot
#   p <- plot_coclust_diff(coclust_diff,
#                          limits = limits,
#                          limit_labels = limit_labels) +
#     facet_wrap(~method)
#   
#   return(p)
# }


compare_coclust_lr_and_refit_scatter <-
  function(coclust_refit, 
           coclust_lr, 
           coclust_init, 
           min_keep = 1e-3, 
           breaks = c(1e3, 1e4, 1e5, Inf)){
  
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
  
  p <- ggplot(data = diffs, 
              aes(x = refit_diff, y = lr_diff)) +
    # the area we excluded
    geom_rect(xmin = -min_keep, xmax = min_keep, 
              ymin = -min_keep, ymax = min_keep, 
              fill = 'grey', color = 'black') + 
    # identity line
    geom_abline(slope = 1, intercept = 0, color = 'red') +
    # the points
    geom_point(alpha = 0.1, shape = 'o', size = 1) +
    # 2d density
    # geom_density_2d(breaks = breaks) +
    # scale_fill_brewer(palette = 'PuBu') + 
    ylab('lr - init') + 
    xlab('refit - init') +
    get_fontsizes()
  
  return(p)
  }

compare_coclust_lr_and_refit <- function(coclust_refit, 
                                 coclust_lr, 
                                 coclust_init, 
                                 limits, 
                                 limit_labels = NULL, 
                                 min_keep = 1e-3, 
                                 breaks = c(1e3, 1e4, 1e5, Inf)){
  
  # make scatter plot
  p_scatter <- compare_coclust_lr_and_refit_scatter(coclust_refit, 
                                                    coclust_lr,
                                                    coclust_init, 
                                                    min_keep, 
                                                    breaks) 
  
  # make coclustering matrix 
  p_coclust_refit <-
    get_coclust_diff(coclust_refit, coclust_init) %>% 
    # sometimes missing the top bin 
    # this is hacky ... fix this
    select(gene1, gene2, diff) %>%
    rbind(data.frame(gene1 = 1001, gene2 = 1001, diff = 1e16)) %>%
    plot_coclust_diff(limits = limits, 
                      limit_labels = limit_labels) + 
    ggtitle('refit - init') + 
    theme(axis.text = element_blank(),
          axis.title = element_blank(),
          axis.ticks = element_blank(),
          plot.title = element_text(size = title_size), 
          legend.position = 'none')
  
  p_coclust_lr <-
    get_coclust_diff(coclust_lr, coclust_init) %>% 
    # sometimes missing the top bin 
    # this is hacky ... fix this
    select(gene1, gene2, diff) %>%
    rbind(data.frame(gene1 = 1001, gene2 = 1001, diff = 1e16)) %>%
    plot_coclust_diff(limits = limits, 
                      limit_labels = limit_labels) + 
    ggtitle('lr - init') + 
    theme(axis.text = element_blank(),
          axis.title = element_blank(),
          axis.ticks = element_blank(),
          legend.key.width = unit(0.2,"cm"), 
          legend.key.height = unit(0.2, "cm"),
          legend.margin=margin(-10,-10,-10,-10),
          plot.title = element_text(size = title_size), 
          legend.text = element_text(size = axis_ticksize))
  
  # p_coclust <- compare_coclust_lr_and_refit_matr(coclust_refit,
  #                                                coclust_lr,
  #                                                coclust_init,
  #                                                limits,
  #                                                limit_labels) + 
  #   get_fontsizes() + 
  #   # remove axis labels
  #   theme(axis.text.y = element_blank(), 
  #         axis.title.y = element_blank(), 
  #         # make it white so it doesnt show up
  #         # but not "removed" so that the spacing works out
  #         axis.title.x = element_text(color = 'white'),
  #         axis.text.x = element_text(color = 'white'),
  #         legend.key.width = unit(0.2,"cm"))
  
  return(list(p_scatter = p_scatter,
              p_coclust_refit = p_coclust_refit, 
              p_coclust_lr = p_coclust_lr))
}

