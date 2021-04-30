#################
# Function to make a trace plot of posterior statistics
#################
plot_post_stat_trace_plot <- function(results_df, 
                                      abbreviate_legend = FALSE){
  
  # results_df is a data frame with three columns: 
  # t, the prior parameter
  # refit, the refitted values
  # and lin, the lr values
  
  trace_df <- results_df %>%
    gather(key = method, value = y, c(refit, lin)) 
  
  
  if(abbreviate_legend == FALSE){
    trace_df$method[trace_df$method == 'lin'] <- 
      'linear approx.'
  }else{
    trace_df$method[trace_df$method == 'lin'] <- 
      'lin.'
  }
  
  
  p <- trace_df %>%
    ggplot(aes(x = t, y = y, color = method, shape = method)) + 
    geom_point() + 
    geom_line() + 
    scale_color_brewer(palette = 'Set1') + 
    theme(legend.title = element_blank(),
          legend.position = 'bottom') + 
    get_fontsizes()
  
  return(p)
}


#################
# Function to plot the influence function
#################
plot_influence_and_logphi <- function(logit_v_grid, 
                                      infl_fun, 
                                      log_phi, 
                                      # option to have different gridpoints
                                      # for log_phi
                                      # (useful for worst-case ones, 
                                      # where we want higher resolution)
                                      logit_v_grid_logphi = NULL, 
                                      # some more flexible option
                                      y_limits = NULL, 
                                      scale = NULL){
  
  # plots the influence function overlayed with a pertrubation log_phi
  
  # logit_v_grid, infl_fun, log_phi are vectors of grid-points, 
  # the influence function evaluated at the gridpoints, 
  # and the perturbation evaluated at the gridpoints, respectively. 
  
  # optionally we might use a different grid for log_phi
  # in which case we pass in logit_v_grid_logphi 
  # the length of log_phi and logit_v_grid_logphi must match
  
  stopifnot(length(logit_v_grid) == length(infl_fun))
  
  if(is.null(logit_v_grid_logphi)){
    logit_v_grid_logphi <- logit_v_grid
  }
  stopifnot(length(logit_v_grid_logphi) == length(log_phi))
  
  # scale the infl so it matches the log phi
  infl_norm <- max(abs(infl_fun))
  log_phi_norm = max(abs(log_phi))
  
  if(is.null(scale)){
    scale <- log_phi_norm / infl_norm
  }
  
  log_phi_df <- 
    data.frame(x = logit_v_grid_logphi, 
               f = log_phi, 
               func_name = 'a_phi')
  
  infl_df <- 
    data.frame(x = logit_v_grid, 
               f = infl_fun * scale, 
               func_name = 'b_psi')
  
  df <- rbind(log_phi_df, 
              infl_df)
  
  p_logphi <- 
    ggplot() +
    # the perturbtion
    geom_area(data = NULL, 
              aes(x = logit_v_grid_logphi, 
                  y = log_phi), 
              fill = 'grey') + 
    # perturbation and influence function
    geom_line(data = df, 
              aes(x = x,
                  y = f,
                  color = func_name)) +
    # set colors
    scale_color_manual(values = c('black', 
                                 'purple'), 
                       breaks = c("a_phi", "b_psi"),
                       labels = c(expression(varphi), 
                                  expression(Psi))) +
    # x-axis
    geom_hline(yintercept = 0., alpha = 0.5) + 
    ylab('value') + 
    xlab(TeX('logit(stick propn.)')) +
    # xlab(TeX('logit($\\nu_k$)')) +
    ggtitle('Perturbation') +
    get_fontsizes() + 
    theme(legend.title = element_blank(), 
          legend.position = 'bottom')

  return(p_logphi)
}

#################
# Function plot prior densities
#################
plot_priors <- function(logit_v_grid, p0, pc){
  p_priors <- 
    data.frame(logit_v_grid = logit_v_grid, 
               P0 = p0, 
               P1 = pc) %>%
    gather(key = prior, value = P, -logit_v_grid) %>% 
    ggplot() + 
    geom_line(aes(x = logit_v_grid, 
                  y = P, 
                  color = prior)) + 
    scale_color_manual(values = c('lightblue', 'blue')) + 
    xlab('stick propn.') + 
    # xlab(TeX('$\\nu_k$')) + 
    ggtitle('Priors') + 
    get_fontsizes() + 
    theme(legend.position = 'bottom', 
          legend.title = element_blank())
  
  return(p_priors)
}

sigmoid <- function(x){
  return(1 / (1 + exp(-x)))
}



