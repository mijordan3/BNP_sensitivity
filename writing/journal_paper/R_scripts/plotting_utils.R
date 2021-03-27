#################
# Function to make a trace plot of posterior statistics
#################
plot_post_stat_trace_plot <- function(alpha_list,
                                      refit_list,
                                      lr_list){
  
  trace_df <- data.frame(alpha = alpha_list, 
                         refit = refit_list, 
                         lr = lr_list) %>% 
    gather(key = method, value = value, -alpha)
  
  p <- trace_df %>%
    ggplot(aes(x = alpha, y = value, color = method)) + 
    geom_point() + 
    geom_line() + 
    scale_color_brewer(palette = 'Dark2') + 
    get_fontsizes()
  return(p)
}


#################
# Function to plot the influence function
#################
plot_influence_and_logphi <- function(logit_v_grid, 
                                      infl_fun, 
                                      log_phi){
  
  # scale the infl so it matches the log phi
  infl_norm <- max(abs(infl_fun))
  log_phi_norm = max(abs(log_phi))
  scale <- log_phi_norm / infl_norm
  
  p_logphi <- 
    ggplot() + 
    # plot influnce function
    geom_line(aes(x = logit_v_grid, 
                  y = infl_fun * scale), 
              color = 'purple') + 
    # plot functional perturbation 
    geom_area(aes(x = logit_v_grid, 
                  y = log_phi), 
              fill = 'grey', color = 'black', alpha = 0.5) + 
    scale_y_continuous(  
      # Features of the first axis
      name = "log phi",
      # Add a second axis and specify its features
      sec.axis = sec_axis(~.*1/scale, name="influence x p0")
    ) + 
    geom_hline(yintercept = 0., alpha = 0.5) + 
    ylab('influence x p0') + 
    xlab('logit-stick') + 
    ggtitle('log phi') +
    theme(axis.title.y.right = element_text(color = 'purple', 
                                            size = axis_title_size), 
          axis.text.y.right = element_text(color = 'purple', 
                                           size = axis_ticksize)) + 
    get_fontsizes()
  
  return(p_logphi)
}

#################
# Function to make a trace plot priors
#################
plot_priors <- function(logit_v_grid, p0, pc){
  p_priors <- 
    data.frame(logit_v_grid = logit_v_grid, 
               p0 = p0, 
               p1 = pc) %>%
    gather(key = prior, value = p, -logit_v_grid) %>% 
    ggplot() + 
    geom_line(aes(x = logit_v_grid, 
                  y = p, 
                  color = prior)) + 
    scale_color_manual(values = c('lightblue', 'blue')) + 
    xlab('logit stick') + 
    ggtitle('priors in logit space') + 
    get_fontsizes() 
  
  return(p_priors)
}

sigmoid <- function(x){
  return(1 / (1 + exp(-x)))
}



