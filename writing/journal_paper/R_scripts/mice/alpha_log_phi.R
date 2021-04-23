logbeta_pdf <- function(logit_v, alpha){
  
  v <- sigmoid(logit_v)
  
  return(log(dbeta(v, shape1 = 1, shape2 = alpha)))
}

log_phi <- function(logit_v, alpha1, alpha0 = 6){
  return(logbeta_pdf(logit_v, alpha1) - logbeta_pdf(logit_v, alpha0))
}


ymax <- 15
scale <- 550

p1 <- plot_influence_and_logphi(influence_df$logit_v, 
                          influence_df$influence_x_prior, 
                          log_phi(influence_df$logit_v, 1),
                          scale = scale) + 
  ggtitle(TeX('$\\alpha - \\alpha_0 =  -5$')) + 
  # manually set y-limits (so it matches with p2 below)
  scale_y_continuous(  
    # label for log phi
    name = TeX("$\\varphi$"),
    limits = c(-ymax, ymax)
  ) + 
  theme(legend.position = 'none')

p2 <- plot_influence_and_logphi(influence_df$logit_v, 
                                influence_df$influence_x_prior, 
                                log_phi(influence_df$logit_v, 11), 
                                scale = scale) + 
  ggtitle(TeX('$\\alpha - \\alpha_0 =  5$')) + 
  # remove left axis
  theme(axis.title.y.left = element_blank(),
        axis.text.y.left = element_blank(),
        axis.ticks.y.left = element_blank()) + 
  # add right axis 
  scale_y_continuous(  
    # Add a second axis for the influence function
    sec.axis = sec_axis(~.*1/scale,
                        TeX("$\\Psi$"))
  ) + 
  theme(axis.title.y.right = element_text(color = 'purple', 
                                          size = axis_title_size), 
        axis.text.y.right = element_text(color = 'purple', 
                                         size = axis_ticksize)) + 
  theme(legend.position = 'none')
  

p1 + p2
