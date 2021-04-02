logbeta_pdf <- function(logit_v, alpha){
  
  v <- sigmoid(logit_v)
  
  return(log(dbeta(v, shape1 = 1, shape2 = alpha)))
}

log_phi <- function(logit_v, alpha1, alpha0 = 6){
  return(logbeta_pdf(logit_v, alpha1) - logbeta_pdf(logit_v, alpha0))
}

p1 <- plot_influence_and_logphi(influence_df$logit_v, 
                          influence_df$influence_x_prior, 
                          log_phi(influence_df$logit_v, 1)) + 
  ggtitle('alpha -5')
  

p2 <- plot_influence_and_logphi(influence_df$logit_v, 
                                influence_df$influence_x_prior, 
                                log_phi(influence_df$logit_v, 11)) + 
  ggtitle('alpha +5')


p1 + p2
