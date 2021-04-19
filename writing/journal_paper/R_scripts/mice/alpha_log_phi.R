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
                          y_limits = c(-ymax, ymax), 
                          scale = scale) + 
  ggtitle('alpha -5') + 
  theme(axis.title.y.right = element_blank(),
        axis.text.y.right = element_blank(),
        axis.ticks.y.right = element_blank())



p2 <- plot_influence_and_logphi(influence_df$logit_v, 
                                influence_df$influence_x_prior, 
                                log_phi(influence_df$logit_v, 11), 
                                y_limits = c(-ymax, ymax), 
                                scale = scale) + 
  ggtitle('alpha +5')  + 
  theme(axis.title.y.left = element_blank(),
        axis.text.y.left = element_blank(),
        axis.ticks.y.left = element_blank())



p1 + p2
