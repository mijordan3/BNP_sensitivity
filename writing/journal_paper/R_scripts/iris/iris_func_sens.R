
plot_func_pert_results <- function(results_list,
                                   remove_legend = FALSE, 
                                   remove_xlab = FALSE, 
                                   remove_title = FALSE, 
                                   ymax = NULL){
  
  # plot influence function and perturbation
  p1 <- plot_influence_and_logphi(influence_df$logit_v, 
                            influence_df$influence_x_prior, 
                            results_list$priors_df$log_phi)
  
  # plot prior densities
  p2 <- plot_priors(results_list$priors_df$v,
                    p0 = results_list$priors_df$p0,
                    pc = results_list$priors_df$p1) 

  # the posterior quantity at the initial fit
  g0 <- results_list$sensitiivty_df$refit[1]
  stopifnot(g0 == results_list$sensitiivty_df$lr[1])
  
  # make trace plot
  results_df <- 
    data.frame(t = results_list$sensitiivty_df$epsilon, 
               refit = results_list$sensitiivty_df$refit - g0,
               lin = results_list$sensitiivty_df$lr - g0)
  
  p3 <- plot_post_stat_trace_plot(results_df, 
                                  abbreviate_legend = TRUE) + 
    ylab(expression(Delta*'E[# clusters]')) + 
    xlab('t') + 
    ggtitle('Sensitivity') + 
    geom_hline(yintercept = 0., color = 'black')
  
  if(!is.null(ymax)){
    p3 <- p3 + ylim(c(-ymax, ymax))
  }
    
  
  if(remove_legend){
    p1 <- p1 + theme(legend.position = 'none')
    p2 <- p2 + theme(legend.position = 'none')
    p3 <- p3 + theme(legend.position = 'none')
  }
  
  if(remove_xlab){
    p1 <- p1 + theme(axis.title.x = element_blank(), 
                     axis.text.x = element_blank())
    p2 <- p2 + theme(axis.title.x = element_blank(), 
                     axis.text.x = element_blank())
    p3 <- p3 + theme(axis.title.x = element_blank(), 
                     axis.text.x = element_blank())
  }
  
  if(remove_title){
    p1 <- p1 + ggtitle(NULL)
    p2 <- p2 + ggtitle(NULL)
    p3 <- p3 + ggtitle(NULL)
  }
  
  # g <- arrangeGrob(p1, p2, p3, nrow = 1)
  g = p1 + p2 +p3
  return(g)
}

# TODO this is set manually ... 
ymax = 0.018

g0 <- plot_func_pert_results(fpert1_results, 
                             remove_legend = TRUE, 
                             remove_xlab = TRUE, 
                             remove_title = FALSE, 
                             ymax = ymax)

g1 <- plot_func_pert_results(fpert3_results, 
                             remove_legend = TRUE, 
                             remove_xlab = TRUE, 
                             remove_title = TRUE, 
                             ymax = ymax)

g2 <- plot_func_pert_results(fpert2_results, 
                             remove_legend = FALSE, 
                             remove_xlab = FALSE, 
                             remove_title = TRUE, 
                             ymax = ymax)


g0 / g1 / g2

