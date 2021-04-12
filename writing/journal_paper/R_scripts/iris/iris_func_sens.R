
plot_step_pert_results <- function(results_list,
                                   remove_legend = FALSE, 
                                   remove_xlab = FALSE, 
                                   remove_title = FALSE, 
                                   ymax = NULL){
  
  p1 <- plot_influence_and_logphi(influence_df$logit_v, 
                            influence_df$influence_x_prior, 
                            results_list$priors_df$log_phi) + 
    ggtitle('perturbation') + 
    theme(axis.ticks.y.right = element_blank(), 
          axis.text.y.right = element_blank())
  
  
  p2 <- plot_priors(results_list$priors_df$v,
                    p0 = results_list$priors_df$p0,
                    pc = results_list$priors_df$p1) + 
    xlab('stick length') + 
    theme(legend.position = 'bottom', 
          legend.title = element_blank())
  
  
  g0 <- results_list$sensitiivty_df$refit[1]
  stopifnot(g0 == results_list$sensitiivty_df$lr[1])
  p3 <- plot_post_stat_trace_plot(results_list$sensitiivty_df$epsilon, 
                                  results_list$sensitiivty_df$refit - g0,
                                  results_list$sensitiivty_df$lr - g0) + 
    # ylab('g(pert) - g(init)') + 
    ylab(expression(Delta*'E[# clusters]')) + 
    xlab('epsilon') + 
    ggtitle('sensitivity') + 
    geom_hline(yintercept = 0., color = 'black') + 
    theme(legend.position = 'bottom', 
          legend.title = element_blank()) 
  
  if(!is.null(ymax)){
    p3 <- p3 + ylim(c(-ymax, ymax))
  }
    
  
  if(remove_legend){
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

g0 <- plot_step_pert_results(fpert1_results, 
                             remove_legend = TRUE, 
                             remove_xlab = TRUE, 
                             remove_title = FALSE, 
                             ymax = ymax)

g1 <- plot_step_pert_results(fpert2_results, 
                             remove_legend = TRUE, 
                             remove_xlab = TRUE, 
                             remove_title = TRUE, 
                             ymax = ymax)

g2 <- plot_step_pert_results(fpert3_results, 
                             remove_legend = TRUE, 
                             remove_xlab = TRUE, 
                             remove_title = TRUE, 
                             ymax = ymax)

g3 <- plot_step_pert_results(wc_results, 
                             remove_legend = FALSE, 
                             remove_xlab = FALSE, 
                             remove_title = TRUE)



g0 / g2 / g1 / g3


# grid.arrange(g0, g1, g2, g3, g4, ncol = 1)

# g_all <- arrangeGrob(g0, g1, g2, g3, g4, ncol = 1)
# 
# ggsave('./R_scripts/iris/figures_tmp/iris_func_pert.png', 
#        g_all, 
#        width = 6, height = 8)
