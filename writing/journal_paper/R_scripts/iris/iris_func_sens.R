plot_step_pert_results <- function(input_file,
                                   remove_legend = FALSE, 
                                   remove_xlab = FALSE, 
                                   remove_title = FALSE){
  stepfun_file <- np$load(input_file)
  logit_v_grid <- stepfun_file['logit_v_grid']
  
  p1 <- ggplot() +
    geom_line(aes(x = logit_v_grid, 
                  y = stepfun_file['influence_x_prior_grid']), 
              color = 'purple') + 
    geom_hline(yintercept = 0., alpha = 0.5) + 
    ylab('influence x p0') + 
    xlab('logit-stick') + 
    ggtitle('log-phi') + 
    geom_rect(aes(xmin=stepfun_file['mu1'],
                  xmax=stepfun_file['mu2'],
                  ymin=-Inf, ymax=Inf), 
              color = 'grey', 
              fill = 'grey', 
              alpha = 0.5) +
    get_fontsizes()
  
  
  p2 <- plot_priors(logit_v_grid,
                    p0 = stepfun_file['p0'],
                    pc = stepfun_file['p1']) + 
    theme(legend.position = 'bottom', 
          legend.title = element_blank())
  
  p3 <- plot_post_stat_trace_plot(stepfun_file['epsilon_vec'], 
                                  stepfun_file['refit'], 
                                  stepfun_file['lr']) + 
    ylab('g(pert) - g(init)') + 
    xlab('epsilon') + 
    ggtitle('Sensitivity') + 
    theme(legend.position = 'bottom', 
          legend.title = element_blank())
  
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

g0 <- plot_step_pert_results('./R_scripts/iris/data/iris_fsens_muindx0.npz', 
                             remove_legend = TRUE, 
                             remove_xlab = TRUE, 
                             remove_title = FALSE)

g1 <- plot_step_pert_results('./R_scripts/iris/data/iris_fsens_muindx1.npz', 
                             remove_legend = TRUE, 
                             remove_xlab = TRUE, 
                             remove_title = TRUE)

g2 <- plot_step_pert_results('./R_scripts/iris/data/iris_fsens_muindx2.npz', 
                             remove_legend = TRUE, 
                             remove_xlab = TRUE, 
                             remove_title = TRUE)

g3 <- plot_step_pert_results('./R_scripts/iris/data/iris_fsens_muindx3.npz', 
                             remove_legend = TRUE, 
                             remove_xlab = TRUE, 
                             remove_title = TRUE)

g4 <- plot_step_pert_results('./R_scripts/iris/data/iris_fsens_muindx4.npz', 
                             remove_legend = FALSE, 
                             remove_xlab = FALSE, 
                             remove_title = TRUE)



g0 / g1 / g2 / g3 / g4


# grid.arrange(g0, g1, g2, g3, g4, ncol = 1)

# g_all <- arrangeGrob(g0, g1, g2, g3, g4, ncol = 1)
# 
# ggsave('./R_scripts/iris/figures_tmp/iris_func_pert.png', 
#        g_all, 
#        width = 6, height = 8)
