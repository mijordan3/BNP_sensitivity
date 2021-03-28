plot_step_pert_results <- function(input_file, remove_legend = TRUE){
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
    theme(legend.position = c(0.8, 0.7), 
          legend.title = element_blank(), 
          legend.key.size = unit(0.3, 'cm'))
  
  p3 <- plot_post_stat_trace_plot(stepfun_file['epsilon_vec'], 
                                  stepfun_file['refit'], 
                                  stepfun_file['lr']) + 
    ylab('g(pert) - g(init)') + 
    xlab('epsilon') + 
    ggtitle(' ') + 
    theme(legend.position = c(0.2, 0.3), 
          legend.title = element_blank(), 
          legend.key.size = unit(0.4, 'cm'))
  
  if(remove_legend){
    p2 <- p2 + theme(legend.position = 'none')
    p3 <- p3 + theme(legend.position = 'none')
  }
  
  g <- arrangeGrob(p1, p2, p3, nrow = 1)
  
  return(g)
}

g0 <- plot_step_pert_results('./R_scripts/iris/data/iris_fsens_muindx0.npz', 
                             remove_legend = FALSE)
g1 <- plot_step_pert_results('./R_scripts/iris/data/iris_fsens_muindx1.npz')
g2 <- plot_step_pert_results('./R_scripts/iris/data/iris_fsens_muindx2.npz')
g3 <- plot_step_pert_results('./R_scripts/iris/data/iris_fsens_muindx3.npz')
g4 <- plot_step_pert_results('./R_scripts/iris/data/iris_fsens_muindx4.npz')

g_all <- arrangeGrob(g0, g1, g2, g3, g4, ncol = 1)

ggsave('./R_scripts/iris/figures_tmp/iris_func_pert.png', 
       g_all, 
       width = 6, height = 8)
