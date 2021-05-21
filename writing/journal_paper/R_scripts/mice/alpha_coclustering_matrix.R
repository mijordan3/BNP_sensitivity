###################
# results at alpha = 1
###################
min_keep <- 1e-4 # in the scatter-plot, grey out these values
vmin <- 1e-5 # what we define as nonzero in the scatter plot

plots_alpha1 <- compare_coclust_lr_and_refit(coclust_refit_alpha1, 
                                      coclust_lr_alpha1,
                                      coclust_init, 
                                      vmin = vmin,
                                      min_keep = min_keep)

plots_alpha1$p_scatter <- plots_alpha1$p_scatter + 
  ggtitle(TeX(paste0('$\\alpha = $', alpha_pert1))) + 
  theme(title = element_text(size = title_size))

plots_alpha1_summed <- 
  plots_alpha1$p_scatter + 
  plots_alpha1$p_coclust_refit + theme(legend.position = 'none') + 
  plots_alpha1$p_coclust_lr
  
###################
# results at alpha = 11
###################
plots_alpha11 <- compare_coclust_lr_and_refit(coclust_refit_alpha11, 
                                              coclust_lr_alpha11,
                                              coclust_init, 
                                              vmin = vmin,
                                              min_keep = min_keep)

plots_alpha11$p_scatter <- plots_alpha11$p_scatter +
  ggtitle(TeX(paste0('$\\alpha = $', alpha_pert2))) + 
  theme(title = element_text(size = title_size))


plots_alpha11_summed <- 
  plots_alpha11$p_scatter + 
  plots_alpha11$p_coclust_refit + 
  plots_alpha11$p_coclust_lr


plots_alpha1_summed / plots_alpha11_summed

