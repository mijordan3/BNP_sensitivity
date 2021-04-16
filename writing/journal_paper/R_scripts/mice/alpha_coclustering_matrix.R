###################
# results at alpha = 1
###################
vmax = 0.001
min_keep = 1e-4 # in the scatter-plot, grey out these values
breaks = c(1e3, 1e4, 1e5, Inf) # breaks for the contours

plots_alpha1 <- compare_coclust_lr_and_refit(coclust_refit_alpha1, 
                                      coclust_lr_alpha1,
                                      coclust_init, 
                                      vmax = vmax,
                                      min_keep = min_keep)

plots_alpha1$p_scatter <- plots_alpha1$p_scatter + 
  ggtitle('alpha = 1') + 
  theme(title = element_text(size = title_size))

plots_alpha1_summed <- 
  plots_alpha1$p_scatter + 
  plots_alpha1$p_coclust_refit + 
  plots_alpha1$p_coclust_lr
  
###################
# results at alpha = 11
###################
plots_alpha11 <- compare_coclust_lr_and_refit(coclust_refit_alpha11, 
                                              coclust_lr_alpha11,
                                              coclust_init, 
                                              vmax = vmax,
                                              min_keep = min_keep)

plots_alpha11$p_scatter <- plots_alpha11$p_scatter +
  ggtitle('alpha = 11') + 
  theme(title = element_text(size = title_size))


plots_alpha11_summed <- 
  plots_alpha11$p_scatter + 
  plots_alpha11$p_coclust_refit + 
  plots_alpha11$p_coclust_lr

plots_alpha1_summed / plots_alpha11_summed

# layout_matrix <- matrix(c(1, 3, 2, 4, 2, 4), ncol = 3)
# 
# # grid.arrange(plots$p_scatter, plots$p_coclust, 
# #               plots11$p_scatter, plots11$p_coclust, 
# #               layout_matrix = layout_matrix)
# 
# g <- arrangeGrob(plots$p_scatter, plots$p_coclust, 
#                  plots11$p_scatter, plots11$p_coclust,
#                  layout_matrix = layout_matrix)
# 
# ggsave('./R_scripts/mice/figures_tmp/alpha_coclust_sensitivity.png', 
#        g, width = 6, height = 4.5)
