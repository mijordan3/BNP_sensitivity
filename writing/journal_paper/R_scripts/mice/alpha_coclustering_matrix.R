###################
# results at alpha = 1
###################
alpha1_coclust_file <- np$load('./R_scripts/mice/data/coclustering_alpha1.0.npz')

coclust_refit1 <- 
  load_coclust_file(alpha1_coclust_file, 'coclust_refit') 


coclust_lr1 <-
  load_coclust_file(alpha1_coclust_file, 'coclust_lr') 

# bins for the co-clustering matrix
limits <- c(1e-5, 1e-4, 1e-3, Inf)
limit_labels <- construct_limit_labels(limits)

min_keep = 1e-4 # in the scatter-plot, grey out these values
breaks = c(1e3, 1e4, 1e5, Inf) # breaks for the contours

plots_alpha1 <- compare_coclust_lr_and_refit(coclust_refit1, 
                                      coclust_lr1,
                                      coclust_init, 
                                      limits,
                                      limit_labels,
                                      min_keep,
                                      breaks)

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
alpha1_coclust_file <- np$load('./R_scripts/mice/data/coclustering_alpha11.0.npz')

coclust_refit11 <- 
  load_coclust_file(alpha1_coclust_file, 'coclust_refit') 


coclust_lr11 <-
  load_coclust_file(alpha1_coclust_file, 'coclust_lr') 

plots_alpha11 <- compare_coclust_lr_and_refit(coclust_refit11, 
                                      coclust_lr11,
                                      coclust_init, 
                                      limits,
                                      limit_labels,
                                      min_keep,
                                      breaks)

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
