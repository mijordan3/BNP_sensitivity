fpert_coclust_file <-
  np$load('./R_scripts/mice/data/functional_coclustering_gauss_pert1.npz')

######################
# Plot priors
######################

p_logphi <- plot_influence_and_logphi(fpert_coclust_file['logit_v_grid'], 
                                      infl_data['influence_grid_x_prior'], 
                                      fpert_coclust_file['log_phi'])

p_priors <- 
  plot_priors(fpert_coclust_file['logit_v_grid'], 
              p0 = fpert_coclust_file['p0_logit'], 
              pc = fpert_coclust_file['pc_logit']) + 
  theme(legend.position = 'none')

p_priors_contr <- 
  plot_priors(sigmoid(fpert_coclust_file['logit_v_grid']), 
             p0 = fpert_coclust_file['p0_constrained'], 
             pc = fpert_coclust_file['pc_constrained']) + 
  theme(legend.title = element_blank(), 
        legend.position = c(0.75, 0.85), 
        legend.key.size = unit(0.2, "cm"))

top_row <- p_logphi + p_priors + p_priors_contr

######################
# Plot co-clustering results
######################

coclust_refit_fpert <- 
  load_coclust_file(fpert_coclust_file, 'coclust_refit') 

coclust_lr_fpert <-
  load_coclust_file(fpert_coclust_file, 'coclust_lr') 

# foo2 = coclust_init$coclustering
# foo = load_coclust_file(fpert_coclust_file, 'coclust_init')$coclustering 

# bins for the co-clustering matrix
limits <- c(1e-3, 1e-2, 1e-1, Inf)
limit_labels <- construct_limit_labels(limits)

plots <- compare_coclust_lr_and_refit(coclust_refit_fpert, 
                                      coclust_lr_fpert,
                                      coclust_init, 
                                      limits,
                                      limit_labels,
                                      min_keep,
                                      breaks)

bottom_row <- plots$p_scatter + plots$p_coclust_refit + plots$p_coclust_lr

top_row / bottom_row


# layout_matrix <- matrix(c(1, 2, 3, 4, 5, 5), 
#                         nrow = 2, byrow = TRUE)
# g <- arrangeGrob(p_logphi, p_priors, p_priors_contr, 
#                  plots$p_scatter, plots$p_coclust, 
#                  layout_matrix = layout_matrix)
# ggsave('./R_scripts/mice/figures_tmp/fpert_coclust_sensitivity.png', 
#        g, width = 8, height = 5.)
