######################
# Plot priors
######################

p_logphi <- plot_influence_and_logphi(prior_pert_df$logit_v, 
                                      influence_df$influence_x_prior, 
                                      prior_pert_df$log_phi)

p_priors <- 
  plot_priors(prior_pert_df$logit_v, 
              p0 = prior_pert_df$p0_logit, 
              pc = prior_pert_df$pc_logit) + 
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
