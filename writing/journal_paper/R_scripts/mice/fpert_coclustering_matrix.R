######################
# Plot perturbation
######################
p_logphi <- plot_influence_and_logphi(prior_pert_df$logit_v, 
                                      influence_df$influence_x_prior, 
                                      prior_pert_df$log_phi)

######################
# Plot priors
######################
p_priors_contr <- 
  plot_priors(sigmoid(prior_pert_df$logit_v), 
             p0 = prior_pert_df$p0_constr, 
             pc = prior_pert_df$pc_constr) # + 
  # legend.theme

######################
# Plot co-clustering results
######################
# bins for the co-clustering matrix
vmin = 1e-5
plots <- compare_coclust_lr_and_refit(coclust_refit_fpert, 
                                      coclust_lr_fpert,
                                      coclust_init, 
                                      vmin = vmin)

top_row <- p_logphi + p_priors_contr
bottom_row <- plots$p_scatter + plots$p_coclust_refit + plots$p_coclust_lr 

# grid.arrange(patchworkGrob(top_row), 
#              patchworkGrob(bottom_row))

top_row / bottom_row
