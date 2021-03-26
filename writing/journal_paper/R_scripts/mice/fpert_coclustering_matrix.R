fpert_coclust_file <-
  np$load('./R_scripts/mice/data/functional_coclustering_pert.npz')

######################
# Plot priors
######################

p_logphi <- plot_influence_and_logphi(logit_v_grid, 
                                      infl_data['influence_grid_x_prior'], 
                                      fpert_coclust_file['log_phi'])

p_priors <- 
  data.frame(logit_v_grid = fpert_coclust_file['logit_v_grid'], 
             p0 = fpert_coclust_file['p0_logit'], 
             p1 = fpert_coclust_file['pc_logit']) %>%
  gather(key = prior, value = p, -logit_v_grid) %>% 
  ggplot() + 
  geom_line(aes(x = logit_v_grid, 
                y = p, 
                color = prior)) + 
  scale_color_manual(values = c('lightblue', 'blue')) + 
  xlab('logit stick') + 
  ggtitle('priors in logit space') + 
  fontsize_theme + 
  theme(legend.position = 'none')

p_priors_contr <- 
  data.frame(logit_v_grid = sigmoid(fpert_coclust_file['logit_v_grid']), 
             p0 = fpert_coclust_file['p0_constrained'], 
             p1 = fpert_coclust_file['pc_constrained']) %>%
  gather(key = prior, value = p, -logit_v_grid) %>% 
  ggplot() + 
  geom_line(aes(x = logit_v_grid, 
                y = p, 
                color = prior)) + 
  scale_color_manual(values = c('lightblue', 'blue')) + 
  xlab('stick') + 
  ggtitle('priors in constrained space') + 
  fontsize_theme + 
  theme(legend.title = element_blank(), 
        legend.position = c(0.85, 0.75))

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
labels <- c('<-1e-1', '(-1e-1, -1e-2]', '(-1e-2, -1e-3]', 
            '(-1e-3, 1e-3]', '(1e-3, 1e-2]', '(1e-2, 1e-1]', '>1e-1')

min_keep = 1e-3 # in the scatter-plot, grey out these values
breaks = c(1e0, 1e2) # breaks for the contours

plots <- compare_coclust_lr_and_refit(coclust_refit_fpert, 
                                      coclust_lr_fpert,
                                      coclust_init, 
                                      limits,
                                      labels,
                                      min_keep,
                                      breaks)

layout_matrix <- matrix(c(1, 2, 3, 4, 5, 5), 
                        nrow = 2, byrow = TRUE)
g <- arrangeGrob(p_logphi, p_priors, p_priors_contr, 
                 plots$p_scatter, plots$p_coclust, 
                 layout_matrix = layout_matrix)
ggsave('./R_scripts/mice/figures_tmp/fpert_coclust_sensitivity.png', 
       g, width = 8, height = 5.)
