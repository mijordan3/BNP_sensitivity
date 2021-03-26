# data frame concerning number of clusters
n_clusters_refit_df <-
  data.frame(alpha = alpha_sens_file['alpha_list'],
             n_clusters = alpha_sens_file['n_clusters_refit'], 
             n_clusters_thresh = alpha_sens_file['n_clusters_thresh_refit'], 
             method = 'refit')

n_clusters_lr_df <-
  data.frame(alpha = alpha_sens_file['alpha_list'],
             n_clusters = alpha_sens_file['n_clusters_lr'], 
             n_clusters_thresh = alpha_sens_file['n_clusters_thresh_lr'], 
             method = 'lr')


n_clusters_df <- rbind(n_clusters_refit_df, n_clusters_lr_df)

alpha0 <- alpha_sens_file['alpha0']

# plot number of clusters
p1 <- n_clusters_df %>% 
  ggplot(aes(x = alpha, y = n_clusters, color = method)) + 
  geom_point() + 
  geom_line() + 
  scale_color_brewer(palette = 'Dark2') +
  geom_vline(xintercept = alpha0, color = 'red', linetype = 'dashed') + 
  ylab('E[# clusters]') + 
  theme(legend.position = 'none') + 
  fontsize_theme

# plot number of clusters thresholded
p2 <- n_clusters_df %>% 
  ggplot(aes(x = alpha, y = n_clusters_thresh, color = method)) + 
  geom_point() + 
  geom_line() + 
  scale_color_brewer(palette = 'Dark2') + 
  geom_vline(xintercept = alpha0, color = 'red', linetype = 'dashed') + 
  ylab('E[# clusters thresh.]') + 
  theme(legend.key.size = unit(0.5, 'cm'), 
        legend.position = c(0.8, 0.3), 
        legend.title = element_blank()) + 
  fontsize_theme

grid.arrange(p1, p2, nrow = 1)
