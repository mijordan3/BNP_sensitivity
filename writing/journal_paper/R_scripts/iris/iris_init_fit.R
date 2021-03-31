iris_fit_file <- np$load('./R_scripts/iris/data/iris_fit.npz')

##################
# plot iris data (has already been transformed into PC space)
##################
iris_obs <- as.data.frame(iris_fit_file['pc_iris_obs']) %>%
  mutate(est_z = iris_fit_file['cluster_memberships']) %>%
  mutate(est_z = as.factor(est_z))

# colors: this is 3-class Set2 from colorbrewer
# didn't just use scale_color_brewer bc we will manually label 
# the covariances with these colors, later
colors <- c('#66c2a5','#fc8d62','#8da0cb')
p <- ggplot() + 
  geom_point(data = iris_obs, 
             aes(x = V1, y = V2, 
                 shape = est_z, 
                 color = est_z), 
             size = 0.75) + 
  scale_color_manual(values = colors) 

##################
# plot centroids / covariances
##################
# these are are also already in pc space
est_centroids <- iris_fit_file['pc_centroids']
est_covariances <- iris_fit_file['pc_cov']

unique_clusters = unique(iris_fit_file['cluster_memberships'])
n_unique_clusters = length(unique_clusters)
for(i in 1:n_unique_clusters){
  
  # python was 0 indexed ...
  k = unique_clusters[i] + 1
  
  centroids_k <- est_centroids[k, ]
  eigs_k <- eigen(est_covariances[k, , ],
                  symmetric = TRUE)
  
  # need aes_string when adding plots in a for-loop?
  p <- p + geom_ellipse(aes_string(x0 = centroids_k[1],
                                   y0 = centroids_k[2], 
                                   a = sqrt(eigs_k$values[1]) * 3,
                                   b = sqrt(eigs_k$values[2]) * 3,
                                   angle = atan(eigs_k$vectors[2, 1] / 
                                                  eigs_k$vectors[1, 1])), 
                        color = colors[i])
}

p + 
  xlab('PC1') + 
  ylab('PC2') + 
  theme(legend.position = 'none') + 
  get_fontsizes()
