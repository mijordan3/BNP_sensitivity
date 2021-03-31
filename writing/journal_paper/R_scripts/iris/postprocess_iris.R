data_dir <- './R_scripts/data/iris/'

####################
# Load data for initial fit
###################
iris_fit_file <- np$load(paste0(data_dir, 'iris_fit.npz'))

# the observations in PC space and inferred memberships 
iris_obs <- data.frame(PC1 = iris_fit_file['pc_iris_obs'][, 1], 
                       PC2 = iris_fit_file['pc_iris_obs'][, 2], 
                       est_z = iris_fit_file['cluster_memberships'])

# estimated centroids and covariances
est_centroids <- iris_fit_file['pc_centroids']
est_covariances <- iris_fit_file['pc_cov']


####################
# Load data alpha sensitivity
###################
alpha_sens_file <- np$load(paste0(data_dir, "iris_alpha_sens.npz"))

alpha0 <- alpha_sens_file['alpha0']

alpha_sens_df <- 
  data.frame(
    alpha = alpha_sens_file['alpha_list'],
    n_clusters_refit = alpha_sens_file['n_clusters_refit'],
    n_clusters_lr = alpha_sens_file['n_clusters_lr'], 
    n_clusters_thresh_refit = alpha_sens_file['n_clusters_thresh_refit'],
    n_clusters_thresh_lr = alpha_sens_file['n_clusters_thresh_lr'])
