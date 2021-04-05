library(tidyverse)
library(reticulate)
np <- import("numpy")

data_dir <- './R_scripts/data_raw/iris/'

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

alpha_sens_df_pred <- 
  data.frame(
    alpha = alpha_sens_file['alpha_list'],
    n_clusters_refit = alpha_sens_file['n_clusters_pred_refit'],
    n_clusters_lr = alpha_sens_file['n_clusters_pred_lr'], 
    n_clusters_thresh_refit = alpha_sens_file['n_clusters_pred_thresh_refit'],
    n_clusters_thresh_lr = alpha_sens_file['n_clusters_pred_thresh_lr'])

####################
# Load data functional sensitivity
###################
influence_data <- np$load(paste0(data_dir, 'iris_influence_fun.npz'))

influence_df <- 
  data.frame(logit_v = influence_data['logit_v_grid'], 
             influence = influence_data['influence_grid'], 
             influence_x_prior = influence_data['influence_grid_x_prior'])

load_fsens_results <- function(fsens_file){
  fsens_data <- np$load(fsens_file)
  
  priors_df <- 
    data.frame(logit_v = fsens_data['logit_v_grid'], 
               v = fsens_data['v_grid'], 
               log_phi = fsens_data['log_phi_grid'], 
               p0 = fsens_data['p0_constr'], 
               p1 = fsens_data['p1_constr'])
  
  sensitivity_df <- 
    data.frame(epsilon = fsens_data['epsilon_vec'], 
               refit = fsens_data['refit_g_vec'],
               lr = fsens_data['lr_g_vec'])
  
  return(list(priors_df = priors_df, 
              sensitiivty_df = sensitivity_df))
}

wc_results <- load_fsens_results(paste0(data_dir, "iris_worst_case.npz"))
fpert1_results <- load_fsens_results(paste0(data_dir, "iris_fpert1.npz"))
fpert2_results <- load_fsens_results(paste0(data_dir, "iris_fpert2.npz"))
fpert3_results <- load_fsens_results(paste0(data_dir, "iris_fpert3.npz"))

save.image('./R_scripts/data_processed/iris.RData') 

