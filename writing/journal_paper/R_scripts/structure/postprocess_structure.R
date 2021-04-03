library(tidyverse)
library(reticulate)
np <- import("numpy")

data_dir <- './R_scripts/data_raw/structure/'

#################
# the initial fit
#################
stru_init_file <- np$load(paste0(data_dir, 'init_fit.npz'))

e_ind_admix_init <- stru_init_file['e_ind_admix']
geographic_labels <- stru_init_file['labels']

#################
# alpha sensitivity results
#################
alpha_sens_file <- np$load(paste0(data_dir, 'alpha_sens.npz'))
alpha0 <- alpha_sens_file['alpha0']

alpha_sens_df <- 
  data.frame(
    alpha = alpha_sens_file['alpha_list'], 
    n_clusters_refit = alpha_sens_file['n_clusters_refit'], 
    n_clusters_lr = alpha_sens_file['n_clusters_lr'], 
    n_clusters_thresh_refit = alpha_sens_file['n_clusters_thresh_refit'],
    n_clusters_thresh_lr = alpha_sens_file['n_clusters_thresh_lr']) %>%
  filter(alpha <= 10)

threshold <- alpha_sens_file['threshold']

# cluter weights
weights_keep <- 9

weights_refit_df <-
  data.frame(alpha_sens_file['cluster_weights_refit'][, 1:weights_keep]) %>% 
  mutate(alpha = alpha_sens_file['alpha_list'], 
         method = 'refit')

weights_lr_df <- 
  data.frame(alpha_sens_file['cluster_weights_lr'][, 1:weights_keep]) %>% 
  mutate(alpha = alpha_sens_file['alpha_list'], 
         method = 'lr')

weights_df <- rbind(weights_refit_df, weights_lr_df) %>% 
  gather(key = cluster, value = weight, -c('alpha', 'method')) %>% 
  mutate(cluster = sub('X', 'cluster ', cluster)) %>% 
  filter(alpha <= 10)

#################
# functional sensitivity results
#################
load_fsens_data <- function(input_file){
  fsens_results <- np$load(input_file)
  
  infl_df <- data.frame(logit_v = fsens_results['logit_v_grid'], 
                        infl_x_prior = fsens_results['influence_grid_x_prior'])
  
  pert_df <- data.frame(logit_v = fsens_results['logit_v_grid2'], 
                        log_phi = fsens_results['log_phi'], 
                        p0 = exp(fsens_results['log_p0']), 
                        pc = exp(fsens_results['log_pc']))
  
  sensitivity_df <- data.frame(epsilon = fsens_results['epsilon_vec'], 
                               refit = fsens_results['refit_vec'], 
                               lr = fsens_results['lr_vec'])
  
  return(list(infl_df = infl_df, 
              pert_df = pert_df, 
              sensitivity_df = sensitivity_df))
}

mbololo_fsens_results <- load_fsens_data(paste0(data_dir, 
                                                'stru_fsens_mbololo.npz'))
ngangao_fsens_results <- load_fsens_data(paste0(data_dir, 
                                                'stru_fsens_ngangao.npz'))
chawia_fsens_results <- load_fsens_data(paste0(data_dir, 
                                               'stru_fsens_chawia.npz'))


chawia_data_file <- np$load('./R_scripts/data_raw/structure/stru_fsens_chawia.npz')
admix1_refit <- chawia_data_file['admix1_refit']
admix2_refit <- chawia_data_file['admix2_refit']
admix1_lr <- chawia_data_file['admix1_lr']
admix2_lr <- chawia_data_file['admix2_lr']

save.image('./R_scripts/data_processed/structure.RData') 
