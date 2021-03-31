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

# TODO: 


save.image('./R_scripts/data_processed/structure.RData') 
