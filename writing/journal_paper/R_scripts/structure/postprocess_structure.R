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

# we considered different thresholds
threshold0 <- alpha_sens_file['threshold0']
threshold1 <- alpha_sens_file['threshold1']
threshold2 <- alpha_sens_file['threshold2']

alpha_sens_df_thresh0 <- 
  data.frame(
    alpha = alpha_sens_file['alpha_list'], 
    refit = alpha_sens_file['n_clusters_thresh_refit0'], 
    lin = alpha_sens_file['n_clusters_thresh_lr0'],
    threshold = threshold0)

alpha_sens_df_thresh1 <- 
  data.frame(
    alpha = alpha_sens_file['alpha_list'], 
    refit = alpha_sens_file['n_clusters_thresh_refit1'], 
    lin = alpha_sens_file['n_clusters_thresh_lr1'],
    threshold = threshold1)


alpha_sens_df_thresh2 <- 
  data.frame(
    alpha = alpha_sens_file['alpha_list'], 
    refit = alpha_sens_file['n_clusters_thresh_refit2'], 
    lin = alpha_sens_file['n_clusters_thresh_lr2'],
    threshold = threshold2)

alpha_sens_df <- rbind(alpha_sens_df_thresh0, 
                       alpha_sens_df_thresh1, 
                       alpha_sens_df_thresh2)


# cluster weights
weights_keep <- 6

weights_refit_df <-
  data.frame(alpha_sens_file['cluster_weights_refit'][, 1:weights_keep]) %>% 
  mutate(alpha = alpha_sens_file['alpha_list'], 
         method = 'refit')

weights_lr_df <- 
  data.frame(alpha_sens_file['cluster_weights_lr'][, 1:weights_keep]) %>% 
  mutate(alpha = alpha_sens_file['alpha_list'], 
         method = 'lin')

weights_df <- rbind(weights_refit_df, weights_lr_df) %>% 
  gather(key = cluster, value = weight, -c('alpha', 'method')) %>% 
  mutate(cluster = sub('X', 'Population ', cluster)) 

# we report some of these numbers in the text
weights_alpha_init <- weights_df %>% 
  filter(alpha == 3) %>% 
  filter(method == 'refit')

weights_alpha_large <- weights_df %>% 
  filter(alpha == 7) %>% 
  filter(method == 'refit')

weights_alpha_small <- weights_df %>% 
  filter(alpha == 1) %>% 
  filter(method == 'refit')

weights_alpha_lr <- weights_df %>% 
  filter(alpha == 1) %>% 
  filter(method == 'lin')

#################
# functional sensitivity results
#################
load_infl_prior_data <- function(input_file){
  fsens_results <- np$load(input_file)
  
  infl_df <- data.frame(logit_v = fsens_results['logit_v_grid'], 
                        infl_x_prior = fsens_results['influence_grid_x_prior'])
  
  pert_df <- data.frame(logit_v = fsens_results['logit_v_grid2'], 
                        log_phi = fsens_results['log_phi'], 
                        p0 = exp(fsens_results['log_p0']), 
                        pc = exp(fsens_results['log_pc']))
  return(list(infl_df = infl_df, 
              pert_df = pert_df))
}
load_fsens_data <- function(input_file){
  
  infl_prior_data <- load_infl_prior_data(input_file)
  
  fsens_results <- np$load(input_file)
  
  sensitivity_df <- data.frame(epsilon = fsens_results['epsilon_vec'], 
                               refit = fsens_results['refit_vec'], 
                               lr = fsens_results['lr_vec'])
  
  return(list(infl_df = infl_prior_data$infl_df, 
              pert_df = infl_prior_data$pert_df, 
              sensitivity_df = sensitivity_df))
}

mbololo_fsens_results <- load_fsens_data(paste0(data_dir, 
                                                'stru_fsens_mbololo.npz'))
ngangao_fsens_results <- load_fsens_data(paste0(data_dir, 
                                                'stru_fsens_ngangao.npz'))
chawia_fsens_results <- load_fsens_data(paste0(data_dir, 
                                               'stru_fsens_chawia.npz'))


##############
# mbololo example admixture
##############
mbololo_admix_file <- np$load(paste0(data_dir, 
                                     'mbololo_fsens_admix_example.npz'))
admix_refit <- mbololo_admix_file['admix_refit']
admix_lr <- mbololo_admix_file['admix_lr']

# the scalar posterior quantites on which our linear approximation failed
logit_stick_df <- 
  read_csv(paste0(data_dir, 'mbololo_logit_stick_bad_example.csv'))

admix_df <- 
  read_csv(paste0(data_dir, 'mbololo_admix_bad_example.csv'))

# we report some of these quantities
admix_df$y[(admix_df$method == 'refit') & 
           (admix_df$population == 1) & 
           (admix_df$epsilon == 1)]
admix_df_refit <- 
  admix_df %>% 
  filter(method == 'refit')

# the scalar posterior quantites on which the fully-linearized quantity did ok
logit_stick_flin_df <- 
  read_csv(paste0(data_dir, 'mbololo_logit_stick_fully_lin.csv'))

admix_flin_df <- 
  read_csv(paste0(data_dir, 'mbololo_admix_fully_lin.csv'))


#################
# timing results
#################

# for alpha sensitivity
alpha_timing_results <- 
  np$load('./R_scripts/data_raw/structure/structure_alphasens_timing.npz')
init_fit_time <- alpha_timing_results['init_optim_time']
alpha_hess_time <- alpha_timing_results['hess_solve_time']
total_alpha_refit_time <- sum(alpha_timing_results['refit_time_vec'])
total_alpha_lr_time <- sum(alpha_timing_results['lr_time_vec'])
median_alpha_refit_time <- median(alpha_timing_results['refit_time_vec'])
median_alpha_lr_time <- median(alpha_timing_results['lr_time_vec'])

# for functional sensitivity
# just report for the mbololo region
mbololo_data_file <- np$load(paste0(data_dir, 
                                    'stru_fsens_mbololo.npz'))
fsens_hess_time <- mbololo_data_file['hess_solve_time']

n <- length(mbololo_data_file['refit_time_vec'])
fsens_refit_time <- mbololo_data_file['refit_time_vec'][n]
fsens_lr_time <- mbololo_data_file['lr_time_vec'][n]

infl_time <- mbololo_data_file['grad_g_time'] + mbololo_data_file['infl_time']

##############
# more mbololo examples admixture: mainly for slides
##############

# the gaussian pertrubation
mbololo_admix_gausspert_file <- np$load(paste0(data_dir, 
                                     'mbololo_fpert_gaussian_null.npz'))

mbololo_admix_gausspert_priors <- 
  load_infl_prior_data(paste0(data_dir,
                              'mbololo_fpert_gaussian_null.npz'))

mbololo_admix_gausspert_admix <- 
  mbololo_admix_gausspert_file['admix_refit']

# the sigmoidal perturbation
mbololo_admix_sigmoidal_file <- np$load(paste0(data_dir, 
                                              'mbololo_fpert_sigmoidal_large.npz'))

mbololo_admix_sigmoidal_priors <- 
  load_infl_prior_data(paste0(data_dir,
                              'mbololo_fpert_sigmoidal_large.npz'))

mbololo_admix_sigmoidal_admix <- 
  mbololo_admix_sigmoidal_file['admix_refit']

structure_timing_dict <- 
  list(init_fit_time = init_fit_time, 
       alpha_hess_time = alpha_hess_time, 
       alpha_refit_time = median_alpha_refit_time,
       alpha_lr_time = median_alpha_lr_time, 
       phi_hessian_time = fsens_hess_time, 
       phi_refit_time = fsens_refit_time, 
       phi_lr_time = fsens_lr_time, 
       infl_time = infl_time)

save(structure_timing_dict, 
     file="./R_scripts/data_processed/structure_timing.RData")

save.image('./R_scripts/data_processed/structure.RData') 

