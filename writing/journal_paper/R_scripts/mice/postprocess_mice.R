library(tidyverse)
library(reticulate)
np <- import("numpy")

data_dir = './R_scripts/data_raw/mice/'

######################
# load example gene observations
######################
example_genes_file <- np$load(paste0(data_dir, 'example_genes.npz'))
timepoints <- example_genes_file[['timepoints']]
example_data <- data.frame(time = timepoints,
                           y = example_genes_file[['obs']], 
                           fitted = example_genes_file[['fitted']])

######################
# load regressor matrix
######################
regr_df <- data.frame(example_genes_file[['regressors']])
regr_df$time <- example_genes_file[['timepoints']]


########################
# load data to plot centroids
########################
centroids_file <- np$load(paste0(data_dir, 'fitted_centroids.npz'))

# Load data points
# these have already been aggregated and shifted
expression_data <- 
  data.frame(t(centroids_file['y_shifted'])) %>% 
  # unique because the data is already aggregated
  mutate(time = unique(timepoints)) %>% 
  gather(key = gene_id, value = expression, -time) %>% 
  # remove "X" in front of gene ID
  mutate(gene_id = sub('.', '', gene_id)) %>% 
  mutate(gene_id = as.numeric(gene_id))

# now load inferred cluster memberships
cluster_memberships <- 
  data.frame(cluster_id = centroids_file[['est_z']]) %>% 
  mutate(gene_id = 1:n()) 

# count number of observations per cluster
n_keep <- 12
cluster_weights <- 
  cluster_memberships %>% 
  group_by(cluster_id) %>% 
  summarize(counts = n()) %>% 
  # we are only going to keep the top clusters
  arrange(desc(counts)) %>% 
  filter(1:n() <= n_keep) 

# get centroids
centroids_df <-
  as.data.frame(centroids_file[['centroids']]) %>% 
  # create time points
  mutate(time = timepoints) %>% 
  # gather by clusters
  gather(key = cluster_id, value = centroid_value, -time) %>% 
  # remove the letter in front of cluster id
  mutate(cluster_id = sub('.', '', cluster_id)) %>% 
  # minus one because the saved inferred memberships are
  # start from zero ... 
  mutate(cluster_id = as.numeric(cluster_id) - 1) %>% 
  # again, we don't need to keep replicates for the centroids
  distinct()

###########################
# function to load coclustering matrix
###########################
load_coclust_file <- function(file, key){
  # expects a coclustering matrix
  # converts the matrix to a long data frame
  
  coclust_wide <- as.data.frame(file[[key]])
  
  coclust <- coclust_wide %>% 
    mutate(gene1 = colnames(coclust_wide)) %>% 
    # make it a long data frame
    gather(key = gene2, value = coclustering, -gene1) %>%
    # clean up the gene names
    mutate(gene1 = sub('.', '', gene1), 
           gene2 = sub('.', '', gene2)) %>% 
    mutate(gene1 = as.numeric(gene1), 
           gene2 = as.numeric(gene2))
  return(coclust)
}


################
# The coclustering at first alpha perturation
################
alpha_pert1 <- 0.1
alpha1_coclust_file <- np$load(paste0(data_dir, 
                                      'coclustering_alpha', 
                                      alpha_pert1, 
                                      '.npz'))

# the fit at the initial alpha: 
# TODO: put this in its own file, this is confusing
# TODO: save the alpha0
alpha0 <- 6
coclust_init <- load_coclust_file(alpha1_coclust_file, 'coclust_init')

# the refit at alpha = 1
coclust_refit_alpha1 <- 
  load_coclust_file(alpha1_coclust_file, 'coclust_refit') 

# lr at alpha = 1
coclust_lr_alpha1 <-
  load_coclust_file(alpha1_coclust_file, 'coclust_lr') 

# some summary statistics to report in the main tex
get_max_diff <- function(coclust_refit, 
                         coclust_init){
  diff_df <- 
    inner_join(coclust_refit, 
               coclust_init, 
               by = c('gene1', 'gene2')) %>%
    mutate(diff = coclustering.x - coclustering.y)
  
  maxdiff <- max(abs(diff_df$diff))
  
  return(maxdiff)
}

maxdiff_alpha1 <- get_max_diff(coclust_refit_alpha1, coclust_init)

################
# The coclustering at alpha = 12
################
alpha_pert2 = 12.0
alpha11_coclust_file <- np$load(paste0(data_dir, 
                                       'coclustering_alpha', 
                                       sprintf('%.01f', alpha_pert2), 
                                       '.npz'))

coclust_refit_alpha11 <- 
  load_coclust_file(alpha11_coclust_file, 'coclust_refit') 


coclust_lr_alpha11 <-
  load_coclust_file(alpha11_coclust_file, 'coclust_lr') 

# max change in alpha = 11
maxdiff_alpha11 <- get_max_diff(coclust_refit_alpha11, coclust_init)

###################
# data for the influence function
###################
infl_data <- 
  np$load(paste0(data_dir, 'coclustering_worstcase.npz'))

influence_df <- 
  data.frame(logit_v = infl_data['logit_v_grid'], 
             influence = infl_data['influence_grid'], 
             p0_logit = infl_data['p0_logit'], 
             influence_x_prior = infl_data['influence_grid_x_prior'])

#################
# data for the functional perturbation results
#################
# the priors, initial and perturbed
fpert_coclust_file <-
  np$load(paste0(data_dir, 'functional_coclustering.npz'))

prior_pert_df <- 
  data.frame(logit_v = fpert_coclust_file['logit_v_grid'],
             log_phi = fpert_coclust_file['log_phi'], 
             p0_logit = fpert_coclust_file['p0_logit'], 
             pc_logit = fpert_coclust_file['pc_logit'], 
             p0_constr = fpert_coclust_file['p0_constrained'], 
             pc_constr = fpert_coclust_file['pc_constrained'])

# the coclustering matrices
coclust_refit_fpert <- 
  load_coclust_file(fpert_coclust_file, 'coclust_refit') 

coclust_lr_fpert <-
  load_coclust_file(fpert_coclust_file, 'coclust_lr') 

####################
# timing results
####################

# paramametric sensitivity timing results
alpha_timing_results <- 
  np$load(paste0(data_dir, 'mice_alphasens_timing.npz'))
init_fit_time <- alpha_timing_results['init_optim_time']
alpha_hess_time <- alpha_timing_results['hessian_solve_time']
refit_time_vec <- alpha_timing_results['refit_time_vec']
lr_time_vec <- alpha_timing_results['lr_time_vec']

# functional sensitivity timing results
fsens_timing_results <- np$load(paste0(data_dir, 
      'functional_coclustering_timing.npz'))

phi_hessian_time <- fsens_timing_results['hess_solve_time']

n <- length(fsens_timing_results['refit_time_vec'])
phi_refit_time <- fsens_timing_results['refit_time_vec'][n]
phi_lr_time <- fsens_timing_results['lr_time_vec'][n]

infl_time <- fsens_timing_results['grad_time'] +
              fsens_timing_results['infl_time']

# save everything into one dictionary
mice_timing_dict <- 
  list(init_fit_time = init_fit_time, 
       alpha_hess_time = alpha_hess_time, 
       alpha_refit_time = median(refit_time_vec), 
       alpha_lr_time = median(lr_time_vec), 
       phi_hessian_time = phi_hessian_time, 
       phi_refit_time = phi_refit_time, 
       phi_lr_time = phi_lr_time, 
       infl_time = infl_time)

save(mice_timing_dict, 
     file="./R_scripts/data_processed/mice_timing.RData")

save.image('./R_scripts/data_processed/mice.RData') 
