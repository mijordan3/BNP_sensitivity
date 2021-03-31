data_dir = './R_scripts/data/mice/'

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
# The coclustering at alpha = 1
################
alpha1_coclust_file <- np$load(paste0(data_dir, 'coclustering_alpha1.0.npz'))

# the fit at the initial alpha: 
# TODO: put this in its own file, this is confusing
coclust_init <- load_coclust_file(alpha1_coclust_file, 'coclust_init')

# the refit at alpha = 1
coclust_refit_alpha1 <- 
  load_coclust_file(alpha1_coclust_file, 'coclust_refit') 

# lr at alpha = 1
coclust_lr_alpha1 <-
  load_coclust_file(alpha1_coclust_file, 'coclust_lr') 

################
# The coclustering at alpha = 11
################
alpha11_coclust_file <- np$load(paste0(data_dir, 
                                       'coclustering_alpha11.0.npz'))

coclust_refit_alpha11 <- 
  load_coclust_file(alpha11_coclust_file, 'coclust_refit') 


coclust_lr_alpha11 <-
  load_coclust_file(alpha11_coclust_file, 'coclust_lr') 


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
  np$load(paste0(data_dir, 'functional_coclustering_gauss_pert1.npz'))

prior_pert_df <- 
  data.frame(logit_v = fpert_coclust_file['logit_v_grid'],
             log_phi = fpert_coclust_file['log_phi'], 
             p0_logit = fpert_coclust_file['p0_logit'], 
             pc_logit = fpert_coclust_file['pc_logit'])

# the coclustering matrices
coclust_refit_fpert <- 
  load_coclust_file(fpert_coclust_file, 'coclust_refit') 

coclust_lr_fpert <-
  load_coclust_file(fpert_coclust_file, 'coclust_lr') 
