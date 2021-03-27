centroids_file <- np$load('./R_scripts/mice/data/fitted_centroids.npz')

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

expression_data %>%
  # join cluster memberships
  inner_join(cluster_memberships, by = 'gene_id') %>%
  # join centroids
  inner_join(centroids_df, by = c('cluster_id', 'time')) %>% 
  # now filter to those top clusters
  inner_join(cluster_weights, by = 'cluster_id') %>%
  # re-order by cluster weights
  mutate(cluster_id = fct_reorder(as.factor(cluster_id),
                                  counts, .desc = TRUE)) %>% 
  # plot
  ggplot() + 
  geom_line(aes(x = time, y = expression, group = gene_id),
            alpha = 0.5) +
  geom_line(aes(x = time, y = centroid_value), 
            color = 'blue', size = 1.5) +
  facet_wrap(~cluster_id, scales = 'free_y') + 
  theme_bw() + 
  theme(strip.background = element_blank(),
        strip.text.x = element_blank()) + 
  xlab('time (hours)') + 
  ylab('gene expr. (shifted)') + 
  get_fontsizes()

