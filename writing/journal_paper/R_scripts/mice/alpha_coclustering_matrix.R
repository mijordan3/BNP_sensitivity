###################
# Load co-clustering files
###################
alpha1_coclust_file <- np$load('./R_scripts/mice/data/coclustering_alpha1.0.npz')

alpha11_coclust_file <- np$load('./R_scripts/mice/data/coclustering_alpha11.0.npz')

###################
# results at alpha = 1
###################
coclust_refit1 <- 
  load_coclust_file(alpha1_coclust_file, 'coclust_refit') %>% 
  # compute differences
  get_coclust_diff(coclust_init) %>% 
  # label appropriately 
  mutate(method = 'refit', alpha = 1)


coclust_lr1 <-
  load_coclust_file(alpha1_coclust_file, 'coclust_lr') %>% 
  # compute differences
  get_coclust_diff(coclust_init) %>% 
  # label appropriately 
  mutate(method = 'lr', alpha = 1)

###################
# results at alpha = 11
###################
coclust_refit11 <- 
  load_coclust_file(alpha11_coclust_file, 'coclust_refit') %>% 
  # compute differences
  get_coclust_diff(coclust_init) %>% 
  # label appropriately 
  mutate(method = 'refit', alpha = 11)

coclust_lr11 <- 
  load_coclust_file(alpha11_coclust_file, 'coclust_lr') %>% 
  # compute differences
  get_coclust_diff(coclust_init) %>% 
  # label appropriately 
  mutate(method = 'lr', alpha = 11)


# combine: 
coclust_diff <- rbind(coclust_refit1, 
                      coclust_lr1, 
                      coclust_refit11, 
                      coclust_lr11)

# plot
limits <- c(1e-4, 1e-3, 1e-2, Inf)
labels <- c('< -1e-2', '(-1e-2, -1e-3]', '(-1e-3, -1e-4]',
            '(-1e-4, 1e-4]', '(1e-4, 1e-3]', '(1e-3, 1e-2]', '> 1e-2')

p <- coclust_diff %>% 
  # to make the labels look good
  mutate(alpha = paste0('alpha = ', alpha)) %>% 
  plot_coclust_diff(limits = limits,
                    limit_labels = labels) +
  facet_grid(cols = vars(alpha), 
             rows = vars(method)) + 
  theme(axis.title = element_blank(), 
        axis.text = element_blank())

p
