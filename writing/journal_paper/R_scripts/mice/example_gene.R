# load example gene observations
example_genes_file <- np$load('./R_scripts/mice/data/example_genes.npz')
timepoints <- example_genes_file[['timepoints']]
example_data <- data.frame(time = timepoints,
                           y = example_genes_file[['obs']], 
                           fitted = example_genes_file[['fitted']])

# plot example gene
p1 <- example_data %>% 
  mutate(y_demean = y - mean(y)) %>% 
  ggplot() +
  geom_line(aes(x = time, y = fitted), 
            color = 'grey') + 
  geom_point(aes(x = time, y = y_demean), 
             color = 'blue', 
             shape = 'x', 
             size = 2) + 
  ylab('gene expr. (de-meaned)') + 
  xlab('time (hours)') + 
  theme_bw() + 
  get_fontsizes()

# load regressor matrix
regr_df <- data.frame(example_genes_file[['regressors']])
regr_df$time <- example_genes_file[['timepoints']]
p2 <- 
  regr_df %>%
  distinct() %>% 
  gather(key = 'basis_id', 
         value = 'regressor', 
         -time) %>% 
  ggplot() + 
  geom_point(aes(x = time, y = regressor, color = basis_id), size = 0.5) + 
  geom_line(aes(x = time, y = regressor, color = basis_id)) + 
  ylab('B-spline basis value') + 
  xlab('time (hours)') + 
  theme_bw() + 
  theme(legend.position = 'none') + 
  get_fontsizes()

grid.arrange(p1, p2, nrow = 1)
