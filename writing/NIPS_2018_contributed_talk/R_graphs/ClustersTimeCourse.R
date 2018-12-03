# load data
results_file <- "clusters.RData"
data_env <- LoadIntoEnvironment(file.path(data_path, results_file))
clusters_df <- data_env$clusters_df

n_obs <- dim(clusters_df)[1] # number of observations
n_time <- dim(clusters_df)[2] - 1 # number of timepoints
colnames(clusters_df)[2:(n_time+1)] <- 1:n_time 

# make long data
clusters_df$obs_indx <- as.factor(1:n_obs)
clusters_df_long <- gather(clusters_df, time, expression, 2:15)
clusters_df_long$time <- as.numeric(clusters_df_long$time)

# sort clusters by size
clusters <- unique(clusters_df$z_ind)
cluster_size <- c()
for(k in clusters){
  cluster_size <- c(cluster_size, sum(clusters_df$z_ind == k))
}
clusters <- clusters[order(cluster_size, decreasing = TRUE)]

cluster_plots <- list()
i <- 1
for(k in clusters){
  cluster_size <- sum(clusters_df$z_ind == k)
  if(cluster_size > 1){
    cluster_plots[[i]] <- clusters_df_long %>% filter(z_ind == k) %>% 
      ggplot() + geom_line(aes(x = time, y = expression, color = obs_indx)) + 
      theme(legend.position="none", 
            axis.title.x=element_blank(),
            axis.text.x=element_blank(),
            axis.ticks.x=element_blank(), 
            axis.title.y=element_blank(),
            axis.text.y=element_blank(),
            axis.ticks.y=element_blank()) + 
      ylim(-4, 5) + coord_fixed(ratio = 1.3)
      theme(plot.margin=grid::unit(c(0,0,0,0), "mm"))
    i <- i + 1
  }
}

grid.arrange(grobs = cluster_plots, ncol = 7, nrow = 3)

