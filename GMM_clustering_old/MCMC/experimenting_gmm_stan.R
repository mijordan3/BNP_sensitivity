library(rstan)
library(tidyverse)
library(MASS)

set.seed(54)

##########################
# DRAW / LOAD DATA
##########################
draw_gmm_data <- function(n_obs, n_clusters, d, sigma){
  # draw centroids
  mu <- matrix(rnorm(n_clusters * d, mean = 0, sd = 1), nrow = n_clusters)
  
  # draw cluster belongings with
  # uniform weights
  weights <- rep(1, n_clusters) * 1 / n_clusters
  z_ind <- sample.int(n_clusters, prob = weights, size = n_obs, replace = TRUE)
  
  # draw data
  y <- matrix(rep(0, n_obs* d), nrow = n_obs)
  
  for(k in 1:n_clusters){
    num_in_cluster <- sum(z_ind == k)
    samples <- mvrnorm(num_in_cluster, mu = mu[k, ], Sigma = diag(d) * sigma)
    y[z_ind == k, ] <- samples
  }
  return(list(y = y, 
              mu = mu, 
              z_ind = z_ind))
}


load_iris_data <- function(){
  iris_features <- as.matrix(iris[, 1:4]) 
  iris_species <- as.matrix(iris[, 5])
  
  # run PCA
  ir.pca <- prcomp(iris_features,
                   center = TRUE,
                   scale. = TRUE) 
  
  # transform features
  iris_pc_features <- predict(ir.pca, newdata=iris_features)
  
  return(list(iris_features = iris_features, 
              iris_species = iris_species, 
              iris_pc_features = iris_pc_features))
}

simulate_data <- FALSE
if(simulate_data){
  n_obs <- 1000
  n_clusters <- 3
  d <- 2
  sigma <- 0.02
  gmm_data <- draw_gmm_data(n_obs, n_clusters, d, sigma)
  
  data.frame(y1 = gmm_data$y[, 1], 
             y2 = gmm_data$y[, 2], 
             labels = gmm_data$z_ind) %>% 
    ggplot() + geom_point(aes(y1, y2, color = as.factor(labels)))
  
  y <- gmm_data$y
}else{
  iris_data <- load_iris_data()
  y <- iris_data$iris_pc_features
  d <- dim(y)[2]
  
  data.frame(y1 = y[, 1], 
             y2 = y[, 2], 
             labels = iris_data$iris_species) %>% 
    ggplot() + geom_point(aes(y1, y2, color = as.factor(labels)))
  
}

##########################
# DEFINE MODEL
##########################

# Define stan object
load_model <- TRUE
if(load_model){
  model_file <- './gmm_model_compiled.RData'
  load(model_file)
}else{
  gmm_mixture_model <- stan_model(file = './gmm_model.stan')
  save('gmm_mixture_model', file = './gmm_model_compiled.RData')
}

# Get data for stan model
get_default_data_model <- function(y, n_clusters){
  n_obs <- dim(y)[1]
  d <- dim(y)[2]
  
  wishart_df <- 8
  inv_wishart_scale <- diag(d) * 0.62
  
  
  data <- list(n_obs = n_obs, 
              dim = d, 
              n_clusters = n_clusters, 
              y = y, 
              alpha = 4.0, # DP parameter
              mu_prior_var = 10.0, # Prior variance on centroids
              wishart_df = wishart_df, # inv. wishart prior df
              inv_wishart_scale = inv_wishart_scale # inv. wishart prior scale 
  )
  return(data)
}

n_clusters <- 3
model_data <- get_default_data_model(y, n_clusters)

##########################
# SAMPLE
##########################
fit <- sampling(gmm_mixture_model, 
                data = model_data, 
                chains = 1, 
                warmup = 100,
                iter = 200)

##########################
# CHECK RESULTS
##########################
print(fit)

save_fit <- TRUE
if(save_fit){
  save('fit', file = './gmm_model_fit_iris.RData')
}

samples <- rstan::extract(fit)

divergent <- get_sampler_params(fit, inc_warmup=FALSE)[[1]][,'divergent__']
cat('propn divergent: ', mean(divergent))

mu_samples <- matrix(samples$mu, ncol = d)
sigma_samples <- array(samples$sigma, dim = c(dim(mu_samples)[1], d, d))

e_z <- apply(samples$e_z, c(2, 3), mean)
z_ind <- apply(e_z, 1, which.max)

p <- ggplot() + geom_point(aes(x = mu_samples[, 1], y = mu_samples[, 2]), 
                          color = 'black', alpha = 0.5) +
      geom_point(aes(y[, 1], y[, 2], color = as.factor(z_ind)), alpha = 0.5) 
  
p

get_ellipse <- function(Sigma, center, s=3, npoints = 100){
  t <- seq(0, 2*pi, len=npoints)

  a <- s * sqrt(eigen(Sigma)$values[2])
  b <- s * sqrt(eigen(Sigma)$values[1])
  x_ <- a*cos(t)
  y_ <- b*sin(t)
  X <- cbind(x_, y_)
  R <- eigen(Sigma)$vectors
  ellipse <- (X%*%R)
  return(data.frame(x = ellipse[, 1] + center[1], 
                    y = ellipse[, 2] + center[2]))
}

for(i in 1:dim(mu_samples)[1]){
  Sigma <- sigma_samples[i, c(1, 2), c(1, 2)]
  center <- mu_samples[i, c(1,2)]
  ellipse <- get_ellipse(Sigma, center)
  
  ggplot() + geom_path(data=ellipse, aes(x=x, y=y), colour='blue', 
            linetype = 'dashed', alpha = 1)
  
  p <- p + geom_path(data=ellipse, aes(x=x, y=y), colour='blue', 
                     linetype = 'dashed', alpha = 0.1)
}
p

