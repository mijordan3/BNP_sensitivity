functions {
  vector get_weights_from_sticks (vector sticks){
    int n_clust = num_elements(sticks) + 1; // number of clusters
    vector[n_clust] weights; // the weights we shall return
    
    vector[n_clust - 1] sticks_1m = 1 - sticks; // 1 minus the sticks
    
    // get sticks remaining
    vector[n_clust] sticks_remaining; 
    sticks_remaining[1] = 1; 
    for(k in 2:n_clust){
      sticks_remaining[k] = sticks_remaining[k-1] * sticks_1m[k-1]; 
    }
    
    // get weights
    weights[1:(n_clust -1)] = sticks_remaining[1:(n_clust -1)] .* sticks; 
    weights[n_clust] = sticks_remaining[n_clust]; 
    
    
    return weights; 
  }
  
}
data {
  // size parameters
  int n_obs; // number of observations
  int dim; // number of dimensions in data
  int n_clusters; // number of components
  
  // the data    
  matrix[n_obs, dim] y; // observed data
  
  // prior parameters
  real <lower=0> alpha; // DP parameter 
  real mu_prior_var; // prior variance on the centroids
  real <lower=0> wishart_df;  // wishart degrees of freedom parameter
  cov_matrix[dim] inv_wishart_scale;  // inv wishart scale parameter
}

parameters {
  matrix[n_clusters, dim] mu; // the centroids
  vector <lower=0,upper=1>[n_clusters - 1] sticks; // BNP sticks
  cov_matrix[dim] sigma[n_clusters]; 
}
transformed parameters {
  vector[n_clusters] weights; 
  vector[n_clusters] contributions[n_obs];

  weights = get_weights_from_sticks(sticks); 
  for(i in 1:n_obs) {
    for(k in 1:n_clusters) {
      contributions[i][k] = log(weights[k]) + 
                multi_normal_lpdf(y[i] | mu[k], sigma[k]);
    }
  }
}
model {
  // priors
  for(k in 1:n_clusters) {
    // draw centroids
    mu[k] ~ multi_normal(rep_vector(0.0, dim), 
              diag_matrix(rep_vector(mu_prior_var, dim)));
    // draw covariances 
    sigma[k] ~ inv_wishart(wishart_df, inv_wishart_scale); 
    
    if (k < n_clusters){
      sticks[k] ~ beta(1, alpha); 
    }
  }
  
  // likelihood
  for(i in 1:n_obs) {
    target += log_sum_exp(contributions[i]);
  }
}

generated quantities {
  // We get the cluster belongings
  vector[n_clusters] e_z[n_obs];

  for(i in 1:n_obs) {
    e_z[i] = exp(contributions[i] - log_sum_exp(contributions[i]));
  }
}

