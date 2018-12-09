
library(Matrix)
library(reshape2)
library(dplyr)
library(ggplot2)
library(StructureLRVB)



# library(gtools)
# alpha <- c(3, 4, 5)
# n_sim <- 1e4
# q <- rdirichlet(n_sim, alpha)
# q_prob <- ddirichlet(q, alpha)
# mean(-log(q_prob))
# GetDirichletEntropy(alpha)
# 
# apply(log(q), 2, mean)
# GetELogDirichlet(alpha)




repo_loc <- file.path(Sys.getenv("GIT_REPO_LOC"),
                      "variational_bayes/structure")
setwd(file.path(repo_loc, "r_package/StructureLRVB/inst/R/"))
source(file.path(repo_loc, "get_sensitivity_lib.R"))
source(file.path(repo_loc, "r_package/StructureLRVB/inst/R/get_sensitivity_lib.R"))


# data <- ReadStructureData(input.basename = file.path(repo_loc, "test_input_data/testdata_plain"),
#                           output.basename = file.path(repo_loc, "test_input_data/testdata_plain_out.3"))
data <- ReadStructureData(input.basename = file.path(repo_loc, "test_input_data/testdata_small"),
                          output.basename = file.path(repo_loc, "test_input_data/testdata_small_out.3"))

start_nat_params <- SetNaturalParams(data)
prior_params <- SetPriors(start_nat_params$n_ind, start_nat_params$n_gene, start_nat_params$n_pop)
genome_matrix_int <- GetGenomeIntegerMatrix(data)

nat_params <- start_nat_params
moment_params <- GetMomentParameters(nat_params)




##############################
# Check the ELBO is maximized

elbo <- GetELBO(genome_matrix_int, nat_params, prior_params)

log_lik <- GetLogLikelihood(genome_matrix_int, moment_params, prior_params)
entropy <- GetEntropy(genome_matrix_int, nat_params)
elbo <- GetELBO(genome_matrix_int, nat_params, prior_params)

ELBOCheckQ <- function(orig_nat_params, n, k, epsilon) {
  nat_params_pert <- orig_nat_params
  nat_params_pert$q_vec[[n]][k] <- orig_nat_params$q_vec[[n]][k] + epsilon
  elbo_pert <- GetELBO(genome_matrix_int, nat_params_pert, prior_params)
  return(elbo_pert)
}

elbo_diff <- sapply(1:nat_params$n_ind, function(n) ELBOCheckQ(nat_params, n, 1, 1)) - elbo
max(elbo_diff)
  

n <- 2

elbo <- GetELBO(genome_matrix_int, nat_params, prior_params)

nat_params_pert <- nat_params
nat_params_pert$q_vec[[n]][1] <- 100
moment_params_pert <- GetMomentParameters(nat_params_pert)
moment_params_pert$q_vec[[n]]
moment_params_pert$p_vec[[1]]
moment_params_pert$omp_vec[[1]]
genome_matrix_int[n, 1]

GetDirichletEntropy(nat_params_pert$q_vec[[n]])
GetDirichletEntropy(nat_params$q_vec[[n]])
elbo_pert <- GetELBO(genome_matrix_int, nat_params_pert, prior_params)

elbo
elbo_pert
elbo - elbo_pert > 0


######################
# Get sensitivity to a data point removal

missing_n <- 37

pop_cov <- GetPopulationCovariance(nat_params)
lrvb_cpp_time <- Sys.time()
lrvb_cov_cpp <- GetLRVBCovariance(genome_matrix_int, nat_params, moment_params, verbose=TRUE)
print(lrvb_cpp_time <- Sys.time() - lrvb_cpp_time)
min(diag(lrvb_cov_cpp))

e_log_q_df <- ConvertParameterListToDataFrame(moment_params$q_vec)
e_log_q_df$group <- "original"

# Sensitivity
sens_terms <- GetHazVzGnTerm(genome_matrix_int, moment_params, n=missing_n - 1)
sens_vec <- as.numeric(lrvb_cov_cpp %*% (sens_terms$g_alpha + sens_terms$haz_vz_gn))
sens_par_list <- DecodeParameters(sens_vec, moment_params)


###########################
# Use fastStructure's output:

# data_pert <-
#   ReadStructureData(input.basename = file.path(repo_loc, "test_input_data/testdata_plain_no165"),
#                     output.basename = file.path(repo_loc, "test_input_data/testdata_plain_no165_out.3"))
data_pert <-
  ReadStructureData(input.basename = file.path(repo_loc, "test_input_data/testdata_small_no37"),
                    output.basename = file.path(repo_loc, "test_input_data/testdata_small_no37_out.3"))
moment_params_pert <- SetMomentParams(data_pert)

e_log_q_pert_df <-
  ConvertParameterListToDataFrame(moment_params_pert$q_vec) %>%
  mutate(n=ifelse(n < missing_n, n, n + 1))
e_log_q_pert_df$group <- "perturbed"

log_q_sens <- ConvertParameterListToDataFrame(sens_par_list$q_vec)
log_q_sens$group <- "sensitivity"

# Figure out the mapping from one clustering to the other.
# Note: now that I am using my own starting values in fastStructure, this should not
# be necessary.
permutation_df <-
  rbind(group_by(e_log_q_df, n) %>% top_n(n=1, wt=value),
        group_by(e_log_q_pert_df, n) %>% top_n(n=1, wt=value)) %>%
  dcast(n ~ group, value.var="k") %>%
  ungroup() %>% group_by(original, perturbed) %>%
  summarise(n=length(n)) %>%
  top_n(n=1, wt=n) %>% arrange(original)
perm <- permutation_df$perturbed

e_log_q_comp <-
  rbind(mutate(e_log_q_pert_df, k=perm[k]),
        log_q_sens,
        e_log_q_df) %>%
  dcast(n + k ~ group) %>%
  mutate(diff = perturbed - original) %>%
  group_by(n) %>%
  mutate(max_x=which.max(original)) %>%
  filter(n != missing_n)

if (FALSE) {
  ggplot(filter(e_log_q_comp, original > -10)) +
    geom_point(aes(x=diff, y=-1 * sensitivity, color=original)) +
    xlab("Actual change from deleting a data point") +
    ylab("LRVB predicted change") +
    ggtitle("Expected log dirichlet parameter for population assignment")
}


###########################
# Use fastStructure's output to check the prior sensitivity:

# epsilon is set by the parameters passed to fastStructure.
epsilon <- 0.1

data_pert <-
  ReadStructureData(input.basename = file.path(repo_loc, "test_input_data/testdata_small"),
                    output.basename = file.path(repo_loc, "test_input_data/testdata_small_beta_pert_out.3"))

nat_params_pert <- SetNaturalParams(data_pert)
moment_params_pert <- SetMomentParams(data_pert)

e_log_q_pert_df <- ConvertParameterListToDataFrame(moment_params_pert$q_vec)
e_log_q_pert_df$group <- "perturbed"

# Get sensitivity to the prior "beta" paramter (in fastStructure's notation)
encoder <- GetParameterEncoder(moment_params)
prior_beta_g <- rep(0, nrow(lrvb_cov_cpp))
for (l in 1:moment_params$n_gene) {
  ind <- encoder$p_vec[[l]]
  prior_beta_g[ind:(ind + moment_params$n_pop)] <- 1
}

prior_sens_vec <- as.numeric(lrvb_cov_cpp %*% prior_beta_g)
prior_sens_par_list <- DecodeParameters(prior_sens_vec, moment_params)

log_q_sens <- ConvertParameterListToDataFrame(prior_sens_par_list$q_vec)
log_q_sens$group <- "sensitivity"

# Figure out the mapping from one clustering to the other.
# Note: now that I am using my own starting values in fastStructure, this should not
# be necessary.
permutation_df <-
  rbind(group_by(e_log_q_df, n) %>% top_n(n=1, wt=value),
        group_by(e_log_q_pert_df, n) %>% top_n(n=1, wt=value)) %>%
  dcast(n ~ group, value.var="k") %>%
  ungroup() %>% group_by(original, perturbed) %>%
  summarise(n=length(n)) %>%
  top_n(n=1, wt=n) %>% arrange(original)
perm <- permutation_df$perturbed


e_log_q_comp <-
  rbind(e_log_q_pert_df, log_q_sens, e_log_q_df) %>%
  dcast(n + k ~ group) %>%
  mutate(diff = perturbed - original) %>%
  group_by(n) %>%
  mutate(max_x=which.max(original))


if (FALSE) {
  ggplot(filter(e_log_q_comp, original > -10)) +
    geom_point(aes(x=diff, y=epsilon * sensitivity, color=original)) +
    xlab("Actual change from changing prior") +
    ylab("LRVB predicted change") +
    ggtitle("Effect of prior parameter change") +
    geom_abline(aes(intercept=0, slope=1))
}


##########################
# Look in changes in the natural parameters instead.

encoder <- GetParameterEncoder(moment_params)

rate_results <- GetRateJacobian(nat_params)
rate_results$rate_jac <- Matrix(rate_results$rate_jac)

pop_cov <- GetPopulationCovariance(nat_params)
haz_vz_hza <- GetHazVzHzaTerm(genome_matrix_int, moment_params, TRUE)
qaa_t <- (Diagonal(encoder$dim) - haz_vz_hza %*% pop_cov)

prior_beta_g <- rep(0, encoder$dim)
for (l in 1:moment_params$n_gene) {
  ind <- encoder$p_vec[[l]]
  prior_beta_g[ind:(ind + moment_params$n_pop)] <- 1
}

rate_sens_vec <- as.numeric(rate_results$rate_jac %*% solve(qaa_t, prior_beta_g))
prior_sens_par_list <- DecodeParameters(rate_sens_vec, moment_params)
e_q_sens_df <- ConvertParameterListToDataFrame(prior_sens_par_list$q_vec)
e_q_sens_df$group <- "sensitivity"

e_q_df <- ConvertParameterListToDataFrame(rate_results$rate_params$q_vec)
e_q_df$group <- "original"
rate_results_pert <- GetRateJacobian(nat_params_pert)
e_q_pert_df <- ConvertParameterListToDataFrame(rate_results_pert$rate_params$q_vec)
e_q_pert_df$group <- "perturbed"

e_q_comp <-
  rbind(e_q_pert_df, e_q_sens, e_q_df) %>%
  dcast(n + k ~ group) %>%
  mutate(diff = perturbed - original) %>%
  group_by(n) %>%
  mutate(max_x=which.max(original))

ggplot(e_q_comp) +
  geom_point(aes(x=diff, y=epsilon * sensitivity, color=original), size=3) +
  xlab("Actual change from changing prior") +
  ylab("LRVB predicted change") +
  ggtitle("Effect of prior parameter change") +
  geom_abline(aes(intercept=0, slope=1))


##############
# Check the accelerated method

data <- ReadStructureData(input.basename = file.path(repo_loc, "test_input_data/testdata_small"),
                          output.basename = file.path(repo_loc, "test_input_data/testdata_small_out.3"))
nat_params <- start_nat_params
moment_params <- GetMomentParameters(nat_params)

data_pert <-
  ReadStructureData(input.basename = file.path(repo_loc, "test_input_data/testdata_small"),
                    output.basename = file.path(repo_loc, "test_input_data/testdata_small_restart_out.3"))

nat_params_pert <- SetNaturalParams(data_pert)
moment_params_pert <- SetMomentParams(data_pert)

alpha_q_df <- ConvertParameterListToDataFrame(nat_params$q_vec)
alpha_q_df$group <- "original"

alpha_q_pert_df <- ConvertParameterListToDataFrame(nat_params_pert$q_vec)
alpha_q_pert_df$group <- "perturbed"

alpha_q_comp <-
  rbind(alpha_q_pert_df, alpha_q_df) %>%
  dcast(n + k ~ group) %>%
  mutate(diff = perturbed - original) %>%
  group_by(n) %>%
  mutate(max_x=which.max(original))





