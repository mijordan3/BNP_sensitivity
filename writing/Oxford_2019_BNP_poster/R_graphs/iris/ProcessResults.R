library(ggplot2)
library(reshape2)
library(dplyr)
library(reticulate)
library(gridExtra)

use_python("/usr/bin/python3")

# TODO: make relative
repo_path <- "/home/rgiordan/Documents/git_repos/genomic_time_series_bnp"
data_path <- file.path(repo_path, "data/vb_fits")
paper_data_path <- file.path(repo_path, "writing/nips_workshop_2017/R_graphs/data")
setwd(data_path)

`%_%` <- function(x, y) paste(x, y, sep="")

InitializePython <- function() {
  py_run_string("import sys")
  py_run_string("import json")
  py_run_string("import json_tricks")
  py_run_string("import numpy as np")
}

InitializePython()

py_main <- reticulate::import_main()

# Read the results from process_results_for_paper.py
use_best_init <- TRUE
if (!use_best_init) {
  # These are results from the "warm" start.  This is only for the appendix.
  output_filename <- "results_for_paper_warm.Rdata"
  json_for_paper_filename <-
   "mice_data_results_for_paper_wed_seed3356363_date10312017_fiverestarts_data10312017.json"
} else {
  # These are results from the "hot" start.  These are the main results in the paper.
  json_for_paper_filename <- "mice_data_results_for_paper_wed_seed3356363_date10312017_best_date10312017.json"
  #json_for_paper_filename <- "mice_data_results_for_paper.json"
  output_filename <- "results_for_paper.Rdata"
}


json_filename <- file.path(data_path, json_for_paper_filename)

reticulate::py_run_string(
"
with open('" %_% json_filename %_% "', 'r') as json_file:
  json_results = json.load(json_file)
")

names(py_main$json_results)
reticulate::py_run_string(
"
def unjsonify_results(json_result_dict):
    results_dict = {}
    for k in json_result_dict.keys():
      if '_np_packed' in str(k):
        k_new = str(k).replace('_np_packed', '')
        results_dict[k_new] = json_tricks.loads(json_result_dict[k])
      else:
        results_dict[k] = json_result_dict[k]
    return results_dict  
")

reticulate::py_run_string("
hot = unjsonify_results(json_results['hot'])
warm = unjsonify_results(json_results['warm'])
cold = unjsonify_results(json_results['cold'])
lr = unjsonify_results(json_results['lr'])
")

names(py_main$lr)

MakeBaseResultDataframe <- function(array, metric, method) {
  return(data.frame(val=array, metric=metric, method=method, draw=1:length(array)))
}

MakeMIResultDataframe <- function(result_dict, method) { 
  return(MakeBaseResultDataframe(result_dict$mi_array, method=method, metric="Mutual information"))
}

MakeFMResultDataframe <- function(result_dict, method) { 
  return(MakeBaseResultDataframe(result_dict$fm_array, method=method, metric="Fowlkes-Mallows"))
}

MakeKLResultDataframe <- function(result_dict, method) { 
  return(MakeBaseResultDataframe(result_dict$kl_array, method=method, metric="KL objective"))
}

MakeCoclusteringResultDataframe <- function(coclustering_mat, method, metric) {
  df <- data.frame(val=coclustering_mat$val, method=method, metric=metric)
  # Since the matrix is symmetric, call the "row" the larger index.
  df$row <- ifelse(coclustering_mat$rows <= coclustering_mat$cols,
                   coclustering_mat$cols, coclustering_mat$cols)
  df$col <- ifelse(coclustering_mat$rows > coclustering_mat$cols,
                   coclustering_mat$rows, coclustering_mat$cols)
  df$ind <- df$row * prod(dim(coclustering_mat)) + df$col
  return(df)
}

MakeCoclusteringMeanResultDataframe <- function(result_dict, method) {
  return(MakeCoclusteringResultDataframe(
    result_dict$cm_e, metric="Coclustering expectation", method=method))
}

MakeCoclusteringSdResultDataframe <- function(result_dict, method) {
  return(MakeCoclusteringResultDataframe(
      result_dict$cm_sd, metric="Coclustering standard deviation", method=method))
}

MakeResultDataframe <- function(MakeDFFun) {
  return(rbind(
    MakeDFFun(py_main$lr, "Linear_response"),
    MakeDFFun(py_main$hot, "Hot_start"),
    MakeDFFun(py_main$warm, "Warm_start"),
    MakeDFFun(py_main$cold, "Cold_start"))
  )
}

mi_df <- MakeResultDataframe(MakeMIResultDataframe)
fm_df <- MakeResultDataframe(MakeFMResultDataframe)
kl_df <- MakeResultDataframe(MakeKLResultDataframe)
cme_df <- MakeResultDataframe(MakeCoclusteringMeanResultDataframe)
cmsd_df <- MakeResultDataframe(MakeCoclusteringSdResultDataframe)

# Choose a random sample of rows to graph, otherwise the co-clustering matrix is too big.
sample_cols <- sample.int(max(cme_df$col), 300, replace=FALSE)
cme_cast_df <-
  dcast(cme_df, metric + row + col + ind ~ method, value.var="val") %>%
  filter(col %in% sample_cols, row %in% sample_cols)

cmsd_cast_df <-
  dcast(cmsd_df, metric + row + col + ind ~ method, value.var="val") %>%
  filter(col %in% sample_cols, row %in% sample_cols)


#########################
# Do some stuff with the model

json_model_filename <- file.path(data_path, "mice_data_fit_best.json")
lrvb_lib_path <- file.path(repo_path, "../LinearResponseVariationalBayes.py/")
autograd_lib_path <- file.path(repo_path, "../autograd/")
model_lib_path <- file.path(repo_path, "src/vb_modeling")

reticulate::py_run_string("
sys.path.insert(0, '" %_% model_lib_path %_% "')
#sys.path.insert(0, '" %_% autograd_lib_path %_% "')
sys.path.insert(0, '" %_% lrvb_lib_path %_% "')
import shift_only_lib as model_lib

with open('" %_% json_model_filename %_% "', 'r') as json_file:
  fit_dict = json.load(json_file)
model = model_lib.get_model_from_checkpoint(fit_dict)
")

# Load the timepoints.
reticulate::py_run_string("
sys.path.insert(0, '../src/vb_modeling')
from load_data import load_data
data_description_dict = fit_dict['data_description_dict']
timepoints, mapping, gene_names, y, meta = load_data(**data_description_dict)
")

reticulate::py_run_string("
import common_utilities_lib as util
means = model_lib.get_posterior_means(model.x, model.vb_params)
b = model.vb_params['global']['b'].e()
e_z = model.vb_params['e_z'].get()
beta = model.vb_params['global']['beta'].get()
")

reticulate::py_run_string("
timepoints_dense = np.linspace(0, np.max(timepoints), 100)
x_non_orth = util.get_bspline_design_matrix(timepoints, 10, model.beta_dim + 1, 3)
x_non_orth_dense = util.get_bspline_design_matrix(timepoints_dense, 10, model.beta_dim + 1, 3)
")

n <- 3
obs_k <- which.max(py_main$e_z[n, ])
py_main$e_z[n, obs_k]

obs_df <- data.frame(n=n,
                     t=py_main$timepoints + 1,
                     y=py_main$model$y[n, ],
                     means=py_main$means[n, ],
                     b=py_main$b[n, obs_k])

constant_basis_name <- paste("X", py_main$model$beta_dim + 1, sep="")
x_non_orth_df <-
  data.frame(py_main$x_non_orth) %>%
  mutate(t=py_main$timepoints + 1) %>%
  melt(id="t") %>%
  filter(variable != constant_basis_name)

x_non_orth_dense_df <-
  data.frame(py_main$x_non_orth_dense) %>%
  mutate(t=py_main$timepoints_dense + 1) %>%
  melt(id="t") %>%
  filter(variable != constant_basis_name)


###################
# Put salient numbers in a list here.

# note that list names must be valid latex commands -- so no underscores.
prior_param_dict <- py_main$model$prior_params$param_dict
numbers_list <- list(
  bnpalpha=prior_param_dict$alpha$get(),
  betamean=prior_param_dict$prior_beta$get(),
  betainfo=prior_param_dict$prior_beta_info$get(),
  gammascale=prior_param_dict$prior_gamma_scale$get(),
  gammashape=prior_param_dict$prior_gamma_shape$get(),
  bmean=prior_param_dict$prior_b$get(),
  binfo=prior_param_dict$prior_b_info$get(),
  
  nboot=max(mi_df$draw),
  kapprox=py_main$model$k_approx,
  nobs=py_main$model$n_obs,
  betadim=py_main$model$beta_dim,
  ntime=length(unique(py_main$timepoints)),
  splinedegree=3)

# Set the number of digits to display the number with in latex.
# Default is 0 (that is, integers)
number_precision_list <- list(
  betamean=2,
  betainfo=2,
  gammascale=2,
  gammashape=2,
  binfo=2)


###############################
# Save

save(mi_df, fm_df, kl_df, cme_cast_df, cmsd_cast_df,
     numbers_list, number_precision_list,
     obs_df, x_non_orth_dense_df, x_non_orth_df,
     file=file.path(paper_data_path, output_filename))


###############################

stop("Graphs following")

ggplot(obs_df) +
  geom_point(aes(x=t, y=y)) +
  geom_line(aes(x=t, y=means)) +
  geom_hline(aes(yintercept=b))

ggplot(x_non_orth_df) + 
  geom_line(aes(x=t, y=value, color=variable)) +
  geom_point(aes(x=t, y=value, color=variable))

ggplot() + 
  geom_line(data=x_non_orth_dense_df, aes(x=t, y=value, color=variable), lwd=1) +
  geom_point(data=x_non_orth_df, aes(x=t, y=value, color=variable), size=3) +
  geom_line(data=x_non_orth_df, aes(x=t, y=value, color=variable), alpha=0.7)

  
# Define LaTeX macros that will let us automatically refer
# to simulation and model parameters.
DefineMacro <- function(macro_name, value, digits=3) {
  sprintf_code <- paste("%0.", digits, "f", sep="")
  cat("\\newcommand{\\", macro_name, "}{", sprintf(sprintf_code, value), "}\n", sep="")
}

data_env <- environment()
data_env$numbers_list <- numbers_list
data_env$number_precision_list <- number_precision_list
# From the parametric sensitivy environment
for (numbername in names(data_env$numbers_list)) {
  if (numbername %in% names(data_env$number_precision_list)) {
    digits <- data_env$number_precision_list[[numbername]]
  } else {
    digits <- 0
  }
  DefineMacro(
    numbername,
    data_env$numbers_list[[numbername]],
    digits=digits)
}



ggplot(cmsd_cast_df) +
  geom_point(aes(x=Linear_response, y=Hot_start), alpha=0.05)

grid.arrange(
  ggplot(mi_df) +
    geom_density(aes(x=val, y=..density.., group=method, fill=method), alpha=0.7)
  ,  
  ggplot(fm_df) +
    geom_density(aes(x=val, y=..density.., group=method, fill=method), alpha=0.7)
  , ncol=2 
)
