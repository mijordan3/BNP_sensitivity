library(tidyverse)
library(reshape2)
library(reticulate)
library(gridExtra)

reticulate::use_python("/usr/bin/python3")
py_main <- reticulate::import_main()

reticulate::py_run_string("
import numpy as np
import json_tricks
")

fit_dir <- "/home/rgiordan/Documents/git_repos/BNP_sensitivity/RegressionClustering/fits/cluster"


genes <- "7000"
results_filename <- sprintf("combined_results_genes%s.Rdata", genes)

raw_results <- data.frame()
for (inflate in c("0.0", "1.0")) {
  fit_files <- system(sprintf(
    "ls %s/*genes%s_*inflate%s*_analysis.json",
    fit_dir, genes, inflate), intern=TRUE)
  analysis_name <- sprintf("genes%s_inflate%s", genes, inflate)
  
  ResultDictToDF <- function(result_dict) {
    df  <- data.frame(analysis=analysis_name)
    for (key in names(result_dict)) {
      df[[key]] <- result_dict[key]
    }
    return(df)
  }
  
  for (fit_file in fit_files) {
    cat("Running for ", fit_file, "\n")
    #py_main$fit_filename <- file.path(fit_dir, fit_file)
    py_main$fit_filename <- fit_file
    reticulate::py_run_string("
with open(fit_filename, 'r') as infile:
  this_result = json_tricks.loads(infile.read())
")
    
    for (i in 1:length(py_main$this_result)) {
      raw_results <- bind_rows(
        raw_results,
        ResultDictToDF(py_main$this_result[[i]]) %>%
          mutate(inflate=inflate != "0.0"))
    }
  }
}

results <- raw_results %>%
  select(-refit_filename)  %>%
  mutate(alpha_increase=alpha1 > alpha0) %>%
  mutate(functional=case_when(is.na(functional) ~ FALSE, TRUE ~ functional))

table(results[c("inflate", "alpha1")])
table(results[c("inflate", "epsilon")])
table(results[c("inflate", "functional")])

# Functional perturbations
MakeFunPlot <- function(use_predictive, use_inflate, use_log_phi_desc) {
  ggplot(filter(results,
                functional==TRUE,
                log_phi_desc == use_log_phi_desc,
                predictive==use_predictive,
                inflate==use_inflate)) +
    geom_point(aes(x=epsilon, y=e_num1, color="Refitting")) +
    geom_point(aes(x=epsilon, y=e_num_pred, color="Approximation")) +
    geom_line(aes(x=epsilon, y=e_num1, color="Refitting")) +
    geom_line(aes(x=epsilon, y=e_num_pred, color="Approximation")) +
    geom_vline(aes(xintercept=0.0)) +
    facet_grid(threshold ~ ., scales="free", labeller = label_context) +
    ggtitle(sprintf("%s\n%s functional perturbation\n%s",
                    ifelse(use_inflate, "Noise added", "No noise added"),
                    log_phi_desc,
                    ifelse(use_predictive, "Predictive", "In-sample"))) +
    xlab("Epsilon") + ylab("Expected number of clusters")
}


grid.arrange(
  MakeFunPlot(TRUE, TRUE, "expit"),
  MakeFunPlot(TRUE, FALSE, "expit"),
  MakeFunPlot(FALSE, TRUE, "expit"),
  MakeFunPlot(FALSE, FALSE, "expit"),
  ncol=4
)


# Parametric perturbations
MakePlot <- function(use_alpha_increase, use_predictive, use_inflate) {
  ggplot(filter(results,
                !functional,
                alpha_increase==use_alpha_increase,
                predictive==use_predictive,
                inflate==use_inflate)) +
    geom_point(aes(x=alpha1, y=e_num1, color="Refitting")) +
    geom_point(aes(x=alpha1, y=e_num_pred, color="Approximation")) +
    geom_line(aes(x=alpha1, y=e_num1, color="Refitting")) +
    geom_line(aes(x=alpha1, y=e_num_pred, color="Approximation")) +
    geom_vline(aes(xintercept=alpha0)) +
    facet_grid(threshold ~ ., scales="free", labeller = label_context) +
    ggtitle(sprintf("%s\n%s\n%s",
            ifelse(use_inflate, "Noise added", "No noise added"),
            ifelse(use_alpha_increase, "Increasing alpha", "Decreasing alpha"),
            ifelse(use_predictive, "Predictive", "In-sample"))) +
    xlab("Alpha") + ylab("Expected number of clusters")
}


use_predictive <- FALSE
grid.arrange(
  MakePlot(FALSE, use_predictive, FALSE),
  MakePlot(FALSE, use_predictive, TRUE),
  MakePlot(TRUE, use_predictive, FALSE),
  MakePlot(TRUE, use_predictive, TRUE),
  ncol=4
)



###################################################
# Load the shape of the prior perturbation


prior_shape_file <- "prior_functional_perturbation_shape_expit.npz"
py_main$prior_shape_file <- file.path(fit_dir, prior_shape_file)
reticulate::py_run_string("
with np.load(prior_shape_file) as infile:
  logit_v_grid = infile['logit_v_grid']
  v_grid = infile['v_grid']
  log_p0_logit = infile['log_p0_logit']
  log_p1_logit = infile['log_p1_logit']
  log_p0 = infile['log_p0']
  log_p1 = infile['log_p1']
  log_phi = infile['log_phi']
")
pert_df <- data.frame(
  logit_v_grid=py_main$logit_v_grid,
  v_grid=py_main$v_grid,
  log_p0_logit=py_main$log_p0_logit,
  log_p1_logit=py_main$log_p1_logit,
  log_p0=py_main$log_p0,
  log_p1=py_main$log_p1,
  log_phi=py_main$log_phi
)

ggplot(pert_df) +
  geom_line(aes(x=v_grid, y=exp(log_p0))) +
  geom_line(aes(x=v_grid, y=exp(log_p1)))



##############################
# Load some gene shapes

inflate <- "1.0"
pre_genes <- as.character(as.integer((as.integer(genes) / 0.7)))
gene_shapes <- data.frame()

for (inflate in c("0.0", "1.0")) {
  py_main$datafile <- file.path(
    fit_dir,
    sprintf('shrunken_transformed_gene_regression_df4_degree3_genes%s_inflate%s.npz', pre_genes, inflate))
  py_run_string("
with np.load(datafile) as infile:
  beta_mean = infile['transformed_beta_mean']
  beta_info = infile['transformed_beta_info']

beta_sd = np.array([ np.diag(np.sqrt(np.linalg.inv(beta_info[n, :, :])))
                     for n in range(beta_info.shape[0])])
")
  
  gene_shapes <- bind_rows(
    gene_shapes,
    data.frame(
      gene=1:nrow(py_main$beta_mean),
      beta_mean=py_main$beta_mean,
      beta_sd=py_main$beta_sd,
      inflate=inflate,
      genes=genes
    )
  )
}


gene_shapes_melt <- 
  melt(gene_shapes, id.vars=c("genes", "inflate", "gene")) %>%
  separate(variable, c("variable", "ind"), "\\.") %>%
  #mutate(ind=paste("d", ind, sep="")) %>%
  mutate(ind=as.numeric(ind), gene=factor(gene)) %>%
  dcast(ind + gene + genes + inflate ~ variable)
head(gene_shapes_melt)

ggplot(filter(gene_shapes_melt, as.integer(gene) <= 6),
       aes(fill=gene, group=gene, color=gene)) +
  geom_ribbon(aes(x=ind,
                  ymin=beta_mean - 1.64 * beta_sd,
                  ymax=beta_mean + 1.64 * beta_sd, group=gene),
              color=NA, alpha=0.4) +
  geom_ribbon(aes(x=ind,
                  ymin=beta_mean - beta_sd,
                  ymax=beta_mean + beta_sd, group=gene),
              color=NA, alpha=0.4) +
  geom_line(aes(x=ind, y=beta_mean), lwd=2) +
  facet_grid(inflate ~ gene)



##############################
# Save for knitr

save(pert_df, results, gene_shapes_melt, file=file.path(fit_dir, results_filename))



