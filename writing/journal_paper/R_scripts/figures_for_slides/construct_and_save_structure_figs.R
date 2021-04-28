# the initialization file from our paper
knitr_debug <- FALSE
source('../initialize.R')

# plotting utils
source('../plotting_utils.R')

# load data 
load('../data_processed/structure.RData')

# initialization file for slides
source('./initialize_figures_for_slides.R')

# construct structure plots
source('./structure/structure_init_fit.R')
source('./structure/structure_n_clusters_alphasens.R')
source('./structure/structure_fsens.R')
