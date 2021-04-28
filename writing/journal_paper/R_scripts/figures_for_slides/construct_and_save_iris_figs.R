# the initialization file from our paper
knitr_debug <- FALSE
source('../initialize.R')

# plotting utils
source('../plotting_utils.R')

# load data 
load('../data_processed/iris.RData')

# initialization file for slides
source('./initialize_figures_for_slides.R')

# construct iris plots
source('./iris/iris_init_fit.R')
source('./iris/iris_alpha_sens.R')
source('./iris/iris_func_sens.R')
