source('./R_scripts/initialize.R')
source('./R_scripts/plotting_utils.R')

#################
# iris results
#################
# this is from the journal writing folder
load('../journal_paper/R_scripts/data_processed/iris.RData')

source('./R_scripts/iris/iris_init_fit.R')
source('./R_scripts/iris/iris_alpha_sens.R')
source('./R_scripts/iris/iris_func_sens.R')

#################
# structure results
#################
load('../journal_paper/R_scripts/data_processed/structure.RData')
source('./R_scripts/structure/structure_plotting_utils.R')

# plot and save initial fit
out <- plot_initial_fit()
out$p
save_last_fig('./figures/structure_example.png', 
              aspect_ratio = 0.45)
