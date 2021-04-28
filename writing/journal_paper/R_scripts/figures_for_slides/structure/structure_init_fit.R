# initialize
source('./initialize_figures_for_slides.R')

# load data for structure
load('../data_processed/structure.RData')
source('../structure/structure_plotting_utils.R')

initial_fit <- plot_initial_fit() 

initial_fit$p + 
  theme(legend.position = 'none')

save_last_fig('structure_example.png', 
              aspect_ratio = 0.5)
