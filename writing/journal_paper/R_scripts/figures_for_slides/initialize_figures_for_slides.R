# the old initialization file
knitr_debug <- FALSE
source('../initialize.R')
source('../plotting_utils.R')


# function to save figures as png file

baseline_width <- 5

save_last_fig <- function(outfile, 
                          base_factor = 1.0, 
                          aspect_ratio = 1.0){
  ggsave(paste0(outfolder, outfile), 
         width = baseline_width * base_factor, 
         height = baseline_width * base_factor * aspect_ratio, 
         unit = 'in')
}