# this is the initialization file from our paper
knitr_debug <- FALSE
source('../initialize.R')

# set folder where figures will be saved
outfolder <- '/home/rliu/Documents/BNP/student_seminar2021/figures/'

# the initilization file for our slides
# contains the function to save plots as png
source('./initialize_figures_for_slides.R')

# run your source code
source('code_to_make_plots.R')

# assume plot is saved into variable ``p"
# set the fontsizes for plot "p"
p <- p + get_fontsizes()
p

# save the last figure constructed (aka p)
base_factor = 0.8 # this is the multiplier in front of \textwdith in TeX
aspect_ratio = 0.5 # some aspect ratio
save_last_fig(name_of_figure, 
              base_factor = base_factor, 
              aspect_ratio = aspect_ratio)
