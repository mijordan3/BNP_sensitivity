# Initialize R for knitr.

library(tidyverse)
library(knitr)
library(dplyr)
library(reshape2)
library(ggplot2)
library(ggforce)
library(xtable)

library(gridExtra)
library(patchwork)

library(latex2exp)

# Set the default ggplot theme
theme_set(theme_bw())

# set fontsizes 
axis_ticksize = 4 
axis_title_size = 7
title_size = 7 

get_fontsizes <- function(scaling = 1){
  axis_ticksize = axis_ticksize * scaling
  axis_title_size = axis_title_size * scaling
  title_size = title_size * scaling
  
  fontsize_theme <- theme(axis.text.x = element_text(size = axis_ticksize),
                          axis.text.y = element_text(size = axis_ticksize),  
                          axis.title.x = element_text(size = axis_title_size),
                          axis.title.y = element_text(size = axis_title_size), 
                          legend.text = element_text(size=axis_title_size), 
                          plot.title = element_text(size = title_size), 
                          axis.ticks.length = unit(0.05, "cm"), 
                          strip.text = element_text(size = axis_title_size, 
                                                    margin = margin(.05, 0, .05, 0, "cm")),
                          legend.margin=margin(-10,-10,-10,-10))
  
  return(fontsize_theme)
}

baseline_width <- 5

save_last_fig <- function(outfile, 
                          base_factor = 1.0, 
                          aspect_ratio = 1.0){
  ggsave(outfile, 
         width = baseline_width * base_factor, 
         height = baseline_width * base_factor * aspect_ratio, 
         unit = 'in')
}
