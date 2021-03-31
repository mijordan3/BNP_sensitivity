# Initialize R for knitr.

library(tidyverse)
library(knitr)
library(dplyr)
library(reshape2)
library(ggplot2)
library(ggforce)
library(xtable)

library(gridExtra)
# this replaces gridExtra ...
library(patchwork) 

library(latex2exp)
library(reticulate)
np <- import("numpy")


# This must be run from within the git repo, obviously.
git_repo_loc <- system("git rev-parse --show-toplevel", intern=TRUE)

paper_directory <- file.path(git_repo_loc, "writing/")
data_path <- file.path(paper_directory, "data/")

# opts_chunk$set(fig.width=4.9, fig.height=3)
opts_chunk$set(fig.pos='!h', fig.align='center', dev='png', dpi=300)
opts_chunk$set(echo=knitr_debug, message=knitr_debug, warning=knitr_debug)

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
                          legend.margin=margin(-10,-10,-10,-10))
  
  return(fontsize_theme)
}

# Load into an environment rather than the global space
LoadIntoEnvironment <- function(filename) {
  my_env <- environment()
  load(filename, envir=my_env)
  return(my_env)
}

# Define LaTeX macros that will let us automatically refer
# to simulation and model parameters.
DefineMacro <- function(macro_name, value, digits=3) {
  #sprintf_code <- paste("%0.", digits, "f", sep="")
  value_string <- format(value, big.mark=",", digits=digits, scientific=FALSE)
  cat("\\newcommand{\\", macro_name, "}{", value_string, "}\n", sep="")
}

base_aspect_ratio <- 8 / (5 * 2)
base_image_width <- 4.

SetImageSize <- function(aspect_ratio, image_width=1.0) {
  
  ow <- sprintf("%0.3f\\linewidth", image_width * 0.98)
  oh <- sprintf("%0.3f\\linewidth", aspect_ratio * image_width * 0.98)
  
  fw <- base_image_width * image_width
  fh <- fw * aspect_ratio
  
  opts_chunk$set(out.width=ow,
                 out.height=oh,
                 fig.width=fw,
                 fig.height=fh)
}


SetFullImageSize <- function() SetImageSize(
    aspect_ratio=base_aspect_ratio, image_width=base_image_width)

# Default to a full image.
SetFullImageSize()

# A convenient funciton for extracting only the legend from a ggplot.
# Taken from
# http://www.sthda.com/english/wiki/ggplot2-easy-way-to-mix-multiple-graphs-on-the-same-page-r-software-and-data-visualization
GetLegend <- function(myggplot){
  tmp <- ggplot_gtable(ggplot_build(myggplot))
  leg <- which(sapply(tmp$grobs, function(x) x$name) == "guide-box")
  legend <- tmp$grobs[[leg]]
  return(legend)
}


# Define common colors.
GGColorHue <- function(n) {
  hues = seq(15, 375, length = n + 1)
  hcl(h = hues, l = 65, c = 100)[1:n]
}

GetGraphColors <- function(legend_breaks) {
  stopifnot(length(legend_breaks) <= 4)
  graph_colors <- GGColorHue(4)[1:length(legend_breaks)]
  names(graph_colors) <- legend_breaks
  return(graph_colors)
}
