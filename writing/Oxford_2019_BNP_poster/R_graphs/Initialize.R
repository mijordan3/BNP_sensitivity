library(tidyverse)
library(knitr)
library(dplyr)
library(reshape2)
library(ggplot2)
library(xtable)
library(gridExtra)
library(latex2exp)

# This must be run from within the git repo, obviously.
git_repo_loc <- system("git rev-parse --show-toplevel", intern=TRUE)

paper_directory <- file.path(git_repo_loc, "writing/Oxford_2019_BNP_poster/")
data_path <- file.path(paper_directory, "R_graphs/data/")

# opts_chunk$set(fig.width=4.9, fig.height=3)
opts_chunk$set(fig.pos='!h', fig.align='center', dev='png', dpi=300)
opts_chunk$set(echo=knitr_debug, message=knitr_debug, warning=knitr_debug)

# Set the default ggplot theme
theme_set(theme_bw())

# Load into an environment rather than the global space
LoadIntoEnvironment <- function(filename) {
  my_env <- environment()
  load(filename, envir=my_env)
  return(my_env)
}

# Define LaTeX macros that will let us automatically refer
# to simulation and model parameters.
DefineMacro <- function(macro_name, value, digits=3) {
  sprintf_code <- paste("%0.", digits, "f", sep="")
  cat("\\newcommand{\\", macro_name, "}{", sprintf(sprintf_code, value), "}\n", sep="")
}

# These are based on one image per row.
base_aspect_ratio <- 3 / (5 * 2)
base_image_width <- 4.9 * 2

SetImageSize <- function(aspect_ratio, image_width=base_image_width) {
  ow <- "0.98\\linewidth"
  oh <- sprintf("%0.3f\\linewidth", aspect_ratio * 0.98)
  fw <- image_width
  fh <- image_width * aspect_ratio
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
