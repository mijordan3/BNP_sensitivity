library(knitr)
library(dplyr)
library(reshape2)
library(ggplot2)
library(xtable)
library(gridExtra)
library(scales)
library(png)
library(latex2exp)

paper_directory <- "."

# opts_chunk$set(fig.width=4.9, fig.height=3)
opts_chunk$set(fig.pos='!h', fig.align='center', dev='png', dpi=300)
knitr_debug <- FALSE
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


# The location of data for this paper.
data_path <- file.path(paper_directory, "R_graphs/data/")

# A convenient function for extracting only the legend from a ggplot.
# Taken from
# http://www.sthda.com/english/wiki/...
# ggplot2-easy-way-to-mix-multiple-graphs-on-the-same-page-r-software-and-data-visualization
get_legend <- function(myggplot){
  tmp <- ggplot_gtable(ggplot_build(myggplot))
  leg <- which(sapply(tmp$grobs, function(x) x$name) == "guide-box")
  legend <- tmp$grobs[[leg]]
  return(legend)
}


# Sizes for images with multiple plots in them.  These sizes can be used
# inside a knitr chunk definition.

# These are based on one image per row.
aspect_ratio <- 2.8 / (5 * 2) # height / width
image_width <- 4.9 * 2

# A list for standardizing the size of images.
imsize <- list()

im1 <- list()
im1$ow <- "0.98\\linewidth"
im1$oh <- sprintf("%0.3f\\linewidth", aspect_ratio * 0.98)
im1$fw <- image_width
im1$fh <- image_width * aspect_ratio

# Make the default a one image..
opts_chunk$set(out.width=im1$ow, out.height=im1$oh,
               fig.width=im1$fw, fig.height=im1$fh)
