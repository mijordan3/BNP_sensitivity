
library(Matrix)
library(reshape2)
library(dplyr)
library(ggplot2)
library(StructureLRVB)


repo_loc <- file.path(Sys.getenv("GIT_REPO_LOC"),
                      "BNP_sensitivity/structure")
source(file.path(repo_loc, "get_sensitivity_lib.R"))

data <- ReadStructureData(
  input.basename = file.path(repo_loc, "test_input_data/testdata_small"),
  output.basename = file.path(repo_loc, "test_input_data/testdata_small_out.3"))

data$genome.matrix
