# This file is produced by R_graphs/data/iris_data/make_knitr_dataset.R.
iris_data <- LoadIntoEnvironment(
    file.path(data_path, "iris_data/iris_data_for_knitr.Rdata"))
# For consistency with the mouse dataset:
iris_data$results_df$method <-
    sub("linear approx", "approx", iris_data$results_df$method)
