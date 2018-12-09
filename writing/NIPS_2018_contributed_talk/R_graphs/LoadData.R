
results_file <- "results_for_paper.Rdata"
data_env <- LoadIntoEnvironment(file.path(data_path, results_file))

# The order also specifies the order of the plot.
keep_methods <- c("Cold_start", "Hot_start", "Linear_response")

GetMethodNameRow <- function(method, method_name) {
  return(data.frame(method=method, method_name=method_name))
}

method_names_df <- rbind(
  GetMethodNameRow("Cold_start", "Cold start"),
  GetMethodNameRow("Hot_start", "Warm start"),
  GetMethodNameRow("Linear_response", "Linear bootstrap")
) %>%
  mutate(method=ordered(method, keep_methods))

CleanDataFrame <- function(df) {
  clean_df <-
    filter(df, method %in% keep_methods) %>%
    mutate(method=ordered(method, keep_methods)) %>%
    inner_join(method_names_df, by="method")
  return(clean_df)
}
