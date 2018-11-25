
cool_data_env <- LoadIntoEnvironment(
    file.path(data_path, "results_for_paper_warm.Rdata"))

cool_mi_df <- CleanDataFrame(cool_data_env$mi_df)
cool_fm_df <- CleanDataFrame(cool_data_env$fm_df)
legend_graph <- get_legend(MakeScoreDensityGraph(cool_mi_df))

grid.arrange(
  MakeScoreDensityGraph(cool_mi_df, use_legends=FALSE) + xlab("Mutual information score")
  ,
  MakeScoreDensityGraph(cool_fm_df, use_legends=FALSE) + xlab("Fowlkes-Mallows score")
  ,
  legend_graph
  ,
  ncol=3, widths=c(0.4, 0.4, 0.2)
)


cmsd_cast_df <- cool_data_env$cmsd_cast_df
grid.arrange(
MakeCoclusterDensityGraph(
  cmsd_cast_df$Hot_start, cmsd_cast_df$Linear_response, min_keep=min_keep) +
  xlab("Warm start") + ylab("Linear response")
,
MakeCoclusterDensityGraph(
  cmsd_cast_df$Hot_start, cmsd_cast_df$Cold_start, min_keep=min_keep) +
  xlab("Warm start") + ylab("Cold start")
,
ncol=2
)
