
MakeScoreDensityGraph <- function(df, use_legends=TRUE) {
  plot_alpha <- 0.8 / 2
  return(
    ggplot(df) +
      geom_histogram(aes(x=val, y=..density.., group=method, fill=method),
                     position="identity", alpha=plot_alpha, bins=30) +
      geom_density(aes(x=val, y=..density.., group=method, color=method),
                     alpha=0.5 * plot_alpha, fill=NA, lwd=1, show.legend=FALSE) +
      scale_fill_discrete(
        name='Method', 
        breaks=levels(df$method), labels=levels(df$method_name)) +
      ylab("") + theme(legend.position=ifelse(use_legends, "right", "none"))
    )
}


mi_df <- CleanDataFrame(data_env$mi_df)
fm_df <- CleanDataFrame(data_env$fm_df)
legend_graph <- get_legend(MakeScoreDensityGraph(mi_df))

grid.arrange(
  MakeScoreDensityGraph(mi_df, use_legends=FALSE) + xlab("Mutual information score")
  ,
  MakeScoreDensityGraph(fm_df, use_legends=FALSE) + xlab("Fowlkes-Mallows score")
  ,
  legend_graph
  ,
  ncol=3, widths=c(0.4, 0.4, 0.2)
)
