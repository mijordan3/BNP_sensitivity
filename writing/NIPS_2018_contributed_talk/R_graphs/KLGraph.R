kl_df <-
  CleanDataFrame(data_env$kl_df) %>%
  dcast(metric + draw ~ method, value.var="val")

ranges <- c(quantile(data_env$kl_df$val, 0.01), quantile(data_env$kl_df$val, 0.99))
bandwidth <- diff(ranges) / 10

MakeKLDensityGraph <- function(x, y) {
  return(
    ggplot(data=NULL, aes(x=x, y=y)) +
      stat_density_2d(aes(fill=..level..), geom="polygon", alpha=1.0, h=c(bandwidth, bandwidth)) +
      geom_point() +
      scale_fill_gradient(low = "white", high = "blue") +
      geom_abline(aes(slope=1, intercept=0), lwd=1) +
      theme(legend.position="none") +
      xlim(ranges[1], ranges[2]) + ylim(ranges[1], ranges[2]) + xlab("Warm start optimal KL")
  )
}

grid.arrange(
  MakeKLDensityGraph(kl_df$Hot_start, kl_df$Cold_start) +
    ggtitle('Warm start vs cold start') + ylab("Cold start optimal KL")
  ,
  MakeKLDensityGraph(kl_df$Hot_start, kl_df$Linear_response) +
    ggtitle('Warm start vs linear approximation') + ylab("Linear approximation KL (not optimal)")
  ,
  ncol=2
)

