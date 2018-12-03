cmsd_cast_df <- data_env$cmsd_cast_df

MakeCoclusterDensityGraph <- function(x, y, min_keep=0.03) {
  non_nan <- c(x, y)
  non_nan <- non_nan[is.finite(non_nan)]
  ranges <- c(-0.0001, quantile(non_nan, 0.99))
  bandwidth <- diff(ranges) / 10
  
  keep <- (x > min_keep) | (y > min_keep)

  return(
    ggplot(data=NULL, aes(x=x[keep], y=y[keep])) +
      stat_density_2d(aes(fill=..level..), geom="polygon", h=c(bandwidth, bandwidth)) +
      geom_point(alpha=0.01) +
      scale_fill_gradient(low = "white", high = "blue") +
      geom_abline(aes(slope=1, intercept=0), lwd=1) +
      theme(legend.position="none") +
      xlim(ranges[1], ranges[2]) + ylim(ranges[1], ranges[2]) +
      geom_polygon(aes(x=c(0, min_keep, min_keep, 0, 0),
                       y=c(0, 0, min_keep, min_keep, 0)),
                   fill="dark gray", color="black", lwd=1) #+
      #geom_text(aes(x=min_keep / 2, y=min_keep / 2, label="Excluded"))
  )
}

min_keep <- 0.03

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
