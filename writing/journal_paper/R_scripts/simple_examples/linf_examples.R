
df_ball <- path_env$df_ball

BaseBallPlot <- function(base_var) {
  base_lower <- paste0(base_var, "_lower")
  base_upper <- paste0(base_var, "_upper")

  ggplot(df_ball, aes(x=theta)) +
      geom_ribbon(aes(ymin=get(base_lower), ymax=get(base_upper)),
                  fill="gray", alpha=0.6, color="dark gray") +
      xlab("stick propn") +
      geom_line(aes(y=get(base_var))) +
      get_fontsizes()
}


df_distant <- path_env$df_distant
distant_plot <- ggplot(df_distant, aes(x=theta)) +
    geom_line(aes(y=0.96 * p), color="blue") +
    geom_line(aes(y=p1), color="red") + ylab(NULL) + xlab("stick propn") +
    get_fontsizes()


grid.arrange(
  BaseBallPlot("p") + ylab("Density")
,
  BaseBallPlot("logp") + ylab("Log density")
,
  distant_plot,
ncol=3
)
