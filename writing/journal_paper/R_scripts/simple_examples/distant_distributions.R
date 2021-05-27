#
# MakeDistPlot <- function(plot_var) {
#   ggplot(df_distant, aes(x=theta)) +
#       geom_line(aes(y=get(plot_var)), color="blue") +
#       ylab("Density") + xlab("stick propn")
# }
#
# grid.arrange(
#   MakeDistPlot("p"),
#   MakeDistPlot("p1"),
#   ncol=2
# )

df_distant <- path_env$df_distant
ggplot(df_distant, aes(x=theta)) +
    geom_line(aes(y=0.96 * p), color="blue") +
    geom_line(aes(y=p1), color="red") + ylab(NULL) + xlab("stick propn")
