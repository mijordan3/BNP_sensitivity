source(file.path(r_script_path, "simple_examples/path_graph_lib.R"))

df <- path_env$df
grid.arrange(
    FuncPathPlot() +
      geom_line(aes(y=exp(logp))) +
      ggtitle("Multiplicative path: Density") +
      ylab("Densities"),
    FuncPathPlot() +
      geom_line(aes(y=logp)) +
      ggtitle("Multiplicative path: Log density") +
      ylab("Log densities"),
    ncol=2
)
