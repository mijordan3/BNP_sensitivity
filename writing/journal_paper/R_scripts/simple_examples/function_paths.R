df <- path_env$df
theta_max <- path_env$theta_max
# TODO: have the multiplicative and linear paths use the same code
BasePlot <- function() {
    ggplot(df %>% filter(theta <= theta_max), aes(color=t, group=t, x=theta)) +
        theme(legend.position="none") +
        scale_color_gradient(low="blue", high="red") +
        xlab(TeX("$\\theta$")) +
        get_fontsizes()
}


grid.arrange(
    BasePlot() + geom_line(aes(y=exp(logp))) +
      ggtitle("Multiplicative Perturbations") + ylab("Densities"),
    BasePlot() + geom_line(aes(y=logp)) +
      ggtitle("Multiplicative Perturbations") + ylab("Log densities"),
    BasePlot() + geom_line(aes(y=p)) +
      ggtitle("Linear Perturbations") + ylab("Densities"),
    BasePlot() + geom_line(aes(y=log(p))) +
      ggtitle("Linear Perturbations") + ylab("Log densities"),
    ncol=2
)
