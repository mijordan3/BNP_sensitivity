df <- path_env$df
theta_max <- path_env$theta_max
label_size <- 2
x_loc <- 0.75
# TODO: have the multiplicative and linear paths use the same code
BasePlot <- function(text, yloc) {
    ggplot(df %>% filter(theta <= theta_max), aes(color=t, group=t, x=theta)) +
        theme(legend.position="none") +
        scale_color_gradient(low="blue", high="red") +
        geom_label(aes(x=x_loc, y=yloc,
                  label=text),
                  color="black",
                  size=label_size,
                  vjust="top") +
        xlab("stick length") +
        get_fontsizes()
}


grid.arrange(
    BasePlot(text="Additive Path\nDensity", yloc=5) +
      geom_line(aes(y=p)) +
      ylab("Densities"),
    BasePlot(text="Additive Path\nLog density", yloc=3) +
      geom_line(aes(y=log(p))) +
      ylab("Log densities"),
    ncol=2
)
