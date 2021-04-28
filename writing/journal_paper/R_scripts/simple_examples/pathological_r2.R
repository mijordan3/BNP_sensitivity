
sim_env  <- LoadIntoEnvironment(
  './R_scripts/data_simulated/nondifferentiable_r2.Rdata')

TruncateForPlot <- function(x, quant=0.95, use_na=TRUE) {
    x_trim_level <- quantile(x, quant)
    if (use_na) {
        x[x > x_trim_level] <- NA
    } else {
        x[x > x_trim_level] <- x_trim_level
    }
    return(x)
}

grid.arrange(
    ggplot(sim_env$df %>% mutate(f=TruncateForPlot(f, quant=0.95, use_na=FALSE))) +
        geom_raster(aes(x=x, y=y, fill=f)) +
        theme_bw() +
        theme(
            plot.background = element_blank(),
            panel.grid.major = element_blank(),
            panel.grid.minor = element_blank(),
            panel.border = element_blank()
        ) +
        xlab(TeX("$x_1$")) + ylab(TeX("$x_2$"))
,
    ggplot(sim_env$df_line %>% mutate(f=TruncateForPlot(f, quant=0.7))) +
        geom_line(aes(x=r, y=f, group=theta, color=log(theta)))
, ncol=1
)
