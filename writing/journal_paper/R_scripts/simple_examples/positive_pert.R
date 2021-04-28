sim_env  <- LoadIntoEnvironment(
  './R_scripts/data_simulated/positive_pert.Rdata')
phi_range <- max(c(sim_env$df$phim, sim_env$df$phip))

grid.arrange(
    ggplot(sim_env$df) +
        geom_area(aes(x=x, y=pp, fill="plus"), alpha=0.1) +
        geom_area(aes(x=x, y=pbase, fill="base"), alpha=0.1) +
        geom_line(aes(x=x, y=pp, color="plus")) +
        geom_line(aes(x=x, y=pbase, color="base")) +
        xlab(TeX("$\\theta$")) + ylab(TeX("$P_1(\\theta)$ and $P_0(\\theta)$")) +
        ylim(0, 2) + theme(legend.position = "none")
    ,
    ggplot(sim_env$df) +
        geom_area(aes(x=x, y=pm, fill="minus"), alpha=0.1) +
        geom_area(aes(x=x, y=pbase, fill="base"), alpha=0.1) +
        geom_line(aes(x=x, y=pm, color="minus")) +
        geom_line(aes(x=x, y=pbase, color="base")) +
        xlab(TeX("$\\theta$")) + ylab(TeX("$P_1(\\theta)$ and $P_0(\\theta)$")) +
        ylim(0, 2) + theme(legend.position = "none")
    ,
    ggplot(sim_env$df) +
        geom_line(aes(x=x, y=phip, color="phi plus")) +
        geom_area(aes(x=x, y=phip, fill="phi plus"), alpha=0.1) +
        xlab(TeX("$\\theta$")) + ylab(TeX("$\\phi^{+}(\\theta)$")) +
        ylim(-1e-3, phi_range) + theme(legend.position = "none")
    ,
    ggplot(sim_env$df) +
        geom_line(aes(x=x, y=phim, color="phi minus")) +
        geom_area(aes(x=x, y=phim, fill="phi minus"), alpha=0.1) +
        xlab(TeX("$\\theta$")) + ylab(TeX("$\\phi^{-}(\\theta)$")) +
        ylim(-1e-3, phi_range) + theme(legend.position = "none")
    , ncol=2
)

