library(tidyverse)
library(gridExtra)
library(latex2exp)

git_repo_loc <- "/home/rgiordan/Documents/git_repos/BNP_sensitivity/"
paper_directory <- file.path(git_repo_loc, "writing/journal_paper")
image_path <- file.path(paper_directory, "static_images")

FunAverbukh <- function(x, y) {
    r <- sqrt(x^2 + y^2)
    theta <- atan(y / x)
    abs_sin <- abs(sin(theta))
    return((r^2 / abs_sin) * exp(-r / abs_sin))
}

FunMe <- function(x, y) {
    r <- sqrt(x^2 + y^2)
    theta <- atan(y / x)
    abs_sin <- abs(sin(theta))
    ratio <- r / abs_sin
    #return(ifelse(r > 0, ratio^2 / (1 + ratio^2), 0))
    return(ifelse(r > 0, ratio^2, 0))
}

TruncateForPlot <- function(x, quant=0.95, use_na=TRUE) {
    x_trim_level <- quantile(x, quant)
    if (use_na) {
        x[x > x_trim_level] <- NA
    } else {
        x[x > x_trim_level] <- x_trim_level
    }
    return(x)
}

#Fun <- FunAverbukh
Fun <- FunMe

# Averbukh
if (FALSE) {
    x_range <- 0.1
    num_points <- 200
}
if (TRUE) {
    x_range <- 0.01
    num_points <- 200
}

x_grid <- seq(-x_range, x_range, length.out=num_points)

df <-
    expand_grid(x=x_grid, y=x_grid) %>%
    mutate(f=Fun(x, y))

r_grid <- seq(0, x_range, length.out=num_points)
theta_vals <- asin(1 / c(1, 3, 4, 10, 100, 1000))
sin(theta_vals)
df_line <- do.call(
    bind_rows,
    lapply(theta_vals, function(theta) {
        # bind_rows(
        #     data.frame(r=sqrt(x_grid^2 + (x_grid * tan(theta))^2), theta=theta,
        #                f=Fun(x_grid, x_grid * tan(theta))),
        #     data.frame(r=0, theta=theta, f=0)
        # )
        bind_rows(
            data.frame(r=r_grid, theta=theta,
                       f=Fun(r_grid * cos(theta), r_grid * sin(theta))),
            data.frame(r=0, theta=theta, f=0)
        )
        }
    ))

ggplot(df_line %>% mutate(f=TruncateForPlot(f, quant=0.7))) +
    geom_line(aes(x=r, y=f, group=theta, color=log(theta)))


png(file.path(image_path, "pathological_r2_example.png"), units="in", width=6, height=3, res=300)
grid.arrange(
    #ggplot(df) +
    ggplot(df %>% mutate(f=TruncateForPlot(f, quant=0.95, use_na=FALSE))) +
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
    ggplot(df_line %>% mutate(f=TruncateForPlot(f, quant=0.7))) +
        geom_line(aes(x=r, y=f, group=theta, color=log(theta)))
, ncol=2
)
dev.off()






#######################################
# Prior perturbation size



PAlt <- function(x, eps, delta) {
    return(
        ifelse(x < eps, delta, (1 - delta * eps) / (1 - eps))
    )
}

x_grid <- seq(0, 1, length.out=1000)

p <- 2
epsilon <- 0.05
df <- data.frame(
    x=x_grid,
    pbase=1,
    pp=PAlt(x_grid, eps=epsilon, delta=2-epsilon),
    pm=PAlt(x_grid, eps=epsilon, delta=epsilon)) %>%
    mutate(ratiom=pbase / pm, ratiop=pbase / pp) %>%
    mutate(alpham=max(ratiom)^(1/p), alphap=max(ratiop)^(1/p)) %>%
    mutate(phim=alpham * pm^(1/p) - pbase^(1/p),
           phip=alphap * pp^(1/p) - pbase^(1/p))

phi_range <- max(c(df$phim, df$phip))


png(file.path(image_path, "positive_phi_example.png"), units="in", width=6, height=6, res=300)
grid.arrange(
    ggplot(df) +
        geom_area(aes(x=x, y=pp, fill="plus"), alpha=0.1) +
        geom_area(aes(x=x, y=pbase, fill="base"), alpha=0.1) +
        geom_line(aes(x=x, y=pp, color="plus")) +
        geom_line(aes(x=x, y=pbase, color="base")) +
        ylim(0, 2)
    ,
    ggplot(df) +
        geom_area(aes(x=x, y=pm, fill="minus"), alpha=0.1) +
        geom_area(aes(x=x, y=pbase, fill="base"), alpha=0.1) +
        geom_line(aes(x=x, y=pm, color="minus")) +
        geom_line(aes(x=x, y=pbase, color="base")) +
        ylim(0, 2)
    ,
    ggplot(df) +
        geom_line(aes(x=x, y=phip, color="phi plus")) +
        geom_area(aes(x=x, y=phip, fill="phi plus"), alpha=0.1) +
        ylim(-1e-3, phi_range)
    ,
    ggplot(df) +
        geom_line(aes(x=x, y=phim, color="phi minus")) +
        geom_area(aes(x=x, y=phim, fill="phi minus"), alpha=0.1) +
        ylim(-1e-3, phi_range)
    , ncol=2
)
dev.off()


