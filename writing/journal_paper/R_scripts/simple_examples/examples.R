library(tidyverse)
library(gridExtra)

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
    return(ifelse(r > 0, ratio / (1 + ratio), 0))
}

#Fun <- FunAverbukh
Fun <- FunMe

# Averbukh
if (FALSE) {
    x_range <- 0.1
    num_points <- 200
}
if (TRUE) {
    x_range <- 0.5
    num_points <- 200
}

x_grid <- seq(-x_range, x_range, length.out=num_points)

df <-
    expand_grid(x=x_grid, y=x_grid) %>%
    mutate(f=Fun(x, y))

theta_vals <- c(0.5, 0.1, 0.001, 0.0001)
df_line <- do.call(
    bind_rows,
    lapply(theta_vals, function(theta) {
        bind_rows(
            data.frame(x=x_grid, theta=theta,
                       f=Fun(x_grid, x_grid * tan(theta))),
            data.frame(x=0, theta=theta, f=0)
        )}
    ))



png(file.path(image_path, "pathological_r2_example.png"), units="in", width=6, height=3, res=300)
grid.arrange(
    ggplot(df) +
        geom_raster(aes(x=x, y=y, fill=f)) +
        theme_bw() +
        theme(
            plot.background = element_blank(),
            panel.grid.major = element_blank(),
            panel.grid.minor = element_blank(),
            panel.border = element_blank()
        )
,
    ggplot(df_line) +
        geom_line(aes(x=x, y=f, group=theta, color=theta))
, ncol=2
)
dev.off()
