library(tidyverse)


git_repo_loc <- "/home/rgiordan/Documents/git_repos/BNP_sensitivity/"
paper_directory <- file.path(git_repo_loc, "writing/journal_paper")
image_path <- file.path(paper_directory, "static_images")

Fun <- function(x, y) {
    r <- sqrt(x^2 + y^2)
    theta <- atan(y / x)
    abs_sin <- abs(sin(theta))
    return((r^2 / abs_sin) * exp(-r / abs_sin))
}


x_range <- 0.1
num_points <- 200
x_grid <- seq(-x_range, x_range, length.out=num_points)

df <-
    expand_grid(x=x_grid, y=x_grid) %>%
    mutate(f=Fun(x, y))


png(file.path(image_path, "averbukh_example.png"), units="in", width=6, height=4, res=300)
ggplot(df) +
    geom_raster(aes(x=x, y=y, fill=f)) +
    theme_bw() +
    theme(
        plot.background = element_blank(),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        panel.border = element_blank()
    )
dev.off()