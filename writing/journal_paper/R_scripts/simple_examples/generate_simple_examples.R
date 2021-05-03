library(tidyverse)
library(gridExtra)
library(latex2exp)

git_repo_loc <- "/home/rgiordan/Documents/git_repos/BNP_sensitivity/"
paper_directory <- file.path(git_repo_loc, "writing/journal_paper")
image_path <- file.path(paper_directory, "static_images")
data_path <- file.path(paper_directory, "R_scripts/data_simulated")


##########################################################
# R2 non-differentiable function

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

if (FALSE) {
    save(df_line, df, file=file.path(data_path, "nondifferentiable_r2.Rdata"))
}


#png(file.path(image_path, "pathological_r2_example.png"), units="in", width=6, height=3, res=300)
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
#dev.off()






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

if (FALSE) {
    save(df, file=file.path(data_path, "positive_pert.Rdata"))
}

#png(file.path(image_path, "positive_phi_example.png"), units="in", width=6, height=6, res=300)
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
#dev.off()







##############################################################################
# Some examples of lienar interpolation

dens_min <- 0.
theta_max <- 1.0

alpha0 <- 1
alpha1 <- 5
# dens_min <- 0.05
# theta_max <- 1.0

theta_grid <- seq(0, 1-1e-4, length.out=300)
p0 <- dbeta(theta_grid, 1, alpha0) + dens_min
#p1 <- dbeta(theta_grid, 1, alpha1) + dens_min
#p1 <- cos((theta_grid - 0.5) * 2* pi) + 1.03
p1 <- cos(theta_grid * pi) + 1.03
p1 <- p1 / (sum(p1) * min(diff(theta_grid)))

log_p0 <- dbeta(theta_grid, 1, alpha0, log=TRUE)
log_p1 <- dbeta(theta_grid, 1, alpha1, log=TRUE)
max(abs(log_p1 - log_p0))

num_t <- 5

df <- data.frame(theta=theta_grid, p=p0, logp=log(p0), t=0)
t_grid <- seq(1 / num_t, 1, length.out=num_t)
for (t in t_grid) {
    df <- bind_rows(
        df,
        data.frame(theta=theta_grid,
                   p=p0 * (1 - t) + t * p1,
                   logp=log(p0) * (1 - t) + t * log(p1), t=t))
}

BasePlot <- function() {
    ggplot(df %>% filter(theta <= theta_max), aes(color=t, group=t, x=theta)) +
        theme(legend.position="none") +
        scale_color_gradient(low="blue", high="red") +
        xlab(TeX("$\\theta$"))
}


grid.arrange(
    BasePlot() + geom_line(aes(y=exp(logp))) + ggtitle("Multiplicative perturbation") + ylab("Densities"),
    BasePlot() + geom_line(aes(y=logp)) + ggtitle("Multiplicative perturbation") + ylab("Log densities"),
    BasePlot() + geom_line(aes(y=p)) + ggtitle("Linear perturbation") + ylab("Densities"),  
    BasePlot() + geom_line(aes(y=log(p))) + ggtitle("Linear perturbation") + ylab("Log densities"),
    ncol=2 
)



pball <- dbeta(theta_grid, 1, 3) + 0.1

ball_width <- 0.5
df_ball <-
    data.frame(theta=theta_grid, p=pball, logp=log(pball)) %>%
    mutate(logp_upper=logp + ball_width,
           logp_lower=logp - ball_width,
           p_upper=exp(logp_upper),
           p_lower=exp(logp_lower))


base_var <- "p"
base_lower <- paste0(base_var, "_lower")
base_upper <- paste0(base_var, "_upper")

ggplot(df_ball, aes(x=theta)) +
    geom_ribbon(aes(ymin=get(base_lower), ymax=get(base_upper)),
                fill="gray", alpha=0.6, color="dark gray") +
    geom_line(aes(y=get(base_var)))


grid.arrange(
    ggplot(df_ball, aes(x=theta)) +
        geom_ribbon(aes(ymin=p_lower, ymax=p_upper), fill="gray", alpha=0.6, color="dark gray") +
        geom_line(aes(y=p))
,
    ggplot(df_ball, aes(x=theta)) +
        geom_ribbon(aes(ymin=logp_lower, ymax=logp_upper), fill="gray", alpha=0.6, color="dark gray") +
        geom_line(aes(y=logp))
, ncol=2
)


df_distant <-
    data.frame(theta=theta_grid, p=pball) %>%
    mutate(p1=ifelse(abs(theta - 0.25) > 0.005, p, 0.01)) %>%
    mutate(p1=p1 * sum(p) / sum(p1))


if (FALSE) {
    grid.arrange(
        ggplot(df_distant, aes(x=theta)) +
            geom_line(aes(y=p), color="blue"),
        ggplot(df_distant, aes(x=theta)) +
            geom_line(aes(y=p1), color="blue"),
        ncol=2
    )

    ggplot(df_distant, aes(x=theta)) +
        geom_line(aes(y=0.96 * p), color="blue") +
        geom_line(aes(y=p1), color="red") + ylab(NULL)
}



if (FALSE) {
    save(dens_min, theta_max, df,
         ball_width, df_ball,
         df_distant,
         file=file.path(data_path, "function_paths.Rdata"))
}


