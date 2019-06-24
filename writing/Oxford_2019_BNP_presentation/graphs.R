# Use the knitr framework from the poster.

setwd("/home/rgiordan/Documents/git_repos/BNP_sensitivity/writing/Oxford_2019_BNP_poster/")
knitr_debug <- FALSE
source("R_graphs/Initialize.R")
source("R_graphs/CommonGraphs.R")

# Puts iris_data in the environment.
source("./R_graphs/LoadIrisData.R")

# Puts genomics_data in the environment.
source("./R_graphs/ProcessGenomicsData.R")

plot_dir <- "/home/rgiordan/Documents/git_repos/BNP_sensitivity/writing/Oxford_2019_BNP_presentation/"

SavePlot <- function(g, filename, width=10, height=5) {
  ggsave(g, width=width, height=height, units="in", file=file.path(plot_dir, filename))
}

# Use both inflated and not for the presentation, unlike the poster
genomics_data$processed_results <-
  ProcessGenomicsData(genomics_data$results)

table(genomics_data$processed_results[, c("functional", "inflate")])
table(genomics_data$results[, c("functional", "inflate")])

# Iris distribution
g <-
  ggplot(iris_data$iris_df) +
  geom_point(aes(x=PC1, y=PC2, color=Species)) +
  theme(legend.position="none")
ggsave(width=5, height=3, units="in", file=file.path(plot_dir, "iris_w_species.png"))

ggplot(iris_data$iris_df) + geom_point(aes(x=PC1, y=PC2))
ggsave(width=5, height=3, units="in", file=file.path(plot_dir, "iris_no_species.png"))


# Iris parametric

results_df_1 <- filter(iris_data$results_df, !pred, pert=="alpha3")
results_df_2 <- filter(iris_data$results_df, !pred, pert=="alpha8")
results_df_3 <- filter(iris_data$results_df, !pred, pert=="alpha13")

# Include the other two plots to get the scaling right.
g <- grid.arrange(
  plot_parametric_sensitivity(
    filter(results_df_1, method=="refitted"), alpha_0 = -1) +
    scale_color_discrete(limits=c("approx", "refitted")) +
    ggtitle('Iris data') + theme(legend.position="none"),
  plot_parametric_sensitivity(results_df_2, alpha_0 = 8.0) +
    ggtitle(' ') + theme(legend.position="none"),
  plot_parametric_sensitivity(results_df_3, alpha_0 = 13.0) +
    ggtitle(' ') + theme(legend.position="None"),
  ncol=3)
SavePlot(g, "iris_parametric_refitonly.png", height=4)


g <- grid.arrange(
  plot_parametric_sensitivity(results_df_1, alpha_0 = 3.0) +
    ggtitle('Iris data') + theme(legend.position=c(0.6, 0.17)),
  plot_parametric_sensitivity(results_df_2, alpha_0 = 8.0) +
    ggtitle(' ') + theme(legend.position="none"),
  plot_parametric_sensitivity(results_df_3, alpha_0 = 13.0) +
    ggtitle(' ') + theme(legend.position="None"),
  ncol=3)
SavePlot(g, "iris_parametric.png", height=4)


# Iris functional
iris_pert1 <- filter(iris_data$prior_pert_df, filename == "prior_pert1.csv")
iris_pert2 <- filter(iris_data$prior_pert_df, filename == "prior_pert2.csv")
iris_pred_pert1 <- filter(iris_data$results_df, pred, pert=="1")
iris_pred_pert2 <- filter(iris_data$results_df, pred, pert=="2")

g <- grid.arrange(
  plot_prior_perturbation(iris_pert1) +
    theme(legend.position = c(0.8, 0.5)) +
    ggtitle(TeX("\\textbf{Iris data, first $p_1$}")),
  plot_parametric_sensitivity(iris_pred_pert1,
                              xlabel=TeX("$\\delta$"),
                              alpha_0=0.0001) + ggtitle(' '), 
  
  plot_prior_perturbation(iris_pert2) +
    theme(legend.position = c(0.8, 0.5)) +
    ggtitle(TeX("\\textbf{Iris data, second $p_1$}")),
  plot_parametric_sensitivity(iris_pred_pert2,
                              xlabel=TeX("$\\delta$"),
                              alpha_0=0.0001) + ggtitle(' '),
  ncol=2, widths=c(0.45, 0.55))
SavePlot(g, "iris_functional.png")


# Mouse parametric
MouseParametricPlot <- function(processed_results, title="Mouse data") {
  g <- grid.arrange(
    plot_parametric_sensitivity(
      processed_results %>%
        filter(!alpha_increase, !functional),
      alpha_0=2.0) +
      ggtitle(title)  + theme(legend.position=c(0.8, 0.25)),
    plot_parametric_sensitivity(
      processed_results %>%
        filter(alpha_increase, !functional),
      alpha_0=2.0) +
      ggtitle(" ") + theme(legend.position="None"),
    ncol=2
  )
  return(g)
}


gp <- MouseParametricPlot(filter(genomics_data$processed_results, !inflate))
SavePlot(gp, "mouse_parametric_notinflate.png")

gpi <- MouseParametricPlot(
  filter(genomics_data$processed_results, inflate),
  title="Mouse data inflated variance")
SavePlot(gpi, "mouse_parametric_inflate.png")

# Mouse functional
MouseFunctionalPlot <- function(processed_results, title="Mouse data") {
  g <- grid.arrange(
    plot_prior_perturbation(genomics_data$processed_pert_df) +
      theme(legend.position = c(0.25, 0.75)) +
      ggtitle(title),
    
    plot_parametric_sensitivity(
        filter(processed_results, functional),
        alpha_0=0.0001,
        xlabel=TeX("$\\delta$")) +
        ggtitle(' ') + theme(legend.position =  c(0.8, 0.65)),
    ncol=2) #, widths=c(0.45, 0.55))
  return(g)
}

gf <- MouseFunctionalPlot(filter(genomics_data$processed_results, !inflate))
SavePlot(gf, "mouse_functional_notinflate.png")

gfi <- MouseFunctionalPlot(
  filter(genomics_data$processed_results, inflate),
  title="Mouse data inflated variance")
SavePlot(gfi, "mouse_functional_inflate.png")


gpf <- grid.arrange(gp, gf, ncol=1)
SavePlot(gpf, "mouse_notinflated.png", width=10, height=6)

gpfi <- grid.arrange(gpi, gfi, ncol=1)
SavePlot(gpfi, "mouse_inflated.png", width=10, height=7)

# Timing data.  You have to get the Hessian timing with hessian_timing.py.
genomics_data$timing_df %>%
  group_by(n_samples,
           taylor_order, inflate, functional, log_phi_desc) %>%
  summarise(mean_refit_time=mean(refit_time),
            mean_lr_time=mean(lr_time),
            total_refit_time=sum(refit_time),
            total_lr_time=sum(lr_time), n=n())
