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


# Iris distribution
g <- ggplot(iris_data$iris_df) + geom_point(aes(x=PC1, y=PC2, color=Species)) + theme(legend.position="none")
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
    filter(results_df_1, method=="refitted"), alpha_0 = 3.0) +
    scale_color_discrete(limits=c("approx", "refitted")) +
    ggtitle('Iris data') + theme(legend.position=c(0.6, 0.17)),
  plot_parametric_sensitivity(results_df_2, alpha_0 = 8.0) +
    ggtitle(' ') + theme(legend.position="none"),
  plot_parametric_sensitivity(results_df_3, alpha_0 = 13.0) +
    ggtitle(' ') + theme(legend.position="None"),
  ncol=3)
SavePlot(g, "iris_parametric_refitonly.png")


g <- grid.arrange(
  plot_parametric_sensitivity(results_df_1, alpha_0 = 3.0) +
    ggtitle('Iris data') + theme(legend.position=c(0.6, 0.17)),
  plot_parametric_sensitivity(results_df_2, alpha_0 = 8.0) +
    ggtitle(' ') + theme(legend.position="none"),
  plot_parametric_sensitivity(results_df_3, alpha_0 = 13.0) +
    ggtitle(' ') + theme(legend.position="None"),
  ncol=3)
SavePlot(g, "iris_parametric.png")


# Mouse parametric
MouseParametricPlot <- function(processed_results) {
  g <- grid.arrange(
    plot_parametric_sensitivity(
      processed_results %>%
        filter(!alpha_increase, !functional),
      alpha_0=2.0) +
      ggtitle("Mouse data")  + theme(legend.position="none"),
    plot_parametric_sensitivity(
      processed_results %>%
        filter(alpha_increase, !functional),
      alpha_0=2.0) +
      ggtitle(" ") + theme(legend.position=c(0.75, 0.65)),
    ncol=2
  )
  return(g)
}

SavePlot(MouseParametricPlot(filter(genomics_data$processed_results, !inflate)),
         "mouse_parametric_notinflate.png")

SavePlot(MouseParametricPlot(filter(genomics_data$processed_results, inflate)),
         "mouse_parametric_inflate.png")


iris_pert1 <- filter(iris_data$prior_pert_df, filename == "prior_pert1.csv")
iris_pert2 <- filter(iris_data$prior_pert_df, filename == "prior_pert2.csv")
iris_pred_pert1 <- filter(iris_data$results_df, pred, pert=="1")
iris_pred_pert2 <- filter(iris_data$results_df, pred, pert=="2")
gene_pred_pert <- filter(genomics_data$processed_results,
                         functional, !inflate)

grid.arrange(
  # First row --- include titles.
  plot_prior_perturbation(iris_pert1) +
    theme(legend.position = c(0.8, 0.5)) +
    ggtitle(TeX("\\textbf{Iris data, first $p_1$}")),
  plot_parametric_sensitivity(iris_pred_pert1, xlabel=TeX("$\\delta$")) + ggtitle(' '), 
  
  # Second row 
  plot_prior_perturbation(iris_pert2) +
    theme(legend.position = c(0.8, 0.5)) +
    ggtitle(TeX("\\textbf{Iris data, second $p_1$}")),
  plot_parametric_sensitivity(iris_pred_pert2, xlabel=TeX("$\\delta$")) + ggtitle(' '),
  ncol=2)


grid.arrange(
  # First row --- include titles.
  plot_prior_perturbation(iris_pert1) +
    theme(legend.position = "None") +
    ggtitle(TeX("\\textbf{Iris data, first $p_1$}")),
  
  plot_prior_perturbation(iris_pert2) +
    theme(legend.position = "None") +
    ggtitle(TeX("\\textbf{Iris data, second $p_1$}")),
  
  plot_prior_perturbation(genomics_data$processed_pert_df) +
    theme(legend.position = c(0.45, 0.80)) +
    ggtitle("Mouse data"),
  
  # Second row --- results
  plot_parametric_sensitivity(iris_pred_pert1, xlabel=TeX("$\\delta$")) +
    theme(legend.position = "None"),
  
  plot_parametric_sensitivity(iris_pred_pert2, xlabel=TeX("$\\delta$")) +
    theme(legend.position =  "None"),
  
  plot_parametric_sensitivity(gene_pred_pert, xlabel=TeX("$\\delta$")) +
    theme(legend.position =  c(0.7, 0.8)) +
    geom_text(aes(x=0.25, y=59.676, label="(lines are\n overplotted)"),
              hjust="left"),
  ncol=3)




x_grid <- seq(0, 1, length.out=100)
p1 <- dbeta(x_grid, shape1=1.0, shape2=2.0)
p2 <- dbeta(x_grid, shape1=1.0, shape2=10.0)


ggplot(data.frame(x=x_grid, p1=p1, p2=p2)) +
  geom_line(aes(x=x, y=p1, color="1")) +
  geom_line(aes(x=x, y=p2, color="2"))
