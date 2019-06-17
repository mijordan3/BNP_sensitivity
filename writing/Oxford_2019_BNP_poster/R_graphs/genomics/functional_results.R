
results <- filter(gene_data$results, functional)

MakePlot <- function(use_preditive, use_inflate, use_alpha_increase, use_threshold=0) {
  ggplot(filter(results,
                predictive==use_predictive,
                inflate==use_inflate,
                threshold == use_threshold)) +
    geom_point(aes(x=epsilon, y=e_num1, color="Refitting")) +
    geom_point(aes(x=epsilon, y=e_num_pred, color="Approximation")) +
    geom_line(aes(x=epsilon, y=e_num1, color="Refitting")) +
    geom_line(aes(x=epsilon, y=e_num_pred, color="Approximation")) +
    geom_vline(aes(xintercept=0.0)) +
    theme(legend.title = element_blank()) +
    ggtitle(sprintf("%s",
                    ifelse(use_inflate, "Noise added", "No noise added"))) +
    xlab(TeX("$\\delta$")) + ylab("Expected number of clusters")
}

legend <- get_legend(MakePlot(use_predictive, FALSE, use_alpha_increase))
nolegend <- theme(legend.position="None")

use_predictive <- TRUE
grid.arrange(
  MakePlot(use_predictive, FALSE, use_alpha_increase) + nolegend,
  MakePlot(use_predictive, TRUE, use_alpha_increase) + nolegend,
  legend,
  widths=c(0.4, 0.4, 0.2),
  ncol=3
)

