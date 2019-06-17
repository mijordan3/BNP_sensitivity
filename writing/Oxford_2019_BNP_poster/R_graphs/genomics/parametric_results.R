
results <- filter(gene_data$results, !functional)

use_predictive <- FALSE
use_inflate <- FALSE

results$functional

MakePlot <- function(use_preditive, use_inflate, use_alpha_increase, use_threshold=0) {
  ggplot(filter(results,
                predictive==use_predictive,
                inflate==use_inflate,
                alpha_increase==use_alpha_increase,
                threshold == use_threshold)) +
    geom_point(aes(x=alpha1, y=e_num1, color="Refitting")) +
    geom_point(aes(x=alpha1, y=e_num_pred, color="Approximation")) +
    geom_line(aes(x=alpha1, y=e_num1, color="Refitting")) +
    geom_line(aes(x=alpha1, y=e_num_pred, color="Approximation")) +
    geom_vline(aes(xintercept=alpha0)) +
    facet_grid(threshold ~ ., scales="free", labeller = label_context) +
    ggtitle(sprintf("%s\n%s",
                    ifelse(use_inflate, "Noise added", "No noise added"),
                    ifelse(use_predictive, "Predictive", "In-sample"))) +
    xlab("Alpha") + ylab("Expected number of clusters")
  
}

use_predictive <- TRUE
use_alpha_increase <- FALSE
grid.arrange(
  MakePlot(use_predictive, FALSE, use_alpha_increase),
  MakePlot(use_predictive, TRUE, use_alpha_increase), ncol=2
)


use_alpha_increase <- TRUE
grid.arrange(
  MakePlot(use_predictive, FALSE, use_alpha_increase),
  MakePlot(use_predictive, TRUE, use_alpha_increase), ncol=2
)
