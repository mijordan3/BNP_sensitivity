plot_parametric_sensitivity <- function(
    results_df, alpha_0, xlabel=TeX('$\\alpha$')) {

  results_df %>%
    gather(method, e_num_clusters, -alpha)

  results_df_long <- results_df %>%
    gather(method, e_num_clusters, -alpha)

  plot <-
    ggplot(results_df_long) +
    geom_point(aes(x = alpha, y = e_num_clusters, color = method)) +
    geom_line(aes(x = alpha, y = e_num_clusters, color = method)) +
    xlab(xlabel) + ylab('E[# clusters]') +
    theme(legend.position = c(0.75, 0.2))

  if(alpha_0 > 0){
    # if we actually want a vertical line
    plot <- plot +
        geom_vline(xintercept = alpha_0, color = 'blue', linetype = 'dashed')
  }
  return(plot)
}

# data = subset(results_df_long, method == 'linear approx')
