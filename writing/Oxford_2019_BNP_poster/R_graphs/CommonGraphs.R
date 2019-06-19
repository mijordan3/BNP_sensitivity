set_font_sizes <- theme(plot.title=element_text(size=20, face="bold"),
                        axis.text=element_text(size=12),
                        axis.title=element_text(size=18),
                        legend.text=element_text(size=15))


plot_parametric_sensitivity <- function(
    results_df, alpha_0, xlabel=TeX('$\\alpha$')) {

  plot <-
    ggplot(results_df) +
    geom_point(aes(x = alpha, y = e_num_clusters, color = method)) +
    geom_line(aes(x = alpha, y = e_num_clusters, color = method)) +
    xlab(xlabel) + ylab('E[# clusters]') +
    theme(legend.position = c(0.75, 0.2)) +
    set_font_sizes

  if(alpha_0 > 0){
    # if we actually want a vertical line
    plot <- plot +
        geom_vline(xintercept = alpha_0, color = 'blue', linetype = 'dashed')
  }
  return(plot)
}

plot_prior_perturbation <- function(prior_pert_df) {
    plot <- ggplot(prior_pert_df) +
        geom_line(aes(x = nu_k, y = p, color = which_prior)) +
        theme(legend.title=element_blank()) +
        xlab(TeX("$\\nu_k$")) + ylab(TeX("$p(\\nu_k)$")) +
        scale_color_manual(values=c("red", "blue")) +
        set_font_sizes
    return(plot)
}
