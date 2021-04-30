out_init <- plot_initial_fit(add_geographic_labels = FALSE)

# box for mbololo outliers
add_box <- mbololo_box 
add_label <- geom_text(aes(x = min(mbololo_outliers$obs_id) - 6, 
                y = 0.2, 
                label = 'A'), 
            size = text_size)

# only plot the mbololo region
n_mbololo <- out_init$intercepts[1]

# trim_plot <- coord_cartesian(xlim = c(0.5, 10 + n_mbololo), 
#                              ylim = c(1, 0), 
#                              expand = FALSE)
trim_plot <- coord_cartesian(xlim = c(0.5, 50),
                             ylim = c(1, 0),
                             expand = FALSE)


# initial fit
p_admix <- out_init$p +
  ggtitle('initial fit') +
  add_box + 
  add_label + 
  theme(title = element_text(size = title_size), 
        legend.position = 'none') + 
  trim_plot 

###################
# results at epsilon = 1.0
###################

plot_admix_here <- function(admix_matr){
  clusters_keep <- 7
  
  out <- plot_structure_fit(admix_matr[, 1:clusters_keep]) 

  return(out$p + 
           add_box + 
           add_label + 
           theme(axis.text.x = element_blank(), 
                 axis.ticks.x = element_blank(), 
                 title = element_text(size = title_size), 
                 legend.position = 'none') +
           trim_plot)
}

p_refit <- plot_admix_here(admix_refit) + 
  ggtitle('refit at t = 1')

p_lr <- plot_admix_here(admix_lr) + 
  ggtitle('lin. at t = 1')

p_admix + p_refit + p_lr

