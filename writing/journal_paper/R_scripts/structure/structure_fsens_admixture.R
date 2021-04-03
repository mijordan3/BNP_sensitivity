out_init <- plot_initial_fit(add_geographic_labels = FALSE)

add_box <- mbololo_box 
add_label <- geom_text(aes(x = min(mbololo_outliers$obs_id) - 6, 
                y = 0.2, 
                label = 'A'), 
            size = text_size)

p_admix <- out_init$p +
  ggtitle('initial fit') +
  add_box + 
  add_label + 
  theme(title = element_text(size = title_size))

###################
# results at epsilon = 0.5
###################

plot_admix_here <- function(admix_matr){
  clusters_keep <- 7
  
  out <- plot_structure_fit(admix_matr[, 1:clusters_keep]) 

  return(out$p + 
           add_box + 
           add_label + 
           theme(axis.text.x = element_blank(), 
                 axis.ticks.x = element_blank(), 
                 title = element_text(size = title_size)))
}

p_refit1 <- plot_admix_here(admix1_refit) + 
  ggtitle('refit at epsilon = 0.5')

p_lr1 <- plot_admix_here(admix1_lr) + 
  ggtitle('lr at epsilon = 0.5')


###################
# results at epsilon = 2
###################
p_refit2 <- plot_admix_here(admix2_refit) + 
  ggtitle('refit at epsilon = 1')

p_lr2 <- plot_admix_here(admix2_lr) + 
  ggtitle('lr at epsilon = 1')

(p_admix + plot_spacer()) / 
  (p_refit1 + p_lr1) /
  (p_refit2 + p_lr2)
