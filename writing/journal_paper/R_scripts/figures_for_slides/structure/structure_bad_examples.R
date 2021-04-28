source('../structure/structure_fsens_admixture.R',
       print.eval = FALSE)
p_admix + 
  (p_refit + theme(legend.position = 'bottom')) + 
  p_lr

save_last_fig('bad_admix_example.png', 
              aspect_ratio = 0.4)

# we made these plots for the journal paper
source('../structure/mbololo_bad_approximation.R',
       print.eval = FALSE)

# the plot of the admixture
(p1 + theme(legend.position = 'top')) / plot_spacer()
save_last_fig('bad_admix_example_trace0.png', 
              aspect_ratio = 0.7)


# the plot of the admixture and the sticks
(p1 + theme(legend.position = 'top', 
            axis.title.x = element_blank(), 
            axis.text.x = element_blank())) / 
  (p0 + get_fontsizes() + theme(legend.position = 'none')) 

save_last_fig('bad_admix_example_trace1.png', 
              aspect_ratio = 0.7)
