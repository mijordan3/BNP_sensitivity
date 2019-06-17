pert_df <-
  gene_data$pert_df

colors <- c(TeX("$p_0(\\nu_k)$"), TeX("$p_1(\\nu_k)$"))
names(colors) <- c("p0", "p1")

ggplot(pert_df) +
  geom_line(aes(x=v_grid, y=exp(log_p0), color="p0"), lwd=2) +
  geom_line(aes(x=v_grid, y=exp(log_p1), color="p1"), lwd=2) +
  scale_color_discrete(labels=colors, breaks=names(colors)) +
  theme(legend.title=element_blank()) +
  xlab(TeX("$\\nu_k$")) + ylab(TeX("$p(\\nu_k)$"))
