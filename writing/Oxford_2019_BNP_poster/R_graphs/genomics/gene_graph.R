gene_shapes_melt <- gene_data$gene_shapes_melt

gene_shapes_melt <- 
  melt(gene_shapes, id.vars=c("genes", "inflate", "gene")) %>%
  separate(variable, c("variable", "ind"), "\\.") %>%
  #mutate(ind=paste("d", ind, sep="")) %>%
  mutate(ind=as.numeric(ind), gene=factor(gene)) %>%
  dcast(ind + gene + genes + inflate ~ variable)
head(gene_shapes_melt)

ggplot(filter(gene_shapes_melt, as.integer(gene) <= 6),
       aes(fill=gene, group=gene, color=gene)) +
  geom_ribbon(aes(x=ind,
                  ymin=beta_mean - 1.64 * beta_sd,
                  ymax=beta_mean + 1.64 * beta_sd, group=gene),
              color=NA, alpha=0.4) +
  geom_ribbon(aes(x=ind,
                  ymin=beta_mean - beta_sd,
                  ymax=beta_mean + beta_sd, group=gene),
              color=NA, alpha=0.4) +
  geom_line(aes(x=ind, y=beta_mean), lwd=2) +
  facet_grid(inflate ~ gene)

