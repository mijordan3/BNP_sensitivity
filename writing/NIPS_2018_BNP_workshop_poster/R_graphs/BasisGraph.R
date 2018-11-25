grid.arrange(
ggplot(data_env$obs_df) +
  geom_point(aes(x=t, y=y)) +
  geom_line(aes(x=t, y=means)) +
  geom_hline(aes(yintercept=b)) +
  xlab("Time t") + ylab("Gene expression Y") +
  ggtitle(sprintf("Data and fit for obs. %d", unique(data_env$obs_df$n)))
,
ggplot() +
  geom_line(data=data_env$x_non_orth_dense_df, aes(x=t, y=value, color=variable), lwd=1) +
  geom_point(data=data_env$x_non_orth_df, aes(x=t, y=value, color=variable), size=3) +
  geom_line(data=data_env$x_non_orth_df, aes(x=t, y=value, color=variable), alpha=0.7) +
  ggtitle("Basis splines with uneven sampling") +
  xlab("Time t") + ylab("X") +
  theme(legend.position="none")
, ncol=2
)
