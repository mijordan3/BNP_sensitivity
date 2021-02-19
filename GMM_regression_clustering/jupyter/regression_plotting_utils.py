# some helpful functions for plotting

import numpy as np
import matplotlib.pyplot as plt

def PlotPredictionLine(timepoints, pred, n, this_plot):
    this_plot.plot(timepoints, pred[n, :],
                   color='green', linewidth=3)

def PlotRegressionLine(y, timepoints, x, beta, beta_info, n,
                       num_draws=30, ax=None):
    
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(15,8))
    
    beta_mean = beta[n, :]
    beta_cov = np.linalg.inv(beta_info[n, :, :])
    
    ax.plot(timepoints, y[n, :], '+', color = 'blue');
    ax.plot(timepoints, x @ beta_mean, color = 'red');
    ax.set_ylabel('gene expression')
    ax.set_xlabel('time')
    ax.set_title('gene number {}'.format(n))

    # draw from the variational distribution, to plot uncertainties
    for j in range(num_draws):
        beta_draw = np.random.multivariate_normal(beta_mean, beta_cov)
        ax.plot(timepoints, x @ beta_draw,
                color = 'red', alpha = 0.08);
