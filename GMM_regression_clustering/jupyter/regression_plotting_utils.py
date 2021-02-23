# some helpful functions for plotting

import numpy as np
import matplotlib.pyplot as plt

def plot_prediction_line(timepoints, x, beta, ax, 
                       color = 'green',
                       linewidth = 3, 
                       alpha = 1.0):
    
    # beta are coefficients for a single gene
    
    assert len(beta.shape) == 1
    assert len(beta) == x.shape[1]
    assert x.shape[0] == len(timepoints)
    
    ax.plot(timepoints,
            x @ beta,
            color=color,
            linewidth=linewidth, 
            alpha = alpha)

def plot_data(timepoints, y, ax, demean = True):
    
    # y are observations from a single gene
    
    assert len(y.shape) == 1
    assert len(y) == len(timepoints)
    
    if demean: 
        _y = y - y.mean()
    else: 
        _y = y
        
    ax.scatter(timepoints, _y, marker = '+', color = 'blue')

def plot_prediction_line_and_sample(timepoints, x, beta, beta_info,
                                     ax,
                                     num_draws=30, 
                                     alpha1 = 1.0,
                                     alpha2 = 0.15):
    
    assert len(beta.shape) == 1
    assert len(beta) == x.shape[1]
    assert x.shape[0] == len(timepoints)
    assert len(beta_info.shape) == 2
    assert beta_info.shape[0] == len(beta)
    assert beta_info.shape[1] == len(beta)
    
    beta_cov = np.linalg.inv(beta_info)

    # draw from the variational distribution, to plot uncertainties
    for j in range(num_draws):
        beta_draw = np.random.multivariate_normal(beta, beta_cov)
        plot_prediction_line(timepoints, x, beta_draw, 
                             ax, 
                             color = 'red',
                             linewidth = 1, 
                             alpha = alpha2)
    
    # plot means
    plot_prediction_line(timepoints, x, beta, ax, 
                         color = 'grey',
                         linewidth = 5, 
                         alpha = alpha1)