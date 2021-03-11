import numpy as np

import matplotlib.pyplot as plt
import colorsys

from bnpmodeling_runjingdev import modeling_lib, cluster_quantities_lib

def plot_admixture(admixture, ax, colors = None):
    # copied over form distruct.py file in faststructure
    # adapted for python 3
    
    N,K = admixture.shape
    if colors is None: 
        colors = [colorsys.hsv_to_rgb(h,0.9,0.7) for h in np.linspace(0,1,K+1)[:-1]]
    else:
        assert len(colors) == K
    
    text_color = 'k'
    bg_color = 'w'
    fontsize = 12

    indiv_width = 1./N

    for k in range(K):
        if k:
            bottoms = admixture[:,:k].sum(1)
        else:
            bottoms = np.zeros((N,),dtype=float)

        lefts = np.arange(N)*indiv_width
        ax.bar(lefts, admixture[:,k], width=indiv_width, bottom=bottoms, 
                    facecolor=colors[k], edgecolor=colors[k], linewidth=0.4)
    ax.set_ylim((0, 1))
    ax.set_xlim((- 0.5 * indiv_width, 1 - 0.5 * indiv_width))
    ax.set_xticks([])
    ax.set_yticks([])
    
    return colors

def plot_top_clusters(e_ind_admix, axarr,
                      n_top_clusters = 5, 
                      cmap_colors = plt.get_cmap('Set2').colors): 
    # plot only the top clusters    
    
    top_n = min(n_top_clusters, e_ind_admix.shape[1])
    assert top_n <= len(cmap_colors)
    
    # find top clusters
    top_clusters_indx = np.argsort(- e_ind_admix.sum(0))
    top_clusters_indx = np.sort(top_clusters_indx[0:top_n])
    e_ind_admix = e_ind_admix[:, top_clusters_indx]
    
    # append remaining probability
    remaining_probs = 1 - e_ind_admix.sum(1, keepdims = True)
    e_ind_admix = np.hstack((e_ind_admix, remaining_probs))
    
    # get colors: last color is grey
    # colors = [colorsys.hsv_to_rgb(h,0.9,0.7) for h in np.linspace(0,1,top_n+1)[:-1]]
    colors = [color for color in cmap_colors[0:top_n]]
    colors += ['grey']
    
    # plot
    plot_admixture(e_ind_admix, axarr, colors)
    
    return e_ind_admix, top_clusters_indx

def get_unique_labels(labels): 
    # np.unique sorts things alphabetically, 
    # we don't want that
    unique_labels = [labels[0]]
    
    for i in range(1, len(labels)): 
        if labels[i] != labels[i-1]: 
            unique_labels.append(labels[i])
            
    return unique_labels
    
def draw_region_separation(labels, axarr): 

    # draw lines between different groups in plotting
    
    # get unique labels
    unique_labels = get_unique_labels(labels)
            
    xint = 0.
    xticks = []
    for i in range(len(unique_labels)): 
        incr = (labels == unique_labels[i]).mean()

        xticks.append(xint + incr*0.5)
        xint += incr
        axarr.axvline(xint,
                      linestyle = ':', 
                      color = 'black',
                      linewidth = 2)

    axarr.set_xticks(xticks)
    axarr.set_xticklabels(unique_labels,
                          rotation=45, ha='left', 
                          fontsize = 12);

    axarr.xaxis.tick_top()

def draw_regions_on_coclust(labels, ax, draw_lines = True): 
    
    unique_labels = get_unique_labels(labels)
    
    xint = 0.
    xticks = []
    
    for i in range(len(unique_labels)): 
        incr = (labels == unique_labels[i]).sum()

        xticks.append(xint + incr * 0.5)
    
        xint += incr
        
        if draw_lines:
            ax.axvline(xint, color = 'grey', linestyle = ':', linewidth = 2)
            ax.axhline(xint, color = 'grey', linestyle = ':', linewidth = 2)
    
    ax.set_xticks(xticks)
    ax.set_xticklabels([x for x in unique_labels],
                       rotation=45, ha='left', 
                       fontsize = 12);
    ax.set_yticks(xticks)
    ax.set_yticklabels([x for x in unique_labels],
                       rotation=45, ha='right', 
                       fontsize = 12);

    ax.set_xlim(0, len(labels))
    ax.set_ylim(len(labels), 0)

