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

def plot_top_clusters(e_ind_admix, axarr, n_top_clusters = 5): 
    # plot only the top clusters    
    
    top_n = min(n_top_clusters, e_ind_admix.shape[1])
    
    # find top clusters
    top_clusters_indx = np.argsort(- e_ind_admix.sum(0))
    top_clusters_indx = np.sort(top_clusters_indx[0:top_n])
    e_ind_admix = e_ind_admix[:, top_clusters_indx]
    
    # append remaining probability
    remaining_probs = 1 - e_ind_admix.sum(1, keepdims = True)
    e_ind_admix = np.hstack((e_ind_admix, remaining_probs))
    
    # get colors: last color is grey
    colors = [colorsys.hsv_to_rgb(h,0.9,0.7) for h in np.linspace(0,1,top_n+1)[:-1]]
    colors += ['grey']
    
    # plot
    plot_admixture(e_ind_admix, axarr, colors)
    
    return e_ind_admix, top_clusters_indx

    
def draw_region_separation(population_vec, axarr): 

    # draw lines between different groups in plotting

    unique_groups = np.sort(np.unique(population_vec))
    
    xint = 0.
    xticks = []
    for i in range(len(unique_groups)): 
        incr = (population_vec == unique_groups[i]).mean()

        xticks.append(xint + incr*0.5)
        xint += incr
        axarr.axvline(xint, color = 'white', linewidth = 2)

    axarr.set_xticks(xticks)
    axarr.set_xticklabels(unique_groups,
                          rotation=45, ha='left', 
                          fontsize = 12);

    axarr.xaxis.tick_top()

    
def get_vb_expectations(vb_params_dict, gh_loc = None, gh_weights = None): 
    
    use_logitnormal_sticks = 'stick_means' in vb_params_dict['ind_admix_params'].keys()
    
    if use_logitnormal_sticks: 
        e_ind_admix = cluster_quantities_lib.get_e_cluster_probabilities(
                            vb_params_dict['ind_admix_params']['stick_means'], 
                            vb_params_dict['ind_admix_params']['stick_infos'],
                            gh_loc, gh_weights)

    else: 
        ind_mix_stick_beta_params = vb_params_dict['ind_admix_params']['stick_beta']
        e_stick_lengths = \
                modeling_lib.get_e_beta(ind_mix_stick_beta_params)
        e_ind_admix = cluster_quantities_lib.get_mixture_weights_from_stick_break_propns(e_stick_lengths)

    e_pop_freq = modeling_lib.get_e_beta(vb_params_dict['pop_freq_beta_params'])
    
    return e_ind_admix, e_pop_freq