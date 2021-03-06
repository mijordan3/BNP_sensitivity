import numpy as np

import jax.numpy as jnp

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.cm as cm

from sklearn import datasets
from sklearn.decomposition import PCA

import paragami

from copy import deepcopy

def load_iris_data(demean=True):
    iris = datasets.load_iris(return_X_y= True)
    iris_features = iris[0]
    demean = True
    if demean:
        iris_features -= np.mean(iris_features, axis = 0)[None, :]

    iris_species = iris[1]
    return jnp.array(iris_features), iris_species


def plot_clusters(x, y, cluster_labels, colors, fig, centroids = None, cov = None):
    if np.all(cov != None):
        assert len(np.unique(cluster_labels)) == np.shape(cov)[0]
    if np.all(centroids != None):
        assert len(np.unique(cluster_labels)) == np.shape(centroids)[0]

    unique_cluster_labels = np.unique(cluster_labels)
    n_clusters = len(unique_cluster_labels)

    # this would be so much easier if
    # python lists supported logical indexing ...
    cluster_labels_color = [colors[k] for n in range(len(x)) \
                            for k in range(n_clusters) \
                            if cluster_labels[n] == unique_cluster_labels[k]]

    # plot datapoints
    fig.scatter(x, y, c=cluster_labels_color, marker = '.')

    if np.all(centroids != None):
        for k in range(n_clusters):
            fig.scatter(centroids[k, 0], centroids[k, 1], marker = '+', color = 'black')

    if np.all(cov != None):
        for k in range(n_clusters):
            eig, v = np.linalg.eig(cov[k, :, :])
            ell = Ellipse(xy=(centroids[k, 0], centroids[k, 1]),
                  width=np.sqrt(eig[0]) * 6, height=np.sqrt(eig[1]) * 6,
                  angle=np.rad2deg(np.arctan(v[1, 0] / v[0, 0])))
            ell.set_facecolor('none')
            ell.set_edgecolor(colors[k])
            fig.add_artist(ell)


def transform_params_to_pc_space(pca_fit, centroids, cov):
    # PCA fit should be the output of
    # pca_fit = PCA()
    # pca_fit.fit(iris_features)

    # centroids is k_approx x dim
    # infos is k_approx x dim x dim

    assert pca_fit.components_.shape[1] == centroids.shape[1]

    centroids_pc = pca_fit.transform(centroids)

    cov_pc = np.zeros(cov.shape)
    for k in range(cov.shape[0]):
        cov_pc[k, :, :] = np.dot(np.dot(pca_fit.components_, cov[k]), pca_fit.components_.T)

    # cov_pc = np.einsum('di, kij, ej -> kde', pca_fit.components_, cov, pca_fit.components_)

    return centroids_pc, cov_pc


def get_plotting_data(iris_features):
    # Define some things that will be useful for plotting.

    # define colors that will be used for plotting later
    # colors = ['red', 'blue', 'green', 'orange', 'purple', 'yellow', 'cyan', 'magenta']
    # colors += colors

    pca_fit = PCA()
    pca_fit.fit(iris_features)
    pc_features = pca_fit.transform(iris_features)

    cmap = cm.get_cmap(name='gist_rainbow')
    colors1 = [cmap(k * 50) for k in range(12)]
    colors2 = [cmap(k * 25) for k in range(12)]
    return pca_fit, pc_features, colors1, colors2



def get_param_indices(param_str, vb_params_dict, vb_params_paragami):
    bool_dict = deepcopy(vb_params_dict)
    for k in vb_params_dict.keys():
        for j in vb_params_dict[k].keys():
            if j == param_str:
                bool_dict[k][j] = (vb_params_dict[k][j] == vb_params_dict[k][j])
            else:
                bool_dict[k][j] = (vb_params_dict[k][j] != vb_params_dict[k][j])

    return vb_params_paragami.flat_indices(bool_dict, free = True)

