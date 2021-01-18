import numpy as np

#################
# Load data
#################
data_dir = '../data/'
data_file = data_dir + 'phased_HGDP+India+Africa_2810SNPs-regions1to36.npz'

data = np.load(data_file)
g_obs = np.array(data['g_obs'], dtype = int)

#################
# Subsample data
#################
n_obs = 25
n_loci = 75

np.random.seed(23323543)
indx_ind = np.random.choice(g_obs.shape[0], n_obs, 
                            replace = False)

indx_loci = np.random.choice(g_obs.shape[1], n_loci, 
                            replace = False)

g_obs_sub = g_obs[indx_ind]
g_obs_sub = g_obs_sub[:, indx_loci]

#################
# save
#################
outfile = '../data/huang2011_sub_nobs{}'.format(n_obs) + '_nloci{}'.format(n_loci)
# outfile = '../data/tmp'

np.savez(outfile, g_obs = g_obs_sub, 
         indx_ind = indx_ind, 
         indx_loci = indx_loci)

print('saved into ', outfile)
