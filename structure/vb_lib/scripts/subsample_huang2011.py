import numpy as np

np.random.seed(23323543)

#################
# Load data
#################
fastStructure_dir = './../../../../fastStructure/'
data_file = fastStructure_dir +  \
                'hgdp_data/huang2011_plink_files/' + \
                'phased_HGDP+India+Africa_2810SNPs-regions1to36.npz'

data = np.load(data_file)
g_obs = np.array(data['g_obs'], dtype = int)

#################
# Subsample data
#################
n_obs = 200
n_loci = 500

indx_ind = np.random.choice(g_obs.shape[0], n_obs, 
                            replace = False)

indx_loci = np.random.choice(g_obs.shape[1], n_loci, 
                            replace = False)

g_obs_sub = g_obs[indx_ind]
g_obs_sub = g_obs_sub[:, indx_loci]

#################
# save
#################
outfile = '../data/huang2011_subsampled'

np.savez(outfile, g_obs = g_obs_sub)

print('saved into ', outfile)
