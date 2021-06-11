from IPython import get_ipython
from copy import deepcopy, copy
get_ipython().magic('reset -sf')
#get_ipython().magic('cls')

import os
os.chdir('D:\\GitHub\\ChangeDetectSim_v2\\')


sim_name     = 'sim2'
current_path = os.getcwd()
fig_save_dir = current_path + '\\figures\\' +  sim_name + '\\'

if os.path.exists(fig_save_dir)==False:  # Make the directory for figures
    os.makedirs(fig_save_dir)

param_path = current_path + '\\save_data\\param_' +  sim_name + '\\'# Set path of directory where the dataset of parameter settings are saved.
    
import matplotlib.pylab as plt
plt.rcParams['font.family']      = 'Arial'#
plt.rcParams['mathtext.fontset'] = 'stix' # math font setting
plt.rcParams["font.size"]        = 28 # Font size

#%%
# from my_modules.my_dynamical_bayes import *
from my_modules.my_graph_visualization import *
from scipy.io import loadmat
from numpy.random import *
from mne.stats import fdr_correction as fdr
import numpy as np
import glob
#%% make surrogate phase
from oct2py import octave as oc
#%%
name     = []
ext      = []
for file in os.listdir(param_path):
    split_str = os.path.splitext(file)
    name.append(split_str[0])
    ext.append(split_str[1])
    
    print(split_str)

fullpath       = param_path + name[0] + ext[0]
param_dict     = np.load(fullpath, encoding='ASCII', allow_pickle='True').item()    
K_tr           = param_dict['K_tr']
State          = param_dict['State']
axis           = np.arange(param_dict['Nt'])
#%% Visualize the result of LEiDA
fname = 'LEiDA_results.mat'
matdict = loadmat(current_path + '\\LEiDA\\' + fname)
Eig          = matdict['Leading_Eig']
clust_labels = matdict['clust_labels']
NofClust     = np.max(clust_labels)
iFCall       = matdict['iFC_all']
FCpattern    = matdict['FCpattern']

clust        = np.zeros((NofClust, len(clust_labels)))
iFC_clust    = np.zeros((iFCall.shape[0], iFCall.shape[0], NofClust))

for c in range(NofClust):
    idx_c = np.where(clust_labels==(c+1))[0]
    clust[c,idx_c]   = 1
    
    iFC_clust[:,:,c] = FCpattern[c,:].reshape(iFCall.shape[0],1) @ FCpattern[c,:].reshape(1,iFCall.shape[0])
    iFC_clust[:,:,c] = iFC_clust[:,:,c] - np.diag(np.diag(iFC_clust[:,:,c]))
    
# fig = plt.figure(constrained_layout = False, figsize=(8, 12));
# plt.subplots_adjust(wspace=0.0, hspace=1);
# gs  = fig.add_gridspec(NofClust, 2)

# for state in range(0,NofClust):
#     ax1 = fig.add_subplot(gs[state, 0])

#     vis_undirected_graph(np.triu((iFC_clust[:,:,state]).T), -0.7, 0.7)
    
#     ax2 = fig.add_subplot(gs[state, 1])
#     ax2.plot(clust[state,:])
#     ax2.set_title('State %d'%(state+1))
#     ax2.set_ylabel('time course')
#     ax2.set_xlabel('# sample')

# plt.savefig(fig_save_dir + 'estimated_state_changes_LEiDA.png')
# plt.savefig(fig_save_dir + 'estimated_state_changes_LEiDA.svg')

tmp          = clust_labels[1:] - clust_labels[0:-1]
cp_idx       = np.where(tmp!=0)[0]+1
cp_idx       = np.hstack((0, cp_idx, len(clust_labels)))
segment      = np.array([cp_idx[0:-1],cp_idx[1:]]).T        
#%%
color = ['b','c','m'] 

fig = plt.figure(figsize = (20, 10))
gs = fig.add_gridspec(2, 4, height_ratios=[1, 0.3])
ax2 = fig.add_subplot(gs[1,:])
ax2.set_xlim(0, len(clust_labels))
ax2.set_ylim(0, 1)

for state in range(0,NofClust):
    ax1 = fig.add_subplot(gs[0,state])
    
    vis_undirected_graph(np.triu((iFC_clust[:,:,state]).T), -0.6, 0.6)
    ax1.set_ylim(-0.8, 1.3)
    ax1.plot([0, 1], [-0.5,-0.5], linewidth = 10, color=color[state])
    ax1.set_title('State ' + str(state+1), fontsize=28)


l1 = l2 = l3 = []    

for sgm in range(0, segment.shape[0]):
    x    = segment[sgm, :]
    clst = int(clust_labels[x[0]])
    
    if clst ==1:
        l1 = ax2.axvspan(x[0], x[1], color = color[clst-1], label='State' + str(clst))
    elif clst==2:
        l2 = ax2.axvspan(x[0], x[1], color = color[clst-1], label='State' + str(clst))
    elif clst==3:
        l3 = ax2.axvspan(x[0], x[1], color = color[clst-1], label='State' + str(clst))
    
    ax2.set_yticks([])
    ax2.set_xlabel('# sample')


ax2.legend(handles=[l1,l2,l3], bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
plt.savefig(fig_save_dir + 'estimated_state_changes_LEiDA.png', bbox_inches="tight")
plt.savefig(fig_save_dir + 'estimated_state_changes_LEiDA.svg', bbox_inches="tight")
plt.savefig(fig_save_dir + 'estimated_state_changes_LEiDA.eps', bbox_inches="tight")

plt.show()