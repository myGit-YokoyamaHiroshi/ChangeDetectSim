from IPython import get_ipython
from copy import deepcopy, copy
get_ipython().magic('reset -sf')
#get_ipython().magic('cls')

import os
os.chdir('D:\\GitHub\\ChangeDetectSim\\')


sim_name     = 'sim2'
current_path = os.getcwd()
fig_save_dir = current_path + '\\figures\\' +  sim_name + '\\'

if os.path.exists(fig_save_dir)==False:  # Make the directory for figures
    os.makedirs(fig_save_dir)

param_path = current_path + '\\save_data\\param_' +  sim_name + '\\'# Set path of directory where the dataset of parameter settings are saved.
    
import matplotlib.pylab as plt
plt.rcParams['font.family']      = 'Arial'#
plt.rcParams['mathtext.fontset'] = 'stix' # math font setting
plt.rcParams["font.size"]        = 18 # Font size

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
theta          = param_dict['theta']
State          = param_dict['State']
axis           = np.arange(theta.shape[0])
#%% Estimate state changes using LEiDA algorithm
fname = 'LEiDA_results.mat'
if os.path.exists(current_path + '\\LEiDA\\' + fname)==False:
#%%
    matpath = current_path + '\\LEiDA\\mat\\'
    fname   = 'phase_data_sim2.mat'
    
    oc.addpath(current_path + '\\LEiDA\\')
    oc.addpath('C:\Octave\Octave-5.2.0\mingw64\share\octave\packages\statistics-1.4.1')
    oc.LEiDA(matpath, fname)
#%% Visualize the result of LEiDA
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
    
fig = plt.figure(constrained_layout = False, figsize=(8, 12));
plt.subplots_adjust(wspace=0.0, hspace=1);
gs  = fig.add_gridspec(NofClust, 2)

for state in range(0,NofClust):
    ax1 = fig.add_subplot(gs[state, 0])
    
    # if state==NofClust-1:
    #     ax_cb = fig.add_subplot(gs[state+1, 0])
    #     cbar_info = [True, {"orientation":"horizontal"},  ax_cb]
    # else:
    #     cbar_info = [False, {"orientation":"horizontal"},  ax1]
    vis_undirected_graph(np.triu((iFC_clust[:,:,state]).T), -0.7, 0.7)
    # vis_heatmap(iFC_clust[:,:,state], -.6, .6, 'bwr', ax1, np.array(['\n $V_{c}V_{c}^{T}$', 'osci. $j$', 'osci. $i$']), cbar_info, linewidths = 0.001)
    # ax1.set_ylim([-.5, 1.5])
    
    ax2 = fig.add_subplot(gs[state, 1])
    ax2.plot(clust[state,:])
    ax2.set_title('State %d'%(state+1))
    ax2.set_ylabel('time course')
    ax2.set_xlabel('# sample')

plt.savefig(fig_save_dir + 'estimated_state_changes_LEiDA.png')
plt.savefig(fig_save_dir + 'estimated_state_changes_LEiDA.svg')
#%% Visualize exact state changes
exact_clust = np.zeros((State, len(axis)))

exact_clust[0, axis<1000]    = 1
exact_clust[1, (axis>=1000)] = 1


fig = plt.figure(constrained_layout = False, figsize=(8, 12));
plt.subplots_adjust(wspace=0.0, hspace=1);
gs  = fig.add_gridspec(NofClust, 2)

for state in range(0,State-1):
    ax1 = fig.add_subplot(gs[state, 0])
    
    # if state==NofClust-1:
    #     ax_cb = fig.add_subplot(gs[state+1, 0])
    #     cbar_info = [True, {"orientation":"horizontal"},  ax_cb]
    # else:
    #     cbar_info = [False, {"orientation":"horizontal"},  ax1]
    vis_directed_graph((K_tr[:,:,state]).T, 0, 0.7)
    # vis_heatmap(iFC_clust[:,:,state], -.6, .6, 'bwr', ax1, np.array(['\n $V_{c}V_{c}^{T}$', 'osci. $j$', 'osci. $i$']), cbar_info, linewidths = 0.001)
    # ax1.set_ylim([-.5, 1.5])
    
    ax2 = fig.add_subplot(gs[state, 1])
    ax2.plot(exact_clust[state,:])
    ax2.set_title('State %d'%(state+1))
    ax2.set_ylabel('time course')
    ax2.set_xlabel('# sample')

plt.savefig(fig_save_dir + 'exact_state_changes.png')
plt.savefig(fig_save_dir + 'exact_state_changes.svg')
