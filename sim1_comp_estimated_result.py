# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 18:37:17 2020

@author: yokoyama
"""
from IPython import get_ipython
from copy import deepcopy, copy
get_ipython().magic('reset -sf')
#get_ipython().magic('cls')

import os
os.chdir('D:\\GitHub\\ChangeDetectSim\\')
# os.chdir('C:\\Users\\H.yokoyama\\Documents\\Python_Scripts\\ChangeDetectSim')
# os.chdir('D:\\Python_Scripts\\test_myBayesianModel_PRC\\') # Set full path of your corrent derectory

simName      = 'sim1'

current_path = os.getcwd()
fig_save_dir = current_path + '\\figures\\' + simName + '\\'

if os.path.exists(fig_save_dir)==False:  # Make the directory for figures
    os.makedirs(fig_save_dir)

param_path = current_path + '\\save_data\\param_' + simName + '\\' # Set path of directory where the dataset of parameter settings are saved.
    
import matplotlib.pylab as plt
import matplotlib.gridspec as gridspec
plt.rcParams['font.family']      = 'Arial'#
plt.rcParams['mathtext.fontset'] = 'stix' # math font setting
plt.rcParams["font.size"]        = 18 # Font size

#%%
from my_modules.my_dynamical_bayes import *
from my_modules.my_graph_visualization import *
from scipy.stats import zscore
from numpy.random import *
import numpy as np
import glob
import scipy.stats  as stats
import scipy.sparse as sparse


def get_fname(path):
    name     = []
    ext      = []
    for file in os.listdir(path):
        split_str = os.path.splitext(file)
        name.append(split_str[0])
        ext.append(split_str[1])
        
        print(split_str)
        
    return name, ext

def calc_error(K_tr, Kest, Nosc):
    error = np.zeros(Kest.shape[0])
    for t in range(Kest.shape[0]):
        K        = Kest[t,:].reshape(Nosc, Nosc)
        # error[t] = np.sqrt(np.mean((K_tr - K)**2))
        error[t] = np.sqrt(np.mean(abs(K_tr - K)))
    
    return error

def get_rand_error_dist(K_tr, K, Nprm):
    Nosc,_ = K_tr.shape
    rand_error_dist = np.zeros(Nprm)
    for prm in range(Nprm):
        tmp                  = deepcopy(K)
        tmp                  = np.random.permutation(tmp.reshape(-1)).reshape(K.shape)
        K_rnd                = tmp[0,:].reshape(Nosc, Nosc)
        rand_error_dist[prm] = np.sqrt(np.mean(abs(K_tr - K_rnd)))
    
    return rand_error_dist
        
#%%
###### Load the estimation result of each window size
Nprm  = 1000
Ncond = np.array([3, 10, 15, 20])
lgnd  = ['$N_{osc}$ = %2d'%(n) for n in Ncond]
fname = 'estimation_result_Twin_'

epoch_idx = np.array([100, 500, 1000, 2000, 3000, 4000]) 
vmin      = 0.0
vmax      = 1.0
cmaps     = 'Blues'


fig       = plt.figure(constrained_layout = False, figsize=(10, 12));
plt.subplots_adjust(wspace=0.8, hspace=0.8);
gs        = fig.add_gridspec(len(Ncond)+1, len(epoch_idx)+1)
ratios    = [val for val in np.ones(len(Ncond))]
ratios.append(0.08)
gs.set_height_ratios(ratios)

ax_cb     = fig.add_subplot(gs[len(Ncond), 0:len(epoch_idx)+1])
cbar_info = [False, {"orientation":"horizontal"}, ax_cb]

error     = []
error_rnd = np.zeros((Nprm, len(Ncond)))

a_all     = []
b_all     = []
K_all     = []

for i in range(len(Ncond)):
    ################### load data
    path        = param_path + 'Nosc_%02d\\'%(Ncond[i])
    name, ext   = get_fname(path)
    
    fullpath    = path + name[0] + ext[0]
    data_dict   = np.load(fullpath, encoding='ASCII', allow_pickle='True').item()
    
    noise_param = data_dict['noise_param']
    prec_param  = data_dict['prec_param'] 
    T           = data_dict['window_size']
    beta        = data_dict['fourier_coeff']
    omega       = data_dict['omega']
    cp_score    = data_dict['cp_score']
    loglike     = data_dict['loglike']
    dphi_hat    = data_dict['y_hat']
    
    a_tr        = data_dict['a_tr']
    b_tr        = data_dict['b_tr']
    K_tr        = data_dict['K_tr']
    
    a_all.append(a_tr)
    b_all.append(b_tr)
    K_all.append(K_tr)
    
    Nosc        = Ncond[i]
    Time        = np.arange(0, dphi_hat.shape[0]) # sample 
    Kest        = np.sqrt(np.sum(beta**2, axis=2))
    
    tmp_error   = calc_error(K_tr, Kest, Nosc)[:, np.newaxis]
    if i ==0:
        error = tmp_error
    else:
        error  = np.concatenate((error, tmp_error), axis=1)
    
    ############
    ax_tr = fig.add_subplot(gs[i, 0])
    vis_heatmap(K_tr, vmin, vmax, cmaps, ax_tr, np.array(['True', 'osci. $j$', 'osci. $i$']), cbar_info)
    
    
    for j in range(len(epoch_idx)):
        idx       = epoch_idx[j]
        title_str = 'Epoch\n%d'%(idx)
        K         = deepcopy(Kest[idx-2,:]).reshape(Nosc, Nosc)
        
        if (i==len(Ncond)-1) & (j==len(epoch_idx)-1):
            cbar_info = [True, {"orientation":"horizontal"},  ax_cb]
            
        ax = fig.add_subplot(gs[i, j+1])          
        vis_heatmap(K, vmin, vmax, cmaps, ax, np.array([title_str, 'osci. $j$', 'osci. $i$']), cbar_info)
    
    error_rnd[:,i] = get_rand_error_dist(K_tr, Kest, Nprm)
    
    del data_dict
plt.savefig(fig_save_dir + 'comp_est_network.png')
plt.savefig(fig_save_dir + 'comp_est_network.svg')
plt.show()
    ######################
#%%
color_index = ['royalblue', 'tomato', 'forestgreen', 'red', 'darkviolet']
idxCI       = int(Nprm*0.95)
err_confCI  = np.array([np.sort(error_rnd[:,i])[idxCI] for i in range(len(Ncond))])

plt.rcParams['font.family']      = 'Arial'#
plt.rcParams['mathtext.fontset'] = 'stix' # math font setting
plt.rcParams["font.size"]        = 15 # Font size

fig       = plt.figure(constrained_layout = False, figsize=(10, 7));
gs        = fig.add_gridspec(3, len(Ncond))
gs.set_height_ratios([1,1,1.2])

plt.subplots_adjust(wspace=0.8, hspace=1.2);

ax_all    = fig.add_subplot(gs[0:2, 1:3])
for i in range(len(Ncond)):
    ax_all.plot(Time, error[:,i], label = lgnd[i], color=color_index[i])
    ax_all.set_xticks([0.0, 1000, 2000, 3000, 4000])
    ax_all.set_ylim([0, 2.5])
    
    ax = fig.add_subplot(gs[2, i])
    ax.plot(Time, error[:,i], label = lgnd[i], color=color_index[i])
    ax.plot(np.array([Time[0]-50,Time[-1]+50]), err_confCI[i] * np.ones(2), label = '$95 \%$ CI', color='k', linestyle='--')
    ax.set_xlim([Time[0]-50, 1100 + 50])
    ax.set_ylim([0, 2.5])
    ax.set_xticks([0.0, 500, 1000])
    ax.set_xlabel('# sample')
    ax.set_ylabel('MAE (a.u.)')
    ax.set_title(lgnd[i])
    ax.text(  # position text relative to data
        800, err_confCI[i], '$95 \%$ CI',  # x, y, text,
        ha='center', va='bottom',   # text alignment,
        transform=ax.transData      # coordinate system transformation
    )
    
ax_all.set_xlabel('# sample')
ax_all.set_ylabel('mean absolute error (a.u.)')
ax_all.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)


plt.savefig(fig_save_dir + 'comp_est_error.png')
plt.savefig(fig_save_dir + 'comp_est_error.svg')
plt.show()
#%%
cbar_info = [False, {"orientation":"horizontal"},  ax_cb]

fig = plt.figure(figsize=(12, 6))
outer = gridspec.GridSpec(3, 2, wspace=0.3, hspace=0.2, height_ratios=[1,1,0.08])

tmp   = plt.Subplot(fig, outer[4:])
ax_cb = fig.add_subplot(tmp)

for i in range(4):
    if i==3:
        cbar_info = [True, {"orientation":"horizontal"},  ax_cb]
    else:
        cbar_info = [False, {"orientation":"horizontal"},  ax_cb]
        
    inner = gridspec.GridSpecFromSubplotSpec(1, 3,
                    subplot_spec=outer[i], wspace=0.8, hspace=0.8)
    
    a  = a_all[i]
    b  = b_all[i]
    K  = K_all[i]
    
    a_ax = plt.Subplot(fig, inner[0])
    vis_heatmap(a, vmin, vmax, cmaps, a_ax, np.array(['\n $a_{ij}$', 'osci. $j$', 'osci. $i$']), cbar_info, linewidths = 0.01)
    fig.add_subplot(a_ax)
    
    b_ax = plt.Subplot(fig, inner[1])
    vis_heatmap(b, vmin, vmax, cmaps, b_ax, np.array(['$N_{osc}$ = %2d \n $b_{ij}$'%(Ncond[i]), 'osci. $j$', 'osci. $i$']), cbar_info, linewidths = 0.01)
    fig.add_subplot(b_ax)
    
    k_ax = plt.Subplot(fig, inner[2])
    vis_heatmap(K, vmin, vmax, cmaps, k_ax, np.array(['\n $K_{ij}$', 'osci. $j$', 'osci. $i$']), cbar_info, linewidths = 0.01)
    fig.add_subplot(k_ax)
    
    
    
plt.savefig(fig_save_dir + 'param_setting.png')
plt.savefig(fig_save_dir + 'param_setting.svg')
plt.show()
