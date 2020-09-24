# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 19:57:54 2020

@author: yokoyama
"""


# simulating oscillation by Kuramoto model
# 3 unit

from IPython import get_ipython
from copy import deepcopy, copy
get_ipython().magic('reset -sf')
#get_ipython().magic('cls')

import os
os.chdir('D:\\GitHub\\ChangeDetectSim\\')

current_path = os.getcwd()
fig_save_dir = current_path + '\\figures\\sim2\\'

if os.path.exists(fig_save_dir)==False:
    os.makedirs(fig_save_dir)

param_path = current_path + '\\save_data\\param_sim2\\'
if os.path.exists(param_path)==False:
    os.makedirs(param_path)
    
import matplotlib.pylab as plt
plt.rcParams['font.family']      = 'Arial'#"IPAexGothic"
plt.rcParams['mathtext.fontset'] = 'stix' # math fontの設定
plt.rcParams["font.size"]        = 26 # 全体のフォントサイズが変更されます。

import networkx as nx
#%%
from my_modules.my_dynamical_bayes import *
from my_modules.my_graph_visualization import *
from scipy.stats import zscore
from numpy.random import *
import numpy as np
import glob
#%%
def random_adjacency_matrix(n):   
    matrix = [[np.random.normal(loc=0.45, scale=.1, size=n) for i in range(n)] for j in range(n)]
    matrix = np.array(matrix)[:,:,0]
    # No vertex connects to itself
    for i in range(n):
        matrix[i, i] = 0

    # If i is connected to j, j is connected to i
    for i in range(n):
        for j in range(n):
            matrix[j, i] = matrix[i, j]

    return matrix
#%%
name     = []
ext      = []
for file in os.listdir(param_path):
    split_str = os.path.splitext(file)
    name.append(split_str[0])
    ext.append(split_str[1])
    
    print(split_str)
#%%
fullpath      = param_path + name[0] + ext[0]
param_dict    = np.load(fullpath, encoding='ASCII', allow_pickle='True').item()
Nosc          = param_dict['Nosc']
time          = param_dict['time']
h             = param_dict['h']
Nt            = param_dict['Nt']
State         = param_dict['State']
omega         = param_dict['omega']
K1_tr         = param_dict['K1_tr']
K2_tr         = param_dict['K2_tr']
K_tr          = param_dict['K_tr']

theta_init    = param_dict['theta_init']
del param_dict
#%%

dtheta        = np.zeros((Nt, Nosc))
theta         = np.zeros((Nt, Nosc))
theta[0, :]   = theta_init
noise_scale   = 0.1

phase_dynamics       = np.zeros((Nt, Nosc))
phase_dynamics[0, :] = func_oscillator_approx_fourier_series(theta[0, :], K1_tr[:,:,0], K2_tr[:,:,0], omega, noise_scale)
for t in range(1, Nt):
    if t < int(Nt/3):
        Nst = 0
        noise_scale = 0.1
    elif int(Nt/3) <= t < int(Nt*2/3):
        Nst = 1
        noise_scale = 0.1
    else:
        Nst = 2
        noise_scale = 1
    
    K1 = K1_tr[:,:,Nst]
    K2 = K2_tr[:,:,Nst]
    
    theta_now  = theta[t-1, :]
    theta_next = runge_kutta_oscillator_approx_fourier_series(h, func_oscillator_approx_fourier_series, theta_now, K1, K2, omega, noise_scale)
    
    theta[t, :]          = theta_next.reshape(1, Nosc)
    phase_dynamics[t, :] = func_oscillator_approx_fourier_series(theta[t, :], K1, K2, omega, noise_scale)

    for i in range(Nosc):
        theta_unwrap = np.unwrap(deepcopy(theta[t-1:t+1, i]))
        
        dtheta[t, i] = (theta_unwrap[1] - theta_unwrap[0])/h
#%% plot phase
        
# fig_save_dir = current_path + '\\figures\\Bayesian_kuramoto_weak\\'

axis=np.arange(theta.shape[0])

fig = plt.figure(figsize=(20, 8))
gs  = fig.add_gridspec(Nosc, 1)
plt.subplots_adjust(wspace=0.4, hspace=0.0)

for n in range(Nosc):
    #####################
    ax11 = fig.add_subplot(gs[n, 0])
    plt.plot(axis, theta[:, n])
    
    plt.ylabel('$\\phi_{%d}$'%(n+1))
    plt.xticks(np.arange(0, Nt+1, int(Nt/3))) 
    plt.yticks([0, np.pi, 2*np.pi], labels=['0', '$\\pi$', '$2 \\pi$'])
    plt.grid()
    
plt.xlabel('# sample', fontsize = 18)

plt.show()
#%% plot phase dynamics
fig = plt.figure(figsize=(20, 4))
plt.plot(axis, phase_dynamics)
plt.title('simulated phase dynamics $(\\dfrac{d \\theta_{ij}}{dt})$')
plt.xticks(np.arange(0, Nt+1, int(Nt/3))) 
plt.xlabel('# sample', fontsize = 18)
plt.ylabel('$\\dfrac{d \\theta_{ij}}{dt} = \\omega_{i} + \\Gamma_{ij}$', fontsize = 18)
plt.grid()
plt.legend(bbox_to_anchor=(1.05, 1), labels = ['node 1', 'node 2', 'node 3'], loc='upper left', borderaxespad=0, fontsize=12)
plt.subplots_adjust(right = 0.7)
plt.savefig(fig_save_dir + 'phase_dynamics.png')
plt.savefig(fig_save_dir + 'phase_dynamics.svg')
plt.show()
#%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
P = 1
T = 1
x = deepcopy(theta)
noise_param = 1E-3 # covariance of process noise
# noise_param = 1E-6 # covariance of process noise
prec_param  = 1/noise_param # precision parameter, cov(process noise) = 1/prec_param
#%%
cnt = 1
beta, OMEGA, Changes, L, y_hat, sigma0, Kb0 = est_dynamical_oscillator_1st_order_fourier(x, P, T, h, prec_param)

if len(OMEGA.shape)==3:
    OMEGA = OMEGA[:,:,0]

Time = np.arange(0, Nt-T)
Kest = np.sqrt(np.sum(beta**2, axis=2))

PRC, phi_dlt_plt  = reconstruct_phase_response_curve(beta, OMEGA, Nosc)
#%%
fig = plt.figure(figsize=(20, 4))

plt.plot(Time, OMEGA)
plt.xlabel('# sample', fontsize = 18)
plt.ylabel('natural frequency (a.u.)', fontsize = 18)
plt.legend(bbox_to_anchor=(1.05, 1), labels = ['node 1', 'node 2', 'node 3'], loc='upper left', borderaxespad=0, fontsize=12)
plt.xticks(np.arange(0, Nt+1, int(Nt/3))) 
# plt.ylim(0.0, 4.0)
plt.grid()
plt.subplots_adjust(right = 0.7)
plt.savefig(fig_save_dir + 'natural_freqs.png')
plt.savefig(fig_save_dir + 'natural_freqs.svg')
plt.show()
#%%
C   = Changes[np.isnan(Changes)==False]
Cth = C.mean() + 3 * C.std()
idx = np.where(Changes > Cth)[0]

fig = plt.figure(figsize=(20, 4))

plt.plot(Time[np.isnan(Changes)==False], Changes[np.isnan(Changes)==False]);
plt.scatter(Time[idx], Changes[idx], marker = 'o', c = 'red', label = '> mean + 3SD');
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0, fontsize=12, frameon=False)
plt.xlabel('# sample', fontsize = 18)
plt.ylabel('Changing ratio (a.u.)', fontsize = 18)
plt.xticks(np.arange(0, Nt+1, int(Nt/3))) 

plt.subplots_adjust(right = 0.7)
plt.grid()
# plt.ylim(-0.0, 10.0)
plt.savefig(fig_save_dir + 'changing_point.png')
plt.savefig(fig_save_dir + 'changing_point.svg')
plt.show()
#%%
fig = plt.figure(figsize=(20, 4))

plt.plot(Time[np.isnan(L)==False], L[np.isnan(L)==False]);
plt.xlabel('# sample', fontsize = 18)
plt.ylabel('Log-likelihood (a.u.)', fontsize = 18)
plt.xticks(np.arange(0, Nt+1, int(Nt/3))) 

plt.subplots_adjust(right = 0.7)
plt.grid()
plt.savefig(fig_save_dir + 'loglikelihood.png')
plt.savefig(fig_save_dir + 'loglikelihood.svg')
plt.show()
#%%
tmp_x = deepcopy(x[:-T, :])
x_hat = np.zeros((Nt-T, Nosc))

x_hat[0, :] = deepcopy(tmp_x[0, :])

#%% plot phase dynamics
fig = plt.figure(figsize=(20, 4))

line1 = plt.plot(Time, y_hat, c = 'k', linestyle = '-', zorder = 1, label = 'pred')
line2 = plt.plot(axis, phase_dynamics, c = np.array([0.5, 0.5, 0.5]), linewidth = 4,zorder = 0, label = 'true')
plt.xticks(np.arange(0, Nt+1, int(Nt/3))) 
plt.xlabel('# sample', fontsize = 18)
plt.ylabel('$\\dfrac{d \\theta_{ij}}{dt} = \\omega_{i} + \\Gamma_{ij}$', fontsize = 18)
plt.grid()

handle = [line1[-1], line2[-1]]
labels = ['pred.', 'true']
plt.legend(handle, labels, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0, fontsize=12)
plt.subplots_adjust(right = 0.7)
plt.savefig(fig_save_dir + 'phase_dynamics_est.png')
plt.savefig(fig_save_dir + 'phase_dynamics_est.svg')
plt.show()
#%%
idx_st1 = Time<(int(Nt/3))
idx_st2 = (Time>=int(Nt/3)) & (Time<int(Nt*2/3))
idx_st3 = (Time>=int(Nt*2/3))
phi     = theta[0:-1,:]
dphi    = dtheta[1:,:]
vmin    = 0
vmax    = 1#Kest.mean() + Kest.std()

Kest_ave1 = np.median(Kest[idx_st1,:], axis=0).reshape(Nosc, Nosc)
Kest_ave2 = np.median(Kest[idx_st2,:], axis=0).reshape(Nosc, Nosc)
Kest_ave3 = np.median(Kest[idx_st2,:], axis=0).reshape(Nosc, Nosc)

Kest_ave  = np.concatenate((Kest_ave1[:,:,np.newaxis], 
                            Kest_ave2[:,:,np.newaxis],
                            Kest_ave3[:,:,np.newaxis]), axis=2)
#%%
plot_PRC(phi[idx_st1,:], dphi[idx_st1,:], phi_dlt_plt, PRC[idx_st1,:,:], Kest_ave1, vmin, vmax, Nosc)
plt.savefig(fig_save_dir + 'PRC_state1.png')
plt.savefig(fig_save_dir + 'PRC_state1.svg')
plt.show()

plot_PRC(phi[idx_st2,:], dphi[idx_st2,:], phi_dlt_plt, PRC[idx_st2,:,:], Kest_ave2, vmin, vmax, Nosc)
plt.savefig(fig_save_dir + 'PRC_state2.png')
plt.savefig(fig_save_dir + 'PRC_state2.svg')
plt.show()

plot_PRC(phi[idx_st3,:], dphi[idx_st3,:], phi_dlt_plt, PRC[idx_st3,:,:], Kest_ave3, vmin, vmax, Nosc)
plt.savefig(fig_save_dir + 'PRC_state3.png')
plt.savefig(fig_save_dir + 'PRC_state3.svg')
plt.show()
#%%
plot_graph(K_tr, Kest_ave, vmin, vmax)
plt.savefig(fig_save_dir + 'estimated_graph.png')
plt.savefig(fig_save_dir + 'estimated_graph.svg')
plt.show()