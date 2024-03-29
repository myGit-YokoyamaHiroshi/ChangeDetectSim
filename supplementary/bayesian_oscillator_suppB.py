from IPython import get_ipython
from copy import deepcopy, copy
get_ipython().magic('reset -sf')
#get_ipython().magic('cls')

import os
current_path = os.path.dirname(__file__)
os.chdir(current_path)

simName      = 'suppB'

fig_save_dir = current_path + '\\figures\\' + simName + '\\'

if os.path.exists(fig_save_dir)==False:  # Make the directory for figures
    os.makedirs(fig_save_dir)

param_path = current_path + '\\save_data\\param_' + simName + '\\' # Set path of directory where the dataset of parameter settings are saved.

if os.path.exists(param_path)==False:  # Make the directory for figures
    os.makedirs(param_path)

import matplotlib.pylab as plt
import matplotlib.gridspec as gridspec
plt.rcParams['font.family']      = 'Arial'#
plt.rcParams['mathtext.fontset'] = 'stix' # math font setting
plt.rcParams["font.size"]        = 28 # Font size

#%%
# from my_modules.my_dynamical_bayes import *
from my_modules.my_oscillator_model import *
from my_modules.my_dynamical_bayes_mod import my_Bayesian_CP

from my_modules.my_graph_visualization import *
from scipy.stats import zscore
from numpy.random import *
from scipy.io import savemat
import numpy as np
import glob
import scipy.stats  as stats
import scipy.sparse as sparse
#%%
name     = []
ext      = []
for file in os.listdir(param_path):
    split_str = os.path.splitext(file)
    name.append(split_str[0])
    ext.append(split_str[1])
    
    print(split_str)
#%% Generate synthetic data
if  (len(name) == 0) & (len(ext) == 0): # 
    ############# set parameters   
    Nosc          = 10
    h             = 0.01
    Nt            = 3000
    State         = 3
    omega         = 2 * np.pi * normal(loc = 4, scale=0.5,  size = Nosc) #np.concatenate((normal(loc = 4, scale=0.5, size = 11), normal(loc = 3.2, scale=0.5, size = 8))) # 
    
    K1_tr         = np.zeros((Nosc, Nosc, State))
    K2_tr         = np.zeros((Nosc, Nosc, State))
    K_tr          = np.zeros((Nosc, Nosc, State))
    
    for st in range(State):
        if st == 0:
            tmp           = 0.4*sparse.random(Nosc, Nosc, format='csr', density=0.3).A
            tmp           = tmp - np.diag(np.diag(tmp))
            
            K1_tr[:,:,st] = tmp
            K2_tr[:,:,st] = tmp
        elif st == 1:   
            tmp           = 0.4*sparse.random(Nosc, Nosc, format='csr', density=0.3).A
            tmp           = tmp - np.diag(np.diag(tmp))
            
            K1_tr[:,:,st] = tmp
            K2_tr[:,:,st] = tmp
        else:
            tmp           = 0.4*sparse.random(Nosc, Nosc, format='csr', density=0.3).A
            tmp           = tmp - np.diag(np.diag(tmp))
            
            K1_tr[:,:,st] = tmp
            K2_tr[:,:,st] = tmp
            
        K_tr[:,:,st] = np.sqrt(K1_tr[:,:,st]**2 + K2_tr[:,:,st]**2)
        
    theta_init    = np.random.uniform(low = 0, high = 2*np.pi, size = Nosc)
    
            
    ############# save_data
    param_dict                   = {} 
    param_dict['Nosc']           = Nosc
    param_dict['h']              = h
    param_dict['Nt']             = Nt
    param_dict['State']          = State
    param_dict['omega']          = omega
    param_dict['K1_tr']          = K1_tr
    param_dict['K2_tr']          = K2_tr
    param_dict['K_tr']           = K_tr
    
    param_dict['theta_init']     = theta_init     # initial value of phase
    
    save_name   = 'Sim_param_' + simName
    fullpath_save   = param_path + save_name 
    np.save(fullpath_save, param_dict)
else:
    ##### Load the parameter settings
    fullpath       = param_path + name[0] + ext[0]
    param_dict     = np.load(fullpath, encoding='ASCII', allow_pickle='True').item()
    
    Nosc           = param_dict['Nosc']
    h              = param_dict['h']
    Nt             = param_dict['Nt']
    State          = param_dict['State']
    omega          = param_dict['omega']
    K1_tr          = param_dict['K1_tr']
    K2_tr          = param_dict['K2_tr']
    K_tr           = param_dict['K_tr']
    
    theta_init     = param_dict['theta_init']
del param_dict
#%% solve stochastic differential equation using Eular-Maruyama Method
dtheta        = np.zeros((Nt, Nosc))
theta         = np.zeros((Nt, Nosc))
theta[0, :]   = theta_init
noise_scale   = 0.001

phase_dynamics       = np.zeros((Nt, Nosc))
phase_dynamics[0, :] = func_oscillator_approx_fourier_series(theta[0, :], K1_tr[:,:,0], K2_tr[:,:,0], omega)

np.random.seed(1)
for t in range(1, Nt):
    if t < int(Nt/3):
        Nst = 0
        noise_scale = 0.001
    elif int(Nt/3) <= t < int(Nt*2/3):
        Nst = 1
        noise_scale = 0.001
    else:
        Nst = 2
        noise_scale = 0.001
    
    K1 = K1_tr[:,:,Nst]
    K2 = K2_tr[:,:,Nst]
    
    theta_now  = theta[t-1, :]
    theta_next = euler_maruyama_oscillator_approx_fourier_series(h, func_oscillator_approx_fourier_series, theta_now, K1, K2, omega, noise_scale)
    
    theta[t, :]          = theta_next.reshape(1, Nosc)
    phase_dynamics[t, :] = func_oscillator_approx_fourier_series(theta[t, :], K1, K2, omega)

    for i in range(Nosc):
        theta_unwrap = np.unwrap(deepcopy(theta[t-1:t+1, i]))
        
        dtheta[t, i] = (theta_unwrap[1] - theta_unwrap[0])/h
#%% plot phase

axis=np.arange(theta.shape[0])

fig = plt.figure(figsize=(20, 8))
gs  = fig.add_gridspec(Nosc, 1)
plt.subplots_adjust(wspace=0.4, hspace=0.0)

for n in range(Nosc):
    #####################
    ax11 = fig.add_subplot(gs[n, 0])
    plt.plot(axis, theta[:, n])
    
    plt.ylabel('$\\phi_{%d}$'%(n+1))
    plt.xticks(np.arange(0, Nt+1, int(Nt/3)))  # plt.xticks(np.arange(0, Nt+1, int(Nt/2)))  # 
    plt.yticks([0, np.pi, 2*np.pi], labels=['0', '$\\pi$', '$2 \\pi$'])
    plt.grid()
    
plt.xlabel('# sample')

plt.show()
#%% plot phase dynamics
fig = plt.figure(figsize=(20, 4))
plt.plot(axis[1:], dtheta[1:,:])
plt.title('synthetic data')
plt.legend(bbox_to_anchor=(1.05, 1), labels = ['oscillator 1', 'oscillator 2', 'oscillator 3'], loc='upper left', borderaxespad=0, fontsize=26)
plt.xticks(np.arange(0, Nt+1, int(Nt/3)))  # plt.xticks(np.arange(0, Nt+1, int(Nt/2)))  # 
plt.xlabel('# sample')
plt.ylabel('phase velocity')
plt.grid()
plt.subplots_adjust(right = 0.7)
plt.savefig(fig_save_dir + 'phase_dynamics.png', bbox_inches="tight")
plt.savefig(fig_save_dir + 'phase_dynamics.svg', bbox_inches="tight")
plt.savefig(fig_save_dir + 'phase_dynamics.eps', bbox_inches="tight")
plt.show()
#%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
P = 1 # order of Forier series
T = 1 # Time steps for sequential bayesian updates
x = deepcopy(theta)

noise_param = 1E-4#1E-3 # covariance of process noise
prec_param  = 1/noise_param # precision parameter, cov(process noise) = 1/prec_param

#%% Bayesian estimation and change point detection
cnt = 1
# beta, OMEGA, Changes, L, y_hat, sigma0, Kb0 = est_dynamical_oscillator_1st_order_fourier(x, P, T, h, prec_param)
bayes_cp = my_Bayesian_CP(x, P, T, h, prec_param)
bayes_cp.est_dynamical_oscillator_1st_order_fourier()

beta    = bayes_cp.beta
OMEGA   = bayes_cp.omega
Changes = bayes_cp.Changes
L       = bayes_cp.loglike
y_hat   = bayes_cp.y_hat
#sigma0, Kb0 


if len(OMEGA.shape)==3:
    OMEGA = OMEGA[:,:,0]
#### calculate coupling strength Kest 
Time = np.arange(0, Nt-T)
Kest = np.sqrt(np.sum(beta**2, axis=2))
#### evaluate threshold Cth for cp-score
C       = Changes
Cth     = C.mean() + 3 * C.std()

# q75, q25 = np.percentile(C, [75 ,25])
# iqr      = q75 - q25
# Cth      = q75 + 1.5 * iqr
#%% calculate epoch-average of coupling strength
idx_st1 = Time<(int(Nt/3))
idx_st2 = (Time>=int(Nt/3)) & (Time<int(Nt*2/3))
idx_st3 = Time>=int(Nt*2/3)

# idx_st1 = np.where(Time>=(int(Nt/3))-10)[0][0]
# idx_st2 = np.where(Time>=int(Nt*2/3)-10)[0][0]
# idx_st3 = len(Time)-10
phi     = theta[0:-1,:]
dphi    = dtheta[1:,:]
vmin    = -0.7
vmax    =  0.7

Kest_ave1 = np.median(Kest[idx_st1,:], axis=0).reshape(Nosc, Nosc)
Kest_ave2 = np.median(Kest[idx_st2,:], axis=0).reshape(Nosc, Nosc)
Kest_ave3 = np.median(Kest[idx_st3,:], axis=0).reshape(Nosc, Nosc)

# Kest_ave1 = Kest[idx_st1,:].reshape(Nosc, Nosc)
# Kest_ave2 = Kest[idx_st2,:].reshape(Nosc, Nosc)
# Kest_ave3 = Kest[idx_st3,:].reshape(Nosc, Nosc)

Kest_ave  = np.concatenate((Kest_ave1[:,:,np.newaxis], 
                            Kest_ave2[:,:,np.newaxis],
                            Kest_ave3[:,:,np.newaxis]), axis=2)
b_est = np.median(beta[-100:,:,1],axis=0).reshape(Nosc, Nosc)
###############################################################################
#%% Figure visualization ######################################################
###############################################################################
#%% plot phase dynamics
fig = plt.figure(figsize=(20, 4))

line1 = plt.plot(Time, y_hat, c = 'k', linestyle = '-', zorder = 1, label = 'pred')
line2 = plt.plot(axis, dtheta, c = np.array([0.5, 0.5, 0.5]), linewidth = 4,zorder = 0, label = 'true')
plt.xticks(np.arange(0, Nt+1, int(Nt/3)))  # plt.xticks(np.arange(0, Nt+1, int(Nt/2)))  # 
plt.xlabel('# sample')
plt.ylabel('phase velocity')
plt.grid()

handle = [line1[-1], line2[-1]]
labels = ['pred.', 'true']
# plt.legend(handle, labels, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0, fontsize=26, frameon=False)
plt.legend(handle, labels, loc='lower right', borderaxespad=0, fontsize=26, frameon=False)
plt.subplots_adjust(right = 0.7)
plt.savefig(fig_save_dir + 'phase_dynamics_est.png', bbox_inches="tight")
plt.savefig(fig_save_dir + 'phase_dynamics_est.svg', bbox_inches="tight")
plt.savefig(fig_save_dir + 'phase_dynamics_est.eps', bbox_inches="tight")
plt.show()

#%%
idx   = np.where(C > Cth)[0]
fig   = plt.figure(figsize=(15, 12))
outer = gridspec.GridSpec(2, 1, wspace=0.25, hspace=0.5, height_ratios=[1,0.8])

vmin  = -0.7
vmax  =  0.7

inner = gridspec.GridSpecFromSubplotSpec(2, 5, subplot_spec=outer[0], wspace=0.2, hspace=0.5, width_ratios=[0.1,1,1,1,0.08])
tmp   = plt.Subplot(fig, inner[:,State+1])
ax_cb = fig.add_subplot(tmp)
cbar_info = [False, {"orientation":"vertical", 'label': 'Coupling strength (a.u.)'}, ax_cb]
for state in range(State):
    
    ax = plt.Subplot(fig, inner[0,state+1])
    vis_heatmap(K_tr[:,:,state], vmin, vmax, ax, np.array(['Segment %d\n'%(state+1), 'osci. $j$', 'osci. $i$']), cbar_info, linewidths = 0.0, fontsize=28)
    fig.add_subplot(ax)
    if state == 0:
        ax_pos = ax.get_position()
        fig.text(ax_pos.x1 - .22, ax_pos.y1-0.08, 'true')
        fig.text(ax_pos.x1 - .3, ax_pos.y1, 'A', fontsize=40)
        fig.text(ax_pos.x1 - .3, ax_pos.y1-0.42, 'B', fontsize=40)
    
    ax = plt.Subplot(fig, inner[1,state+1])
    if state == State-1:
        cbar_info = [True, {"orientation":"vertical", 'label': 'Coupling strength (a.u.)'}, ax_cb]
    elif state == 0:
        ax_pos = ax.get_position()
        fig.text(ax_pos.x1 - .28, ax_pos.y1-0.08, 'pred.')
    vis_heatmap(Kest_ave[:,:,state], vmin, vmax, ax, np.array(['', 'osci. $j$', 'osci. $i$']), cbar_info, linewidths = 0.0)
    fig.add_subplot(ax)

ax = plt.Subplot(fig, outer[1])
ax.plot(Time, C, label='Change point score');
ax.scatter(Time[idx], C[idx], marker = 'o', c = 'red', label = 'Change point\n(KL div > $\\mu$ + 3SD)');
# ax.legend(loc='upper right', borderaxespad=0, fontsize=26, frameon=False)
ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left',  borderaxespad=0, fontsize=26, frameon=True)
ax.set_xlabel('# sample')
ax.set_ylabel('Change-point score \n(KL div.)')
ax.set_xticks(np.arange(0, Nt+1, int(Nt/3)))  # plt.xticks(np.arange(0, Nt+1, int(Nt/2)))  # 

ylims = np.array(ax.get_ylim())
ax.set_ylim(ylims)
ax.text( 200, 1.1*(Changes[np.isnan(Changes)==False].max()), 'Segment 1')
ax.text(1200, 1.1*(Changes[np.isnan(Changes)==False].max()), 'Segment 2')
ax.text(2200, 1.1*(Changes[np.isnan(Changes)==False].max()), 'Segment 3')
fig.add_subplot(ax)
plt.grid()
# plt.ylim(-5000.0, 60000.0)
plt.savefig(fig_save_dir + 'changing_point.png', bbox_inches="tight")
plt.savefig(fig_save_dir + 'changing_point.svg', bbox_inches="tight")
plt.savefig(fig_save_dir + 'changing_point.eps', bbox_inches="tight")

plt.show()

#%%

fig = plt.figure(figsize=(9, 9))
outer = gridspec.GridSpec(3, 2, wspace=0.25, hspace=0.0, width_ratios=[1,0.04])

tmp   = plt.Subplot(fig, outer[:,1])
ax_cb = fig.add_subplot(tmp)

vmin  = -0.6
vmax  =  0.6

inner1 = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=outer[0, 0], wspace=0.9)
inner2 = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=outer[1, 0], wspace=0.9)
inner3 = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=outer[2, 0], wspace=0.9)

for i in range(State):
    
    a_ax = plt.Subplot(fig, inner1[i])
    vis_heatmap(K1_tr[:,:,i], vmin, vmax, a_ax, np.array(['Segment %d\n $a_{ij}$'%(i+1), 'osci. $j$', 'osci. $i$']), cbar_info, linewidths = 0.0, fontsize=28)
    fig.add_subplot(a_ax)
    
    b_ax = plt.Subplot(fig, inner2[i])
    vis_heatmap(K2_tr[:,:,i], vmin, vmax, b_ax, np.array(['\n $b_{ij}$', 'osci. $j$', 'osci. $i$']), cbar_info, linewidths = 0.0, fontsize=28)
    fig.add_subplot(b_ax)
    
    if i==2:
        cbar_info = [True, {"orientation":"vertical", 'label': 'Coupling strength (a.u.)'},  ax_cb]
    else:
        cbar_info = [False, {"orientation":"vertical", 'label': 'Coupling strength (a.u.)'},  ax_cb]
    
    k_ax = plt.Subplot(fig, inner3[i])
    vis_heatmap(K_tr[:,:,i], vmin, vmax, k_ax, np.array(['\n $C_{ij}$', 'osci. $j$', 'osci. $i$']), cbar_info, linewidths = 0.0, fontsize=28)
    fig.add_subplot(k_ax)
    
    
    
plt.savefig(fig_save_dir + 'param_setting.png', bbox_inches="tight")
plt.savefig(fig_save_dir + 'param_setting.svg', bbox_inches="tight")
plt.savefig(fig_save_dir + 'param_setting.eps', bbox_inches="tight")
plt.show()