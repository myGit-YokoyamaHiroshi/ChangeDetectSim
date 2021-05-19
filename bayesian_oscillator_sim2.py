from IPython import get_ipython
from copy import deepcopy, copy
get_ipython().magic('reset -sf')
#get_ipython().magic('cls')

import os
os.chdir('D:\\GitHub\\ChangeDetectSim\\')
# os.chdir('C:\\Users\\H.yokoyama\\Documents\\Python_Scripts\\ChangeDetectSim')
# os.chdir('D:\\Python_Scripts\\test_myBayesianModel_PRC\\') # Set full path of your corrent derectory

simName      = 'sim2'

current_path = os.getcwd()
fig_save_dir = current_path + '\\figures\\' + simName + '\\'

if os.path.exists(fig_save_dir)==False:  # Make the directory for figures
    os.makedirs(fig_save_dir)

param_path = current_path + '\\save_data\\param_' + simName + '\\'# Set path of directory where the dataset of parameter settings are saved.
    
import matplotlib.pylab as plt
plt.rcParams['font.family']      = 'Arial'#
plt.rcParams['mathtext.fontset'] = 'stix' # math font setting
plt.rcParams["font.size"]        = 26 # Font size

#%%
# from my_modules.my_dynamical_bayes import *
from my_modules.my_oscillator_model import *
from my_modules.my_dynamical_bayes_mod import my_Bayesian_CP

from my_modules.my_graph_visualization import *
from scipy.stats import zscore
from scipy.io import savemat
from numpy.random import *
import numpy as np
import glob

#%%
name     = []
ext      = []
for file in os.listdir(param_path):
    split_str = os.path.splitext(file)
    name.append(split_str[0])
    ext.append(split_str[1])
    
    print(split_str)
#%% Generate synthetic data
if (len(name) == 0) & (len(ext) == 0):
    ############# set parameters
    Nosc          = 3
    h             = 0.01
    Nt            = 3000
    State         = 3
    omega         = 2 * np.pi * normal(loc = 4.0, scale=0.5, size = 3) #  array([4.35761233, 4.35600486, 4.00517973])
    
    K1_tr         = np.zeros((Nosc, Nosc, State))
    K2_tr         = np.zeros((Nosc, Nosc, State))
    K_tr          = np.zeros((Nosc, Nosc, State))
    
    for st in range(State):
        if st == 0:
            K1_tr[:,:,st] = np.array([[0.00, 0.00, 0.30],
                                      [0.00, 0.00, 0.30],
                                      [0.00, 0.00, 0.00]])
            
            K2_tr[:,:,st] = np.array([[0.00, 0.00, 0.30],
                                      [0.00, 0.00, 0.30],
                                      [0.00, 0.00, 0.00]])
        elif st == 1:
            K1_tr[:,:,st] = np.array([[0.00, 0.30, 0.30],
                                      [0.30, 0.00, 0.30],
                                      [0.30, 0.30, 0.00]])
            
            K2_tr[:,:,st] = np.array([[0.00, 0.30, 0.30],
                                      [0.30, 0.00, 0.30],
                                      [0.30, 0.30, 0.00]])
        else:
            K1_tr[:,:,st] = np.array([[0.00, 0.30, 0.30],
                                      [0.30, 0.00, 0.30],
                                      [0.30, 0.30, 0.00]])
            
            K2_tr[:,:,st] = np.array([[0.00, 0.30, 0.30],
                                      [0.30, 0.00, 0.30],
                                      [0.30, 0.30, 0.00]])
            
        K_tr[:,:,st] = np.sqrt(K1_tr[:,:,st]**2 + K2_tr[:,:,st]**2)
    
    ############# Generate synthetic data
    theta_init    = np.random.uniform(low = 0, high = 2 * np.pi, size = Nosc)#np.array([np.pi, 0, np.pi/2])
    # dtheta        = np.zeros((Nt, Nosc))
    # theta         = np.zeros((Nt, Nosc))
    # theta[0, :]   = theta_init
    # noise_scale   = 0.01
    
    # phase_dynamics       = np.zeros((Nt, Nosc))
    # phase_dynamics[0, :] = func_oscillator_approx_fourier_series(theta[0, :], K1_tr[:,:,0], K2_tr[:,:,0], omega, noise_scale)
    # for t in range(1, Nt):
    #     if t < int(Nt/3):
    #         Nst = 0
    #         noise_scale = 0.01
    #     elif int(Nt/3) <= t < int(Nt*2/3):
    #         Nst = 1
    #         noise_scale = 0.01
    #     else:
    #         Nst = 2
    #         noise_scale = 0.5
        
    #     K1 = K1_tr[:,:,Nst]
    #     K2 = K2_tr[:,:,Nst]
        
    #     theta_now  = theta[t-1, :]
    #     theta_next = runge_kutta_oscillator_approx_fourier_series(h, func_oscillator_approx_fourier_series, theta_now, K1, K2, omega, noise_scale)
        
    #     theta[t, :]          = theta_next.reshape(1, Nosc)
    #     phase_dynamics[t, :] = func_oscillator_approx_fourier_series(theta[t, :], K1, K2, omega, noise_scale)
    
    #     for i in range(Nosc):
    #         theta_unwrap = np.unwrap(deepcopy(theta[t-1:t+1, i]))
            
    #         dtheta[t, i] = (theta_unwrap[1] - theta_unwrap[0])/h
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
    # param_dict['theta']          = theta          # phase (numerical solution of the model)
    # param_dict['dtheta']         = dtheta         # time deriviation of phase (numerical differentiation)
    # param_dict['phase_dynamics'] = phase_dynamics # time deriviation of phase (model output)
    
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
    # theta          = param_dict['theta']
    # dtheta         = param_dict['dtheta'] # time deriviation of phase (numerical differentiation)
    # phase_dynamics = param_dict['phase_dynamics'] # time deriviation of phase (model)
    
del param_dict
#%% solve stochastic differential equation using Eular-Maruyama Method
dtheta        = np.zeros((Nt, Nosc))
theta         = np.zeros((Nt, Nosc))
theta[0, :]   = theta_init
noise_scale   = 0.01

phase_dynamics       = np.zeros((Nt, Nosc))
phase_dynamics[0, :] = func_oscillator_approx_fourier_series(theta[0, :], K1_tr[:,:,0], K2_tr[:,:,0], omega, noise_scale)
for t in range(1, Nt):
    if t < int(Nt/3):
        Nst = 0
        noise_scale = 0.01
    elif int(Nt/3) <= t < int(Nt*2/3):
        Nst = 1
        noise_scale = 0.01
    else:
        Nst = 2
        noise_scale = 0.1
    
    K1 = K1_tr[:,:,Nst]
    K2 = K2_tr[:,:,Nst]
    
    theta_now  = theta[t-1, :]
    theta_next = euler_maruyama_oscillator_approx_fourier_series(h, func_oscillator_approx_fourier_series, theta_now, K1, K2, omega, noise_scale)
    
    theta[t, :]          = theta_next.reshape(1, Nosc)
    phase_dynamics[t, :] = func_oscillator_approx_fourier_series(theta[t, :], K1, K2, omega, noise_scale)

    for i in range(Nosc):
        theta_unwrap = np.unwrap(deepcopy(theta[t-1:t+1, i]))
        
        dtheta[t, i] = (theta_unwrap[1] - theta_unwrap[0])/h

matpath = current_path + '\\LEiDA\\mat\\'
if os.path.exists(matpath )==False:  # Make the directory for figures
    os.makedirs(matpath)
savemat(matpath+"phase_data_sim2.mat", {'theta':theta, 'h':h, 'fs':1/h})
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
    # plt.xticks(np.arange(0, Nt+1, int(Nt/3))) 
    if n < Nosc-1:
        plt.xticks([]) 
    else:
        plt.xticks(np.arange(0, 1000, 200)) 
    plt.yticks([0, np.pi, 2*np.pi], labels=['0', '$\\pi$', '$2 \\pi$'])
    plt.xlim(0, 1000)
    plt.ylim(-0.8, 2*np.pi + 0.8)
    plt.grid()

plt.xlabel('# sample')

plt.show()
#%% plot phase dynamics
fig = plt.figure(figsize=(20, 4))
plt.plot(axis, phase_dynamics)
plt.title('synthetic data')
plt.legend(bbox_to_anchor=(1.05, 1), labels = ['oscillator 1', 'oscillator 2', 'oscillator 3'], loc='upper left', borderaxespad=0, fontsize=26)
plt.xticks(np.arange(0, Nt+1, int(Nt/3))) 
plt.xlabel('# sample')
plt.ylabel('phase velocity')
plt.grid()
plt.subplots_adjust(right = 0.7)
plt.savefig(fig_save_dir + 'phase_dynamics.png')
plt.savefig(fig_save_dir + 'phase_dynamics.svg')
plt.show()
#%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
P = 1 # order of Forier series
T = 1 # Time steps for sequential bayesian updates
x = deepcopy(theta)

noise_param = 1E-4 # covariance of process noise
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
    
Time = np.arange(0, Nt-T)
Kest = np.sqrt(np.sum(beta**2, axis=2))

PRC, phi_dlt_plt  = reconstruct_phase_response_curve(beta, OMEGA, Nosc)
#%%
fig = plt.figure(figsize=(20, 4))

plt.plot(Time, OMEGA)
plt.xlabel('# sample')
plt.ylabel('natural frequency (a.u.)')
plt.legend(bbox_to_anchor=(1.05, 1), labels = ['oscillator 1', 'oscillator 2', 'oscillator 3'], loc='upper left', borderaxespad=0, fontsize=26)
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
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0, fontsize=26, frameon=False)
plt.xlabel('# sample')
plt.ylabel('Changing ratio (a.u.)')
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
plt.xlabel('# sample')
plt.ylabel('Log-likelihood (a.u.)')
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
plt.xlabel('# sample')
plt.ylabel('phase velocity')
plt.grid()

handle = [line1[-1], line2[-1]]
labels = ['pred.', 'true']
plt.legend(handle, labels, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0, fontsize=26)
plt.subplots_adjust(right = 0.7)
plt.savefig(fig_save_dir + 'phase_dynamics_est.png')
plt.savefig(fig_save_dir + 'phase_dynamics_est.svg')
plt.show()

#%%
fig = plt.figure(figsize=(4, 4))

line1 = plt.plot(Time, y_hat, c = 'k', linestyle = '-', zorder = 1, label = 'pred')
line2 = plt.plot(axis, phase_dynamics, c = np.array([0.5, 0.5, 0.5]), linewidth = 4,zorder = 0, label = 'true')
# plt.xticks(np.arange(0, Nt+1, int(Nt/3)))  # plt.xticks(np.arange(0, Nt+1, int(Nt/2)))  # 
# plt.xlabel('# sample')
# plt.ylabel('phase velocity')
plt.grid()

plt.xlim([-5, 80])
plt.ylim([8, 35])

handle = [line1[-1], line2[-1]]
labels = ['pred.', 'true']
# plt.legend(handle, labels, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0, fontsize=26)
plt.subplots_adjust(right = 0.7)
# plt.savefig(fig_save_dir + 'phase_dynamics_est.png')
# plt.savefig(fig_save_dir + 'phase_dynamics_est.svg')
plt.show()

#%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
####### Phase response curve #################################################
##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
idx_st1 = Time<(int(Nt/3))
idx_st2 = (Time>=int(Nt/3)) & (Time<int(Nt*2/3))
idx_st3 = (Time>=int(Nt*2/3))
phi     = theta[0:-1,:]
dphi    = dtheta[1:,:]

Kest_ave1 = np.median(Kest[idx_st1,:], axis=0).reshape(Nosc, Nosc)
Kest_ave2 = np.median(Kest[idx_st2,:], axis=0).reshape(Nosc, Nosc)
Kest_ave3 = np.median(Kest[idx_st2,:], axis=0).reshape(Nosc, Nosc)

Kest_ave  = np.concatenate((Kest_ave1[:,:,np.newaxis], 
                            Kest_ave2[:,:,np.newaxis],
                            Kest_ave3[:,:,np.newaxis]), axis=2)
#%%
vmin    = 0
vmax    = 0.7#Kest.mean() + Kest.std()

plot_PRC(phi[idx_st1,:], dphi[idx_st1,:], phi_dlt_plt, PRC[idx_st1,:,:], Kest_ave1, 0, vmax, Nosc, ylims=[18, 32])
plt.savefig(fig_save_dir + 'PRC_state1.png')
plt.savefig(fig_save_dir + 'PRC_state1.svg')
plt.show()

plot_PRC(phi[idx_st2,:], dphi[idx_st2,:], phi_dlt_plt, PRC[idx_st2,:,:], Kest_ave2, 0, vmax, Nosc, ylims=[18, 32])
plt.savefig(fig_save_dir + 'PRC_state2.png')
plt.savefig(fig_save_dir + 'PRC_state2.svg')
plt.show()

plot_PRC(phi[idx_st3,:], dphi[idx_st3,:], phi_dlt_plt, PRC[idx_st3,:,:], Kest_ave3, 0, vmax, Nosc, ylims=[18, 32])
plt.savefig(fig_save_dir + 'PRC_state3.png')
plt.savefig(fig_save_dir + 'PRC_state3.svg')
plt.show()
#%%
plot_graph(K_tr, Kest_ave, vmin, vmax)
plt.savefig(fig_save_dir + 'estimated_graph.png')
plt.savefig(fig_save_dir + 'estimated_graph.svg')
plt.show()

#%%
# vmin    = -0.6
# vmax    =  0.6#Kest.mean() + Kest.std()

# fig       = plt.figure(constrained_layout = False, figsize=(10, 8));
# plt.subplots_adjust(wspace=0.8, hspace=0.8);
# gs        = fig.add_gridspec(3, State)
# ratios    = [1,1,0.08]
# gs.set_height_ratios(ratios)

# cmaps     = 'bwr'
# ax_cb     = fig.add_subplot(gs[2, 0:State+1])
# cbar_info = [False, {"orientation":"horizontal"}, ax_cb]

# for state in range(State):
#     ### coeff. a_ij
    
#     ax = fig.add_subplot(gs[0, state])
#     vis_heatmap(K_tr[:,:,state], vmin, vmax, cmaps, ax, np.array(['Segment %d\n'%(state+1), 'osci. $j$', 'osci. $i$']), cbar_info)
    
#     ax = fig.add_subplot(gs[1, state])
#     if state == State-1:
#         cbar_info = [True, {"orientation":"horizontal"}, ax_cb]
#     vis_heatmap(Kest_ave[:,:,state], vmin, vmax, cmaps, ax, np.array(['', 'osci. $j$', 'osci. $i$']), cbar_info)
# plt.savefig(fig_save_dir + 'estimated_graph.png')
# plt.savefig(fig_save_dir + 'estimated_graph.svg')
# plt.show()