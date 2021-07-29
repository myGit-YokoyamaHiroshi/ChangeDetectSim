from IPython import get_ipython
from copy import deepcopy, copy
get_ipython().magic('reset -sf')
#get_ipython().magic('cls')

import os
os.chdir('D:\\GitHub\\ChangeDetectSim\\')# Set full path of your corrent derectory

simName      = 'sim1'
Nosc         = 3

current_path = os.getcwd()
fig_save_dir = current_path + '\\figures\\' + simName + '\\'  + 'Nosc_' + '%02d'%(Nosc) + '\\'

if os.path.exists(fig_save_dir)==False:  # Make the directory for figures
    os.makedirs(fig_save_dir)

param_path = current_path + '\\save_data\\param_' + simName + '\\' + 'Nosc_' + '%02d'%(Nosc) + '\\' # Set path of directory where the dataset of parameter settings are saved.
if os.path.exists(param_path)==False:  # Make the directory for figures
    os.makedirs(param_path)
  
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
from numpy.random import *
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

if  (len(name) != 2) & (len(ext) != 2): # 
    ############# set parameters   
    h            = 0.01
    Nt           = 4000
    State        = 1
    omega        = 2 * np.pi * normal(loc = 4, scale=0.5,  size = Nosc) #np.concatenate((normal(loc = 4, scale=0.5, size = 11), normal(loc = 3.2, scale=0.5, size = 8))) # 
    theta_init  = np.random.uniform(low = 0, high = 2 * np.pi, size = Nosc)
    
    
    K1_tr        = np.zeros((Nosc, Nosc))
    K2_tr        = np.zeros((Nosc, Nosc))
    K_tr         = np.zeros((Nosc, Nosc))
    
    tmp          = np.zeros((Nosc, Nosc))
    tmp[:, 0:int(round(Nosc/3))]  = 0.3
    tmp          = tmp - np.diag(np.diag(tmp))
    K2_tr       = tmp
            
    K_tr        = np.sqrt(K1_tr**2 + K2_tr**2) 
    ####### Generate synthetic data
    # dtheta      = np.zeros((Nt, Nosc))
    # theta       = np.zeros((Nt, Nosc))
    # theta[0, :] = theta_init
    # noise_scale = 0.01
    
    # phase_dynamics       = np.zeros((Nt, Nosc))
    # phase_dynamics[0, :] = func_oscillator_approx_fourier_series(theta[0, :], K1_tr, K2_tr, omega, noise_scale)
    # for t in range(1, Nt):
        
    #     K1 = K1_tr
    #     K2 = K2_tr
        
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
    
    save_name                    = 'Sim_param_' + simName
    fullpath_save                = param_path
    
    if os.path.exists(fullpath_save)==False:  # Make the directory for figures
        os.makedirs(fullpath_save) 
    np.save(fullpath_save + save_name, param_dict)
else:
    ##### Load the parameter settings
    fullpath       = param_path + 'Sim_param_' + simName + '.npy'
    
    param_dict     = np.load(fullpath, encoding='ASCII', allow_pickle='True').item()
    
    # Nosc           = param_dict['Nosc']
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
dtheta      = np.zeros((Nt, Nosc))
theta       = np.zeros((Nt, Nosc))
theta[0, :] = theta_init
noise_scale = 0.001

phase_dynamics       = np.zeros((Nt, Nosc))
phase_dynamics[0, :] = func_oscillator_approx_fourier_series(theta[0, :], K1_tr, K2_tr, omega)

np.random.seed(0)
for t in range(1, Nt):
    
    K1 = K1_tr
    K2 = K2_tr
    
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
    plt.xticks(np.arange(0, Nt+1, int(Nt/2)))  # plt.xticks(np.arange(0, Nt+1, int(Nt/3))) 
    plt.yticks([0, np.pi, 2*np.pi], labels=['0', '$\\pi$', '$2 \\pi$'])
    plt.grid()
    
plt.xlabel('# sample')

plt.show()
#%% plot phase dynamics
fig = plt.figure(figsize=(20, 4))
plt.plot(axis, phase_dynamics)
plt.title('synthetic data')
plt.legend(bbox_to_anchor=(1.05, 1), labels = ['oscillator 1', 'oscillator 2', 'oscillator 3'], loc='upper left', borderaxespad=0, fontsize=26)
plt.xticks(np.arange(0, Nt+1, int(Nt/2)))  # plt.xticks(np.arange(0, Nt+1, int(Nt/3))) 
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
#### calculate coupling strength Kest 
Time = np.arange(0, Nt-T)
Kest = np.sqrt(np.sum(beta**2, axis=2))
#### evaluate threshold Cth for cp-score
C   = Changes[np.isnan(Changes)==False]
Cth = C.mean() + 3 * C.std()
#%% #### calculate phase coupling function
PRC, phi_dlt_plt  = reconstruct_phase_response_curve(beta, OMEGA, Nosc)
#%% calculate epoch-average of coupling strength
idx_st1 = Time<int(Nt/2)
idx_st2 = Time>=int(Nt/2)
phi     = theta[0:-T,:]
dphi    = dtheta[T:,:]
vmin    = 0
vmax    = 1#Kest.mean() + Kest.std()

Kest_ave  = np.median(Kest, axis=0).reshape(Nosc, Nosc)
a         = K1_tr
b         = K2_tr
a_est     = np.median(beta[:,:,0],axis=0).reshape(Nosc, Nosc)
b_est     = np.median(beta[:,:,1],axis=0).reshape(Nosc, Nosc)

###############################################################################
#%% Figure visualization ######################################################
###############################################################################
fig = plt.figure(figsize=(20, 4))

plt.plot(Time, OMEGA)
plt.xlabel('# sample')
plt.ylabel('natural frequency (a.u.)')
plt.legend(bbox_to_anchor=(1.05, 1), labels = ['oscillator 1', 'oscillator 2', 'oscillator 3'], loc='upper left', borderaxespad=0, fontsize=26)
plt.xticks(np.arange(0, Nt+1, int(Nt/2)))  # plt.xticks(np.arange(0, Nt+1, int(Nt/3))) 
# plt.ylim(0.0, 4.0)
plt.grid()
plt.subplots_adjust(right = 0.7)
plt.savefig(fig_save_dir + 'natural_freqs.png')
plt.savefig(fig_save_dir + 'natural_freqs.svg')
plt.show()
#%%
idx = np.where(Changes > Cth)[0]
fig = plt.figure(figsize=(20, 4))

plt.plot(Time[np.isnan(Changes)==False], Changes[np.isnan(Changes)==False]);
plt.scatter(Time[idx], Changes[idx], marker = 'o', c = 'red', label = '> mean + 3SD');
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0, fontsize=26, frameon=False)
plt.xlabel('# sample')
plt.ylabel('Change-point score \n(KL div.)')
plt.xticks(np.arange(0, Nt+1, int(Nt/2)))  # plt.xticks(np.arange(0, Nt+1, int(Nt/3))) 

plt.subplots_adjust(right = 0.7)
plt.grid()
# plt.ylim(-5000.0, 60000.0)
plt.savefig(fig_save_dir + 'changing_point.png')
plt.savefig(fig_save_dir + 'changing_point.svg')
plt.show()
#%% plot phase dynamics
fig = plt.figure(figsize=(20, 4))

line1 = plt.plot(Time, y_hat, c = 'k', linestyle = '-', zorder = 1, label = 'pred')
line2 = plt.plot(axis, phase_dynamics, c = np.array([0.5, 0.5, 0.5]), linewidth = 4,zorder = 0, label = 'true')
plt.xticks(np.arange(0, Nt+1, int(Nt/2)))  # plt.xticks(np.arange(0, Nt+1, int(Nt/3))) 
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

#%% ##########################################################################
#### save data
##############################################################################
save_dict                  = {}

save_dict['noise_param']   = noise_param
save_dict['prec_param']    = prec_param
save_dict['window_size']   = T
save_dict['fourier_order'] = P
save_dict['fourier_coeff'] = beta
save_dict['omega']         = OMEGA
save_dict['cp_score']      = Changes
save_dict['loglike']       = L
save_dict['y_hat']         = y_hat

save_dict['dtheta']        = phase_dynamics
save_dict['theta']         = theta
save_dict['h']             = h
save_dict['Nt']            = Nt
save_dict['State']         = State
save_dict['omega_init']    = omega
save_dict['theta_init']    = theta_init


save_dict['a_tr']          = a
save_dict['b_tr']          = b
save_dict['K_tr']          = K_tr

save_dict['a_est']         = a_est
save_dict['b_est']         = b_est
save_dict['K_est']         = Kest_ave


fullpath_save              = param_path 
save_name                  = 'estimation_result_' + 'Nosc_' + '%02d'%(Nosc)
np.save(fullpath_save + save_name, save_dict)
