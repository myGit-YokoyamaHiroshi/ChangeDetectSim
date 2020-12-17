from IPython import get_ipython
from copy import deepcopy, copy
get_ipython().magic('reset -sf')
#get_ipython().magic('cls')

import os
os.chdir('D:\\GitHub\\ChangeDetectSim\\')
# os.chdir('C:\\Users\\H.yokoyama\\Documents\\Python_Scripts\\ChangeDetectSim')
# os.chdir('D:\\Python_Scripts\\test_myBayesianModel_PRC\\') 

sim_name     = 'sim2'
current_path = os.getcwd()
fig_save_dir = current_path + '\\figures\\' +  sim_name + '\\'

if os.path.exists(fig_save_dir)==False:  # Make the directory for figures
    os.makedirs(fig_save_dir)

param_path = current_path + '\\save_data\\param_' +  sim_name + '\\'# Set path of directory where the dataset of parameter settings are saved.
    
import matplotlib.pylab as plt
plt.rcParams['font.family']      = 'Arial'#
plt.rcParams['mathtext.fontset'] = 'stix' # math font setting
plt.rcParams["font.size"]        = 26 # Font size

#%%
from my_modules.my_dynamical_bayes import *
from my_modules.my_graph_visualization import *
from scipy.stats import zscore
from numpy.random import *
from mne.stats import fdr_correction as fdr
import numpy as np
import glob
#%% make surrogate phase
def phase_shuffle(phase): # phase (rad)
    
    sig     = np.exp(1j * phase)
    phs_rnd = np.angle(np.random.permutation(sig))
    phs_rnd = np.mod(np.unwrap(phs_rnd), 2*np.pi)
    
    return phs_rnd
#### calc pval with surrogate distribution
def surrogate_pvals(value, nulldist, tail='one'):
    nulldist = np.sort(nulldist)
    N        = len(nulldist)
    
    if (np.sum(nulldist == 0) == N) & (value == 0):
        pvals    = 1
    else:
        pvals    = np.sum((nulldist>value)==True)/N
    
    if tail == 'both':
        pvals = np.min(pvals, 1-pvals)
        pvals = 2*pvals
    
    if pvals == 0:
        pvals = 1/N
    return pvals  
#%%
name     = []
ext      = []
for file in os.listdir(param_path):
    split_str = os.path.splitext(file)
    name.append(split_str[0])
    ext.append(split_str[1])
    
    print(split_str)
#%% Load the parameter settings
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
theta          = param_dict['theta']
dtheta         = param_dict['dtheta'] # time deriviation of phase (numerical differentiation)
phase_dynamics = param_dict['phase_dynamics'] # time deriviation of phase (model)
    
del param_dict
#%% get the number of index for each segment

time    = np.arange(0, Nt)

idx_st1 = time<(int(Nt/3))
idx_st2 = (time>=int(Nt/3)) & (time<int(Nt*2/3))
idx_st3 = (time>=int(Nt*2/3))

#%% make surrogate distribution
Nprm     = 1000
PSI_prm  = np.zeros((Nosc, Nosc, Nprm))
R_prm    = np.zeros((Nosc, Nosc, Nprm))


for prm in range(Nprm):
    for i in range(Nosc):
        phs1 = deepcopy(theta[:,i])
        sig1 = np.cos(phs1)
        for j in range(Nosc):
            S    = np.random.permutation(np.arange(0, State))[0]
            if S == 0:
                idx = np.where(idx_st1==True)[0]
            elif S == 1:
                idx = np.where(idx_st2==True)[0]
            else:
                idx = np.where(idx_st3==True)[0]
            
            N    = len(idx)
            
            phs2 = deepcopy(theta[:,j])
            phs2 = phase_shuffle(phs2)
            sig2 = np.cos(phs2)
            
            dif  = phs2[idx] - phs1[idx]
            
            PSI_prm[i,j,prm] = abs(np.sum(np.exp(1j * dif)))/N
            R_prm[i,j,prm]   = abs(np.corrcoef(sig1[idx], sig2[idx])[0,1])

#%% calculate connectivity metric
PSI     = np.zeros((Nosc, Nosc, State))
R       = np.zeros((Nosc, Nosc, State))

p_PSI   = np.zeros((Nosc, Nosc, State))
p_R     = np.zeros((Nosc, Nosc, State))

for S in range(State):
    if S == 0:
        idx = np.where(idx_st1==True)[0]
    elif S == 1:
        idx = np.where(idx_st2==True)[0]
    else:
        idx = np.where(idx_st3==True)[0]
    
    N = len(idx)
    
    for i in range(Nosc):
        for j in range(Nosc):
            dif          = theta[idx,j] - theta[idx,i]
            psi_tmp      = abs(np.sum(np.exp(1j * dif)))/N
            PSI[i,j,S]   = psi_tmp
            p_PSI[i,j,S] = surrogate_pvals(psi_tmp, PSI_prm[i,j,:], tail='one')
            ########
            sig1         = np.cos(theta[idx, i])
            sig2         = np.cos(theta[idx, j])
            r_tmp        = np.corrcoef(sig1, sig2)
            R[j,i,S]     = abs(r_tmp[0,1])
            p_R[j,i,S]   = surrogate_pvals(abs(r_tmp[0,1]), R_prm[i,j,:], tail='one')
            
h_PSI, p_PSI = fdr(p_PSI, 0.05)
h_R,   p_R   = fdr(p_R,   0.05)
#%%
vmin    = 0
vmax    = 0.7#Kest.mean() + Kest.std()

fig = plt.figure(constrained_layout = False, figsize=(10, 6));
plt.subplots_adjust(wspace=0.3, hspace=0.3);
gs  = fig.add_gridspec(2, State)
    
for state in range(State):
    connect_R                 = deepcopy(h_R[:,:,state])
    R_tmp                     = deepcopy(R[:,:,state])
    R_tmp[connect_R==False]   = 0
    
    connect_psi               = deepcopy(h_PSI[:,:,state])
    PSI_tmp                   = deepcopy(PSI[:,:,state])
    PSI_tmp[connect_psi==False] = 0
    
    
    #### graph estimated by correlation matrix
    ax1 = fig.add_subplot(gs[0, state])
    vis_undirected_graph(np.triu(R_tmp.T), vmin, vmax)
    ax1.set_title('Segment %d\n (corr.)'%(state+1), fontsize =12)
    
    #### graph estimated by PSI
    ax2 = fig.add_subplot(gs[1, state])
    vis_undirected_graph(np.triu(PSI_tmp.T), vmin, vmax)
    ax2.set_title('Segment %d\n (PSI)'%(state+1), fontsize =12)

plt.savefig(fig_save_dir + 'estimated_graph_conventional.png')
plt.savefig(fig_save_dir + 'estimated_graph_conventional.svg')
plt.show()