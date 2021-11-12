from IPython import get_ipython
from copy import deepcopy, copy
get_ipython().magic('reset -sf')
#get_ipython().magic('cls')

import os
current_path = os.path.dirname(__file__)
os.chdir(current_path)


import sys 
sys.path.append(current_path)

fig_save_dir = current_path + '\\figures\\VAR\\' 
if os.path.exists(fig_save_dir)==False:  # Make the directory for figures
    os.makedirs(fig_save_dir)

param_path = current_path + '\\save_data\\param_sim\\'# Set path of directory where the dataset of parameter settings are saved.
    
import matplotlib.pylab as plt
import matplotlib.gridspec as gridspec
plt.rcParams['font.family']      = 'Arial'#
plt.rcParams['mathtext.fontset'] = 'stix' # math font setting
plt.rcParams["font.size"]        = 28 # Font size

#%%
# from my_modules.my_dynamical_bayes import *
from my_modules.my_oscillator_model import *
from my_modules.ar_kalman_connectivity import AR_Kalman

from my_modules.my_graph_visualization import *
from numpy.random import *
import numpy as np
import joblib


#%%
def calc_cp_est_precision(est_cp_index, true_cp_index):
    TP    = np.sum( (true_cp_index==1) & (est_cp_index == 1) ) # Num of Hit
    FN    = np.sum( (true_cp_index==1) & (est_cp_index == 0) ) # Num of Miss
    
    TN    = np.sum( (true_cp_index==0) & (est_cp_index == 0) ) # Num of Correct Rejection
    FP    = np.sum( (true_cp_index==0) & (est_cp_index == 1) ) # Num of False Alarm
    
    P      = TP + FN
    N      = TN + FP

    #######
    #### F measure
    TPR    = TP/P
    FPR    = FP/N

    return TPR, FPR


def calc_ROCcurve(true_labels, cp_score, th_max, bins = 0.01):
    th  = np.arange(0,1.0 + bins, bins) * th_max#np.array([weight * 1 for weight in np.arange(0,1.0, bins)]) * thresholds
    
    ROC = np.zeros((len(th),2))
    
    for i in range(len(th)):
        est_labels           = np.zeros(cp_score.shape)
        idx_posi             = np.where(cp_score>=th[i])[0]
        est_labels[idx_posi] = 1
        
        tmp_tpr, tmp_fpr  = calc_cp_est_precision(est_labels, true_labels)
        ROC[i, 0] = tmp_fpr
        ROC[i, 1] = tmp_tpr
    
    idx_th    = np.argmax(np.sqrt((1-ROC[:,0])**2 + ROC[:,1]**2))
    threshold = th[idx_th]
    
    return ROC, threshold, idx_th

def calc_AUC(ROC):
    ROC = np.concatenate((ROC, np.zeros((1,2))), axis=0)
    ROC = np.flipud(ROC)
    
    AUC = 0.
    for i in range(ROC.shape[0]-1):    
        AUC += (ROC[i+1,0]-ROC[i,0]) * (ROC[i+1,1]+ROC[i,1])
    AUC *= 0.5
    
    return AUC

def calc_model(x, p, uc, flimits):
    model = AR_Kalman(x, p, uc, flimits)
    model.est_kalman()
    
    return model, p

def calc_error_PDC(K_tr, PDC):
    error = np.zeros(PDC.shape[2])
    for t in range(PDC.shape[2]):
        K        = PDC[:,:,t]
        K        = K - np.diag(np.diag(K))
        error[t] = np.sqrt(np.mean(abs(K_tr - K)))
    return error
#%%
name     = []
ext      = []
for file in os.listdir(param_path):
    split_str = os.path.splitext(file)
    name.append(split_str[0])
    ext.append(split_str[1])
    
    print(split_str)


##### Load the parameter settings
fullpath       = param_path + name[0] + ext[0]
param_dict     = np.load(fullpath, encoding='ASCII', allow_pickle='True').item()

Nosc           = param_dict['Nosc']
h              = param_dict['h']
Nt_cond        = param_dict['Nt_cond']
State          = param_dict['State']
omega          = param_dict['omega']
K1_tr          = param_dict['K1_tr']
K2_tr          = param_dict['K2_tr']
K_tr           = param_dict['K_tr']

theta_init     = param_dict['theta_init']
theta          = param_dict['theta'] 
dtheta         = param_dict['dtheta'] 
phase_dynamics = param_dict['phase_dynamics'] 
    
del param_dict
#%%

Ncond = len(Nt_cond)
#%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
vmin  = -0.6
vmax  =  0.6
err   = []
for i in range(Ncond):    
    #%% Bayesian estimation and change point detection
    Nt = Nt_cond[i]
    
    fs = 1/h
    
    x  = deepcopy(np.cos(theta[i]))
    t  = np.arange(Nt)/fs
    
    cnt = 1
    flimits = np.array([3.5, 5])
    
    #%%
    P_candi  = np.arange(1, 11, 1)
    UC_candi =  np.array([10**-i for i in range(0,8)])
    
    criteria = np.zeros((len(UC_candi), len(P_candi)))
    
    flimits  = np.array([2, 5])
    
    for n, uc in zip(np.arange(len(UC_candi)), UC_candi):
        #%% Calculate time-variant AR coefficients for each model order P
        processed  = joblib.Parallel(n_jobs=-1, verbose=5)(joblib.delayed(calc_model)(x, p, uc, flimits) for p in P_candi)
        processed.sort(key=lambda x: x[1]) # sort the output list according to the model order
        tmp_result = [tmp[0] for tmp in processed]
        #%% Determine the optimal model order
        for m in range(len(P_candi)):
            tmp          = tmp_result[m]
            k            = tmp.Kb0.shape[0]
            loglike      = tmp.loglike.mean()
            
            criteria[n,m] = loglike
        print(uc)
    #%%
    c_best = criteria.reshape(-1).max()
    for n, uc in zip(np.arange(len(UC_candi)), UC_candi):
        for m, p in zip(np.arange(len(P_candi)), P_candi):
            if criteria[n,m]==c_best:
                UC = uc
                P  = p
                break
    #%%
    model, _ = calc_model(x, P, UC, flimits)
    #%%
    Changes = model.Changes
    L       = model.loglike
    y_hat   = model.y_hat
    PDC     = model.PDC
    
    Time = t[P:]
    
    idx_cp_tr             = np.where(Time>=(Nt/fs)/3)[0][0]
    true_label            = np.zeros(Time.shape)
    true_label[idx_cp_tr] = 1
    
    C        = Changes
    #%%
    fig_path = fig_save_dir + 'Nt_%d'%Nt_cond[i] + '\\' 
    if os.path.exists(fig_path)==False:  # Make the directory for figures
            os.makedirs(fig_path)
    #%%
    ROC, Cth, idx_th = calc_ROCcurve(true_label, C, C.mean() + 5 * C.std(), bins = 0.01)
    AUC = calc_AUC(ROC)
    
    if Cth >= C.mean():
        Nsd = (Cth-C.mean())/C.std() 
        # Cth = C.mean() + Nsd * C.std() 
        str_threshold = ' Threshold ($\zeta^* = \mu + %.3f SD $)'%Nsd
    else:
        Nsd = 0
        str_threshold = ' Threshold ($\zeta^* <~\mu$)'
    
    fig = plt.figure()
    ax  = fig.add_subplot(111)
    plt.plot(ROC[:,0], ROC[:,1], '-o', linewidth=3, label = 'ROC curve (AUC = $%.3f$)'%AUC, zorder=0)
    plt.scatter(ROC[idx_th,0], ROC[idx_th,1], c='r', s=100, label = str_threshold)
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.xticks(np.array([0, 0.5, 1]))
    plt.yticks(np.array([0, 0.5, 1]))
    
    ax.set_aspect('equal')
    plt.legend(bbox_to_anchor=(-0.15, -.7), loc='lower left',  borderaxespad=0, fontsize=18, frameon=True)
    
    plt.savefig(fig_path + 'ROC.png', bbox_inches="tight")
    plt.savefig(fig_path + 'ROC.svg', bbox_inches="tight")
    plt.savefig(fig_path + 'ROC.eps', bbox_inches="tight")
    plt.show()
    
            
    #%%
    idx     = np.where(Changes >= Cth)[0]         
    fig = plt.figure(figsize=(20, 4))
    
    plt.plot(Time, C, label = 'Change point score');
    plt.scatter(Time[idx], C[idx], marker = 'o', c = 'red', label = 'Change point (KL div > $\zeta^*$)');
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0, fontsize=26, frameon=True)
    plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left',  borderaxespad=0, fontsize=26, frameon=True)
    plt.xticks(np.arange(0, Nt/fs + h, int((Nt/fs)/3))) 
    plt.xlabel('time (s)')
    plt.ylabel('Change point score\n(Hotelling T2)')
    
    
    plt.subplots_adjust(right = 0.7)
    plt.grid()
    # plt.ylim(-10.0, 110.0)
    plt.savefig(fig_path + 'changing_point.png', bbox_inches="tight")
    plt.savefig(fig_path + 'changing_point.svg', bbox_inches="tight")
    plt.savefig(fig_path + 'changing_point.eps', bbox_inches="tight")
    plt.show()
    #%%

    
    # #%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # ####### Phase response curve #################################################
    # ##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    idx_st1 = Time<(int((Nt/fs)/3))
    idx_st2 = (Time>=int((Nt/fs)/3)) & (Time<int((Nt/fs)*2/3))
    idx_st3 = (Time>=int((Nt/fs)*2/3))
    phi     = theta[i][0:-P,:]
    dphi    = dtheta[i][P:,:]
    
    Kest_ave1 = np.median(PDC[:,:,idx_st1], axis=2)
    Kest_ave2 = np.median(PDC[:,:,idx_st2], axis=2)
    Kest_ave3 = np.median(PDC[:,:,idx_st3], axis=2)
    
    Kest_ave  = np.concatenate((Kest_ave1[:,:,np.newaxis], 
                                Kest_ave2[:,:,np.newaxis],
                                Kest_ave3[:,:,np.newaxis]), axis=2)

    #%%
    plot_graph(K_tr, Kest_ave, vmin, vmax)
    plt.savefig(fig_path + 'estimated_graph.png', bbox_inches="tight")
    plt.savefig(fig_path + 'estimated_graph.svg', bbox_inches="tight")
    plt.savefig(fig_path + 'estimated_graph.eps', bbox_inches="tight")
    plt.show()
    
    #%%
    idx_sgm = [idx_st1,idx_st2,idx_st3]
    
    tmp_err = np.array([np.median(calc_error_PDC(K_tr[:,:,sgm], PDC[:, :, idx_sgm[sgm]])) for sgm in range(State)])
    err.append(tmp_err)
    #%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
import pandas as pd
import seaborn as sns
df = pd.DataFrame({
    'segment 1': np.array(err)[:, 0],
    'segment 2': np.array(err)[:, 1],
    'segment 3': np.array(err)[:, 2]
})


fig = plt.figure(figsize=(4, 4))
ax = fig.add_subplot(1, 1, 1)
sns.violinplot(data=df, jitter=True, color='gray', ax=ax, zorder=0)
[ax.scatter(np.arange(0,3), err[cnd], label='$N_t = %d$'%Nt_cond[cnd]) for cnd in range(Ncond)]; 
# ax.bar(np.arange(1,4), np.array(err).mean(axis=0), color='b', alpha=0.3, zorder=0)
plt.xticks(rotation=45)
plt.title('VAR method')
ax.set_ylabel('median of error ')
ax.set_ylim(-0.1, 1)
plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left',  borderaxespad=0, fontsize=18, frameon=True)

plt.savefig(fig_save_dir + 'error.png', bbox_inches="tight")
plt.savefig(fig_save_dir + 'error.svg', bbox_inches="tight")
plt.savefig(fig_save_dir + 'error.eps', bbox_inches="tight")

plt.show()