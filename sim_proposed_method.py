from IPython import get_ipython
from copy import deepcopy, copy
get_ipython().magic('reset -sf')
#get_ipython().magic('cls')

import os
import sys
current_path = os.path.dirname(__file__)
os.chdir(current_path)
sys.path.append(current_path)

fig_save_dir = current_path + '\\figures\\our_method\\' 
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
from my_modules.my_dynamical_bayes_mod import my_Bayesian_CP

from my_modules.my_graph_visualization import *
from scipy.stats import zscore
from scipy.io import savemat
from numpy.random import *
import numpy as np
import glob

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

def calc_error(K_tr, Kest, Nosc):
    error = np.zeros(Kest.shape[0])
    for t in range(Kest.shape[0]):
        K        = Kest[t,:].reshape(Nosc, Nosc)
        K        = K - np.diag(np.diag(K))
        # error[t] = np.sqrt(np.mean((K_tr - K)**2))
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
#%% Generate synthetic data
if (len(name) == 0) & (len(ext) == 0):
    ############# set parameters
    Nosc          = 3
    h             = 0.01
    Nt_cond       = np.array([1500, 3000, 4500, 6000])
    State         = 3
    omega         = 2 * np.pi * np.array([4.31, 4.48, 3.7])
    
    Ncond         = len(Nt_cond)
    K1_tr         = np.zeros((Nosc, Nosc, State))
    K2_tr         = np.zeros((Nosc, Nosc, State))
    K_tr          = np.zeros((Nosc, Nosc, State))
    #%%
    for st in range(State):
        if st == 0:
            K1_tr[:,:,st] = np.array([[0.00, 0.00, 0.40],
                                      [0.00, 0.00, 0.40],
                                      [0.00, 0.00, 0.00]])
            
            K2_tr[:,:,st] = np.array([[0.00, 0.00, 0.40],
                                      [0.00, 0.00, 0.40],
                                      [0.00, 0.00, 0.00]])
        elif st == 1:
            K1_tr[:,:,st] = np.array([[0.00, 0.40, 0.40],
                                      [0.40, 0.00, 0.40],
                                      [0.40, 0.40, 0.00]])
            
            K2_tr[:,:,st] = np.array([[0.00, 0.40, 0.40],
                                      [0.40, 0.00, 0.40],
                                      [0.40, 0.40, 0.00]])
        else:
            K1_tr[:,:,st] = np.array([[0.00, 0.40, 0.40],
                                      [0.40, 0.00, 0.40],
                                      [0.40, 0.40, 0.00]])
            
            K2_tr[:,:,st] = np.array([[0.00, 0.40, 0.40],
                                      [0.40, 0.00, 0.40],
                                      [0.40, 0.40, 0.00]])
            
        K_tr[:,:,st] = np.sqrt(K1_tr[:,:,st]**2 + K2_tr[:,:,st]**2)
    #%%
    theta_init     = np.array([0, np.pi, np.pi/2])
    theta          = []
    dtheta         = []
    phase_dynamics = []
    #%% solve stochastic differential equation using Eular-Maruyama Method
    fs    = 1/h
    vmin  = -0.6
    vmax  =  0.6
    
    for i in range(Ncond):
        fig_path = fig_save_dir + 'Nt_%d'%Nt_cond[i] + '\\' 
        if os.path.exists(fig_path)==False:  # Make the directory for figures
            os.makedirs(fig_path)
        
        Nt    = Nt_cond[i]
        noise = np.array([0.001, 0.001, 0.01])
        tmp_theta, tmp_dtheta, tmp_phase_dynamics = generate_sythetic_phase(theta_init, omega, K1_tr, K2_tr, Nt, Nosc, noise, h)
        
        t = np.arange(tmp_theta.shape[0])/fs
        
        
        plot_synthetic_data(t, tmp_dtheta, fs, K1_tr, K2_tr, K_tr, vmin, vmax)
        plt.savefig(fig_path + 'param_setting.png', bbox_inches="tight")
        plt.savefig(fig_path + 'param_setting.svg', bbox_inches="tight")
        plt.savefig(fig_path + 'param_setting.eps', bbox_inches="tight")
        plt.show()
        
        theta.append(tmp_theta)
        dtheta.append(tmp_dtheta)
        phase_dynamics.append(tmp_phase_dynamics)

        matpath = current_path + '\\mat\\' + 'Nt_%d'%Nt_cond[i] + '\\'
        if os.path.exists(matpath )==False:  # Make the directory for figures
            os.makedirs(matpath)
        savemat(matpath+"phase_data.mat", {'theta':tmp_theta, 'dtheta':tmp_dtheta, 't':t, 'h':h, 'fs':1/h})        
    #%% ############# save_data
    param_dict                   = {} 
    param_dict['Nosc']           = Nosc
    param_dict['h']              = h
    param_dict['Nt_cond']        = Nt_cond
    param_dict['State']          = State
    param_dict['omega']          = omega
    param_dict['K1_tr']          = K1_tr
    param_dict['K2_tr']          = K2_tr
    param_dict['K_tr']           = K_tr
    
    param_dict['theta_init']     = theta_init     # initial value of phase
    param_dict['theta']          = theta
    param_dict['dtheta']         = dtheta
    param_dict['phase_dynamics'] = phase_dynamics

    save_name   = 'Sim_param' 
    fullpath_save   = param_path + save_name 
    np.save(fullpath_save, param_dict)
    #%%
else:
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
        
P = 1 # order of Forier series
T = 1 # Time steps for sequential bayesian updates

noise_param = 1E-4 # covariance of process noise
prec_param  = 1/noise_param # precision parameter, cov(process noise) = 1/prec_param

# PRC_all



for st in range(State):
    beta_true = np.zeros((1, Nosc*Nosc, 2))
    beta_true[:,:,0] = K1_tr[:,:,st].reshape(-1)
    beta_true[:,:,1] = K2_tr[:,:,st].reshape(-1)

    tmp_prc, phi_dlt_plt = reconstruct_phase_response_curve(beta_true, omega[np.newaxis,:], Nosc)
    if st == 0:
        PRC_true = tmp_prc
    else:
        PRC_true = np.concatenate((PRC_true, tmp_prc), axis=0)
     
#%%
err = []
for i in range(Ncond):    
    #%% Bayesian estimation and change point detection
    fs = 1/h
    
    Nt = Nt_cond[i]
     
    
    x  = deepcopy(theta[i])
    t  = np.arange(Nt)/fs
    
    cnt = 1
    bayes_cp = my_Bayesian_CP(x, P, T, h, prec_param)
    bayes_cp.est_dynamical_oscillator_1st_order_fourier()
    
    beta    = bayes_cp.beta
    OMEGA   = bayes_cp.omega
    Changes = bayes_cp.Changes
    L       = bayes_cp.loglike
    y_hat   = bayes_cp.y_hat
    
    Time  = t[T:]
    Kest  = np.sqrt(np.sum(beta**2, axis=2))
    err_K = calc_error(K_tr, Kest, Nosc)
    
    
    idx_cp_tr             = np.where(Time>=(Nt/fs)/3)[0][0]
    true_label            = np.zeros(Time.shape)
    true_label[idx_cp_tr] = 1
    
    #%% make direcotry 
    fig_path = fig_save_dir + 'Nt_%d'%Nt_cond[i] + '\\' 
    if os.path.exists(fig_path)==False:  # Make the directory for figures
        os.makedirs(fig_path)
    #%%
    C = Changes

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
    plt.plot(ROC[:,0], ROC[:,1], linewidth=3, label = 'ROC curve (AUC = $%.3f$)'%AUC, zorder=0)
    plt.scatter(ROC[idx_th,0], ROC[idx_th,1], c='k', s=100, label = str_threshold)
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.xticks(np.array([0, 0.5, 1]))
    plt.yticks(np.array([0, 0.5, 1]))
    
    ax.set_aspect('equal')
    plt.legend(bbox_to_anchor=(-0.42, -.7), loc='lower left',  borderaxespad=0, fontsize=22, frameon=True)
    plt.savefig(fig_path + 'ROC.png', bbox_inches="tight")
    plt.savefig(fig_path + 'ROC.svg', bbox_inches="tight")
    plt.savefig(fig_path + 'ROC.eps', bbox_inches="tight")
    plt.show()
    
    
    print('optimal threshold is mean + %dSD'%Nsd)
    #%%
    
    idx = np.where(Changes >= Cth)[0]       
    fig = plt.figure(figsize=(20, 4))
    
    plt.plot(Time, C, label = 'Change point score');
    plt.scatter(Time[idx], C[idx], marker = 'o', c = 'red', label = 'Change point (KL div > $\zeta^*$)');
    # plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0, fontsize=26, frameon=True)
    plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left',  borderaxespad=0, fontsize=26, frameon=True)
    plt.xticks(np.arange(0, Nt/fs + h, int((Nt/fs)/3))) 
    plt.xlabel('time (s)')
    plt.ylabel('Change point score\n(KL div.)')
    
    
    plt.subplots_adjust(right = 0.7)
    plt.grid()
    # plt.ylim(-10.0, 110.0)
    plt.savefig(fig_path + 'changing_point.png', bbox_inches="tight")
    plt.savefig(fig_path + 'changing_point.svg', bbox_inches="tight")
    plt.savefig(fig_path + 'changing_point.eps', bbox_inches="tight")
    plt.show()
    #%%
    tmp_x = deepcopy(x[:-T, :])
    x_hat = np.zeros((Nt-T, Nosc))
    
    x_hat[0, :] = deepcopy(tmp_x[0, :])
    #%% plot phase dynamics
    fig = plt.figure(figsize=(20, 4))
    
    line1 = plt.plot(Time, y_hat, c = 'k', linestyle = '-', zorder = 1, label = 'pred')
    line2 = plt.plot(t, dtheta[i], c = np.array([0.5, 0.5, 0.5]), linewidth = 4,zorder = 0, label = 'true')
    plt.xticks(np.arange(0, Nt/fs + h, int((Nt/fs)/3))) 
    plt.xlabel('time (s)')
    plt.ylabel('phase velocity')
    plt.grid()
    
    handle = [line1[-1], line2[-1]]
    labels = ['pred.', 'true']
    plt.legend(handle, labels, bbox_to_anchor=(1.01, 1), loc='upper left',  borderaxespad=0, fontsize=26, frameon=True)
    plt.subplots_adjust(right = 0.7)
    plt.savefig(fig_path + 'phase_dynamics_est.png', bbox_inches="tight")
    plt.savefig(fig_path + 'phase_dynamics_est.svg', bbox_inches="tight")
    plt.savefig(fig_path + 'phase_dynamics_est.eps', bbox_inches="tight")
    plt.show()
    
    
    # #%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # ####### Visualize Network couplings #################################################
    # ##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    idx_st1 = Time<(int((Nt/fs)/3))
    idx_st2 = (Time>=int((Nt/fs)/3)) & (Time<int((Nt/fs)*2/3))
    idx_st3 = (Time>=int((Nt/fs)*2/3))
    phi     = theta[i][0:-T,:]
    dphi    = dtheta[i][T:,:]
    
    Kest_ave1 = np.median(Kest[idx_st1,:], axis=0).reshape(Nosc, Nosc)
    Kest_ave2 = np.median(Kest[idx_st2,:], axis=0).reshape(Nosc, Nosc)
    Kest_ave3 = np.median(Kest[idx_st3,:], axis=0).reshape(Nosc, Nosc)
    
    Kest_ave  = np.concatenate((Kest_ave1[:,:,np.newaxis], 
                                Kest_ave2[:,:,np.newaxis],
                                Kest_ave3[:,:,np.newaxis]), axis=2)    
    
    plot_graph(K_tr, Kest_ave, vmin, vmax)
    plt.savefig(fig_path + 'estimated_graph.png', bbox_inches="tight")
    plt.savefig(fig_path + 'estimated_graph.svg', bbox_inches="tight")
    plt.savefig(fig_path + 'estimated_graph.eps', bbox_inches="tight")
    plt.show()
    #%%
    tmp_prc = np.zeros(PRC_true.shape)
    tmp_prc1, phi_dlt_plt = reconstruct_phase_response_curve(beta[idx_st1,:,:], OMEGA[idx_st1,:], Nosc)
    tmp_prc2, phi_dlt_plt = reconstruct_phase_response_curve(beta[idx_st2,:,:], OMEGA[idx_st2,:], Nosc)
    tmp_prc3, phi_dlt_plt = reconstruct_phase_response_curve(beta[idx_st3,:,:], OMEGA[idx_st3,:], Nosc)
    #%%
    
    t_list = [10, 50, 75, 100, 400]

    fig = plt.figure(constrained_layout = False, figsize=(22, 11));
    plt.subplots_adjust(wspace=0.0, hspace=0.8);
    gs  = fig.add_gridspec(2, len(t_list))
    for t_idx in range(len(t_list)):
        plt.subplot(gs[0, t_idx])
        plt.plot(phi_dlt_plt, PRC_true[0,:,1], c='k', linewidth=10, label='true');
        
        plt.plot(phi_dlt_plt, tmp_prc1[t_list[t_idx]-1,:, 1], c='b', linewidth=5, label='pred.'); 
        
        plt.xticks([0, np.pi, 2 * np.pi], ['$0$', '$\\pi$', '$2 \\pi$'])
        plt.xlim(-0.8, 2 * np.pi + 0.8)
        plt.xlabel('$\\phi_2 - \\phi_1 $'%())
        
        if t_idx==0:
            plt.ylabel('$d \\phi_{1} / dt $')
        elif t_idx==len(t_list)-1:
            plt.yticks([])
            plt.legend(bbox_to_anchor=(1.05, 1.0), 
                       loc='upper left', borderaxespad=0, fontsize=20, frameon=True)
        else:
            plt.yticks([])
        
        plt.title('$t= %.2f~s$\n(# Iteration:%d)'%(Time[t_list[t_idx]-1], t_list[t_idx]))            
        plt.ylim(19, 29); 
        ##############
        plt.subplot(gs[1, t_idx])
        plt.plot(phi_dlt_plt, PRC_true[0,:,2], c='k', linewidth=10, label='true');
        
        plt.plot(phi_dlt_plt, tmp_prc1[t_list[t_idx]-1,:, 2], c='b', linewidth=5, label='pred.'); 
        
        plt.xticks([0, np.pi, 2 * np.pi], ['$0$', '$\\pi$', '$2 \\pi$'])
        plt.xlim(-0.8, 2 * np.pi + 0.8)
        plt.xlabel('$\\phi_3 - \\phi_1 $'%())
        if t_idx==0:
            plt.ylabel('$d \\phi_{1} / dt $')
        elif t_idx==len(t_list)-1:
            plt.yticks([])
            plt.legend(bbox_to_anchor=(1.05, 1.0), 
                       loc='upper left', borderaxespad=0, fontsize=20, frameon=True)
        else:
            plt.yticks([])
            
        plt.title('$t= %.2f~s$\n(# Iteration:%d)'%(Time[t_list[t_idx]-1], t_list[t_idx]))   
            
        plt.ylim(19, 29); 
    
    plt.savefig(fig_path + 'PRC_est_iter_segment1.png', bbox_inches="tight")
    plt.savefig(fig_path + 'PRC_est_iter_segment1.svg', bbox_inches="tight")
    plt.savefig(fig_path + 'PRC_est_iter_segment1.eps', bbox_inches="tight")
    plt.show()
    #%%
    
    tmp_prc[0,:,:] = np.median(tmp_prc1, axis=0)
    tmp_prc[1,:,:] = np.median(tmp_prc2, axis=0)
    tmp_prc[2,:,:] = np.median(tmp_prc3, axis=0)
    
    
    if i == 0:
        PRC_est      = tmp_prc[:,:,:,np.newaxis]
        Kest_ave_all = Kest_ave[:,:,:,np.newaxis]
    else:
        PRC_est      = np.concatenate((PRC_est, tmp_prc[:,:,:,np.newaxis]), axis=3)
        Kest_ave_all = np.concatenate((Kest_ave_all, Kest_ave[:,:,:,np.newaxis]), axis=3)
    #%%
    
    idx_sgm = [idx_st1,idx_st2,idx_st3]
    
    tmp_err = np.array([np.median(calc_error(K_tr[:,:,sgm], Kest[idx_sgm[sgm], :], Nosc)) for sgm in range(len(idx_sgm))])
    err.append(tmp_err)
#%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

text_list = ['A', 'B', 'C']

for st in range(State):
    legend_labels = ['$N_t = %d$'%Nt_cond[i] for i in range(len(Nt_cond))]
    fig,ax=plot_PRC(phi_dlt_plt, PRC_est[st,:,:,:], PRC_true[st,:,:], K_tr[:,:,st], vmin, vmax, Nosc, legend_labels)
    ax_pos = ax.get_position()
    fig.text(ax_pos.x1-0.25, ax_pos.y1, text_list[st], fontsize=40)
    plt.savefig(fig_save_dir + 'PRC_state%d.png'%(st+1), bbox_inches="tight")
    plt.savefig(fig_save_dir + 'PRC_state%d.svg'%(st+1), bbox_inches="tight")
    plt.savefig(fig_save_dir + 'PRC_state%d.eps'%(st+1), bbox_inches="tight")
    plt.show()

#%%
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
[ax.scatter(np.arange(0,3), err[cnd], label=legend_labels[cnd]) for cnd in range(Ncond)]; 
# ax.bar(np.arange(1,4), np.array(err).mean(axis=0), color='b', alpha=0.3, zorder=0)
plt.xticks(rotation=45)
plt.title('our method')
ax.set_ylabel('median of error ')
ax.set_ylim(-0.1, 1)
plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left',  borderaxespad=0, fontsize=18, frameon=True)

plt.savefig(fig_save_dir + 'error.png', bbox_inches="tight")
plt.savefig(fig_save_dir + 'error.svg', bbox_inches="tight")
plt.savefig(fig_save_dir + 'error.eps', bbox_inches="tight")

plt.show()