# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 10:10:28 2020

@author: yokoyama
"""
from copy import deepcopy
from PyEMD import EMD, EMD_matlab
from fbm import FBM
from scipy import signal as sig
from joblib import Parallel, delayed
from hurst import compute_Hc, random_walk
import numpy as np
import matplotlib.pylab as plt
#%%
EMD.splineKind    = 'cubic'
EMD.nbsym         = 2
EMD.reduceScale   = 1.
EMD.scaleFactor   = 100
EMD.maxIteration  = 100
EMD.stop1         = 0.05
EMD.stop2         = 0.5
EMD.stop3         = 0.05
EMD.DTYPE         = np.float64
EMD.MAX_ITERATION = 500
EMD.extrema_detection = "parabol"


#%%
def EMD_denosing_fGn_thresholding(signal, t, h_exp, nofsifts, threshold):
    """
    Python implementation of EMD-thresholding method, proposed by Flandrin et al. (2015)
    In this script, the parameter settings to estimate the threshold is applied, 
    assuming the "hurst exponent H = 0.5, and confidence interval = 95%", based on the reference paper .
    
    %signal       : Noisy signal
    %t            : time stamp of each samples
    %h_exp        : hurst exponent (e.g. hurst = 0.5)
    %nofsifts     : Number of sifting iterations (it should take a value
    %               between 5 and 10.)
    %
    % REFERECIES:
    % [1] Molla, M. K. I., Islam, M. R., Tanaka, T., & Rutkowski, T. M. (2012). 
    %     Artifact suppression from EEG signals using data adaptive time domain filtering. 
    %     Neurocomputing, 97, 297-308.        
    % [2] Flandrin, P., Gonçalves, P., & Rilling, G. (2005). 
    %     EMD equivalent filter banks, from interpretation to applications. 
    %     In Hilbert-Huang transform and its applications (pp. 57-74).
    % [3] Flandrin, P., Goncalves, P., & Rilling, G. (2004, September). 
    %     Detrending and denoising with empirical mode decompositions. 
    %     In 2004 12th European Signal Processing Conference (pp. 1581-1584). IEEE.
    """
    #%% ##### Apply Empirical mode decomposition (full) 
    X          = deepcopy(signal)
    emd        = EMD()    
    emd.FIXE   = nofsifts
    emd.FIXE_H = nofsifts
    IMFs       = emd(X, t, max_imf = len(threshold));

    #%% ##### Thresholding 
    h_hat     = np.zeros(IMFs.shape)
    for i in range(1, IMFs.shape[0]):
        c = deepcopy(IMFs[i, :])
        E = np.log2(np.median(abs(c)**2))
        if E > threshold[i]:
            h_hat[i,:] = c
    #%% ##### Make partial reconstructed sig
    sig_denoise = np.sum(h_hat[1:, :], axis=0)
    
    return sig_denoise
################################################################################
def EMD_denosing_Flandrin(signal, t, nofsifts, detrend = 0):
    """
    Python implementation of EMD-thresholding method, proposed by Flandrin et al. (2015)
    In this script, the parameter settings to estimate the threshold is applied, 
    assuming the "hurst exponent H = 0.5, and confidence interval = 95%", based on the reference paper .
    
    %signal       : Noisy signal
    %t            : time stamp of each samples
    % REFERECIES:
    % [1] Flandrin, P., Gonçalves, P., & Rilling, G. (2005). 
    %     EMD equivalent filter banks, from interpretation to applications. 
    %     In Hilbert-Huang transform and its applications (pp. 57-74).
    % [2] Flandrin, P., Goncalves, P., & Rilling, G. (2004, September). 
    %     Detrending and denoising with empirical mode decompositions. 
    %     In 2004 12th European Signal Processing Conference (pp. 1581-1584). IEEE.
    """
    #%% setup parameters
    a_h  =  0.474#0.460#
    b_h  = -2.449#-1.919#
    Beta =  0.719
    H    =  0.5 # hurst exponent
    
    # a_h  =  0.495
    # b_h  = -1.833
    # Beta =  1.025
    # H    =  0.8 # hurst exponent
    
    rho  = 2.01 + 0.2*(H-0.5) + 0.12*(H-0.5)**2;
    #%% ##### Estimate IMFs for signal + noise mixture
    X          = deepcopy(signal )
    
    emd        = EMD()    
    emd.FIXE   = nofsifts
    emd.FIXE_H = nofsifts
    IMFs       = emd(X, t);
    
    Num        = IMFs.shape[0]
    #%% ###############################################################
    #### Estimate the noise energy and confidence interval
    Wh         = np.zeros(Num)
    Wh[0]      = np.median(abs(IMFs[0,:])**2)
    
    k          = np.arange(1, Num+1, 1)
    C          = Wh[0]/Beta
    for i in range(1, len(Wh)):
        Wh[i]  = C * rho**(-2*(1-H)*k[i])
        
    CI        =  2**(a_h * k + b_h) + np.log2(Wh)# confidence interval of noise
    threshold = CI 
    ###### Thresholding 
    h_hat     = np.zeros(IMFs.shape)
    for i in range(1, IMFs.shape[0]):
        c = deepcopy(IMFs[i, :])
        E = np.log2(np.median(abs(c)**2))
        if E >= threshold[i]:
            h_hat[i,:] = c
    #%% detrend
    if detrend == 1:
        means      = np.zeros(Num)
        for i in range(1, Num+1):
            means[i-1] = np.mean(np.sum(IMFs[:i, :], axis=0))
            
        means      = means/means.sum()
        diff_means = np.concatenate( (np.array([0]), np.diff(np.sign(means))))
        idx        = np.where((diff_means != 0) & (abs(means) >= 0.05))[0]#np.where(means <= -0.05)[0]#
        
        if len(idx) != 0:
            idx = idx[idx!=0].min()
            h_hat[idx:,:] = 0
    #%% ##### Make partial reconstructed sig
    sig_denoise = np.sum(h_hat, axis=0)
    
    return sig_denoise
###############################################################################
def EMD_denosing_hurst_threshold(signal, t, nofsifts, hexp_th):
    ##### Apply Empirical mode decomposition (full) 
    X          = deepcopy(signal)
    emd        = EMD()    
    emd.FIXE   = nofsifts
    # emd.FIXE_H = nofsifts
    IMFs       = emd(X, t)#, max_imf = 20);
    
    Num        = IMFs.shape[0]
    #%%
    h_hat      = np.zeros(IMFs.shape)
    for i in range(1, IMFs.shape[0]):
        tmp_IMFs = deepcopy(IMFs[i, :])
        H, c, data = compute_Hc(tmp_IMFs, kind =  'random_walk', simplified=False)
        # print(abs(H))
        if (abs(H) <= hexp_th):
            h_hat[i,:] = tmp_IMFs
    #%% ##### Make partial reconstructed sig
    sig_denoise = np.sum(h_hat, axis=0)
    
    return sig_denoise



