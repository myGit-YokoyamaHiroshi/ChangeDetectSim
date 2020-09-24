# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 00:58:16 2020

@author: H.yokoyama
"""

from IPython import get_ipython
from copy import deepcopy, copy
get_ipython().magic('reset -sf')


from scipy.stats  import invgamma
from scipy.linalg import block_diag
from numpy.random import *
from numpy.linalg import inv, pinv
from numpy.matlib import repmat
from scipy.stats import zscore
from scipy import signal as sig

from sklearn.datasets import make_spd_matrix

import numpy as np
import matplotlib.pylab as plt
import os

#%%
def exponential_cov(x, y, params=[1, 1]):
    return params[0] * np.exp( -0.5 * params[1] * np.subtract.outer(x, y)**2)

def inv_use_cholensky(M):
    L     = np.linalg.cholesky(M)
    L_inv = np.linalg.inv(L)
    M_inv = np.dot(L_inv.T, L_inv)
    
    return M_inv

def make_lagged_X(x, T, N, P, confounder):
    tmpX  = deepcopy(np.flipud(x))
    
    for i in range(0, P):
        idx   = np.arange(i, i+T, 1)
        if i == 0:
            X = deepcopy(tmpX[idx, :])
        else:
            X = np.concatenate((X,tmpX[idx, :]), axis=1)
        
    del tmpX
    
    if confounder == True:
        X = np.concatenate((X, np.ones((T, 1))), axis=1)
    
    return X
    
def make_Dx(X, T, N, P, confounder):
    Dx   = np.zeros((N*T,  N*(N*P+1)*T), dtype=float)
    
    cnt = 0;
    order_index = np.sort(repmat(np.arange(0, T), 1, N))
    
    for i in order_index[0,:]:#range(1, (X.shape[0])*(X.shape[1]-1) ):
        tmp_x = deepcopy(X[i, :]).reshape(-1)
        
        idx = np.arange(cnt*(N*P+confounder), (cnt+1)*(N*P+confounder), 1)
            
        Dx[cnt, idx] = tmp_x
        cnt += 1
        
    return Dx

def update_coeff(X, Y, mu_beta, Kb, sigma0, N, P, confounder):
    #### Get prior parameters 
    mu_beta0 = deepcopy(mu_beta)
    Kb0      = deepcopy(Kb)
    Kb0_inv  = inv_use_cholensky(Kb0)
    
    #### Select coefficient with prior distribution 
    B = np.random.multivariate_normal(mu_beta0, Kb0)
    #### Make Dx
    T = X.shape[0]
    
    Dx = make_Dx(X, T, N, P, confounder)
    ############# Estimate posterior distribution #############################
    tmp     = sigma0 * np.eye(N*T) + Dx @ Kb0 @ Dx.T 
    tmp_inv = inv_use_cholensky(tmp)
    
    KbDx = Kb0 @ Dx.T
    DxKb = Dx @ Kb0
    
    Yv    = Y.reshape(-1) - np.dot(Dx, B).reshape(-1)
    #### update covariance
    Kb      = Kb0 - KbDx @ tmp_inv @ DxKb 
    #### update mean
    mu_beta = mu_beta0 + KbDx @ tmp_inv @ Yv 
    ############# Calculate log-likelihood ####################################
    # Computes the log of the marginal likelihood without constant term.
    Kb_inv  = inv_use_cholensky(deepcopy(Kb))
    B_hat   = np.random.multivariate_normal(mu_beta, Kb)
    
    E_err   = Yv  @ Yv
    E_beta  = (B_hat - mu_beta) @ Kb_inv @ (B_hat - mu_beta)
    
    
    dummy, logKb    = np.linalg.slogdet(Kb)
    dummy, logSigma = np.linalg.slogdet((sigma0 * np.eye(N*T)))
    
    loglike = ( - E_err - E_beta )/2
    
    ############# changing ratio ##############################################
    # KL divergense based
    term1 = np.log(np.trace(Kb_inv)/np.trace(Kb0_inv))
    term2 = np.trace(Kb_inv @ Kb @ Kb_inv)/np.trace(Kb_inv)
    term3 = np.trace(Kb0_inv @ Kb @ Kb0_inv)/np.trace(Kb0_inv)
    
    change_ratio = 0.5 * term1 - 0.5 * (term2 - term3)
    # # hotelling's t-square method
    # change_ratio = (B_hat - mu_beta0) @ Kb0_inv @ (B_hat - mu_beta0)
    ###########################################################################
    return mu_beta, Kb, loglike, change_ratio
