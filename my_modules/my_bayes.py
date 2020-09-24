# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 09:54:06 2020

@author: yokoyama
"""

from copy import deepcopy
from numpy.matlib import repmat
import numpy as np

#%%
def exponential_cov(x, y, params=[1, 1]):
    return params[0] * np.exp( -0.5 * params[1] * np.subtract.outer(x, y)**2)

##################################################
def KL_div(mu1, mu2, K1, K2):
    D      = K1.shape[0]
    K1_inv = inv_use_cholensky(K1)
    K2_inv = inv_use_cholensky(K2)
    
    sgn1, logdetK1     = np.linalg.slogdet(K1)
    sgn1, logdetK2     = np.linalg.slogdet(K2)
    
    sgn1, logdetK1_inv = np.linalg.slogdet(K1_inv)
    sgn1, logdetK2_inv = np.linalg.slogdet(K2_inv)
    
    term1 = logdetK1_inv - logdetK2_inv
    term2 = np.trace(K2_inv @ K1) - D
    term3 = (mu2 - mu1).T @ K2_inv @ (mu2 - mu1)
    
    KL    = 0.5 * (term1 + term2 + term3)
    Norm  = 0.5 * (logdetK1 + D * (1 + np.log(2*np.pi)))
    return KL

def likelihood_anomaly(y_hat, y, sigma0):
    p     = sigma0.shape[0]
    n     = y_hat.shape[0]
    
    Sigm0 = np.diag(sigma0)
    
    if np.trace(Sigm0)==0:
        Lamb0 = np.diag(np.zeros(sigma0.shape))
    else:
        Lamb0 = inv_use_cholensky(Sigm0)
    
    err   = y_hat - y
    score = ((err  @ Lamb0)**2)/(2*np.diag(Lamb0)) - 0.5 * p * n *np.log(np.diag(Lamb0)/(2*np.pi))
    
    # score = score/score.max()
    
    return score.mean()

def Mahalanobis_anomaly(y_new, y_pre, sigma_pre):
    Sigm     = np.diag(sigma_pre)
    
    if np.trace(Sigm)==0:
        Lamb     = np.diag(np.zeros(sigma_pre.shape))
    else:
        Lamb     = inv_use_cholensky(Sigm)
    
    lamb = np.diag(Lamb)
    err  = (y_new - y_pre)
    
    
    a = ((err  @ Lamb)**2)/(2*lamb) # = err**2/sigma
    a = a.mean()
    # a = ((a-a.min())/(np.max(a)-np.min(a))).mean()
    
    return a

def BIC(y_hat, y, T, k):
    err  = (y_hat - y)
    
    a = 2*np.log((err**2).mean()) + k * np.log(T)
    a = a.mean()
    # a = (a/np.max(a)).mean()
    
    return a
#############################################

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
    
    if X.shape[0] == 1:
        order_index = np.sort(repmat(np.arange(0, T), 1, N))
        order_index = order_index[0,:]
    else:
        order_index = np.arange(0, X.shape[0])
        
    for i in order_index:#range(1, (X.shape[0])*(X.shape[1]-1) ):
        tmp_x = deepcopy(X[i, :]).reshape(-1)
        
        idx = np.arange(cnt*(N*P+confounder), (cnt+1)*(N*P+confounder), 1)
            
        Dx[cnt, idx] = tmp_x
        cnt += 1
        
    return Dx

def update_coeff(X, Y, mu_beta, Kb, sigma0, T, N, P, confounder):
    #### Get prior parameters 
    mu_beta0 = deepcopy(mu_beta)
    Kb0      = deepcopy(Kb)
    Kb0_inv  = inv_use_cholensky(Kb0)
    
    #### Select coefficient with prior distribution 
    B = np.random.multivariate_normal(mu_beta0, Kb0)
    #### Make Dx    
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
    ############ Ide's method
    # term1 = np.log(np.diag(Kb_inv)/np.diag(Kb0_inv))
    
    # term2 = np.diag(Kb0_inv @ Kb @ Kb0_inv)/np.diag(Kb0_inv)
    
    # term3 = (mu_beta - mu_beta0)**2 @ Kb # np.diag(Kb_inv @ Kb @ Kb_inv)/np.diag(Kb_inv) # 
    
    # change_ratio = np.sum (0.5 * (term1 + term2 + term3 -1)) # np.sum(0.5 * term1 - 0.5 * (term2 - term3) )  # 
    
    ########## KL divergence
    change_ratio = KL_div(mu_beta0, mu_beta, Kb0, Kb)#KL_div(mu_beta, mu_beta0, Kb, Kb0) 
    ########## BIC information
    # Ndim         = mu_beta.shape[0]
    # Nt           = Y.shape[0]
    # y_hat        = Dx @ mu_beta
    # change_ratio = BIC(y_hat.reshape(-1), Y.reshape(-1), Nt, Ndim)
    ###########################################################################
    return mu_beta, Kb, loglike, change_ratio