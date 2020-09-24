# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 09:54:06 2020

@author: yokoyama
"""

from copy import deepcopy
from numpy.matlib import repmat
from numpy.random import randn, rand
import numpy as np

#%%
##############################################################################
def KL_div(mu1, mu2, K1, K2):
    D      = K1.shape[0]
    K1_inv = inv_use_cholensky(K1)
    K2_inv = inv_use_cholensky(K2)
    
    sgn1, logdetK1     = np.linalg.slogdet(K1)
    sgn1, logdetK2     = np.linalg.slogdet(K2)
    
    sgn1, logdetK1_inv = np.linalg.slogdet(K1_inv)
    sgn1, logdetK2_inv = np.linalg.slogdet(K2_inv)
    
    term1 = logdetK2 - logdetK1 # = logdetK1_inv - logdetK2_inv # 
    term2 = np.trace(K2_inv @ K1) - D
    term3 = (mu2 - mu1).T @ K2_inv @ (mu2 - mu1)
    
    KL    = 0.5 * (term1 + term2 + term3) # KL div ~  cross entropy - entropy 
    EntK1 = 0.5 * (logdetK1 + D * (1 + np.log(2*np.pi))) # Entropy of Dist1
    
    # KL    = abs(KL/-EntK1)
    return KL

def JS_div(mu1, mu2, K1, K2):
    mu, K = mean_of_two_gauss(mu1, mu2, K1, K2)
    D_kl1 = KL_div(mu1, mu, K1, K)
    D_kl2 = KL_div(mu2, mu, K2, K)
    
    JS    = 0.5 * (D_kl1 + D_kl2)
    
    return JS

def mean_of_two_gauss(mu1, mu2, K1, K2):
    
    mu = 0.5    * (mu1 + mu2)
    K  = 0.5**2 * (K1  + K2)
    
    return mu, K

def my_det(Mtrx):
    sgn, logdet = np.linalg.slogdet(Mtrx)
    detMtrx     = np.exp(logdet)
    return detMtrx

def func_kuramoto(theta, K, omega):
    Nosc = theta.shape[0]
    phase_diff = theta.reshape(1, Nosc) - theta.reshape(Nosc, 1)
    phase_dynamics = omega + np.sum(K @ np.sin(phase_diff.T), axis=0) + randn(Nosc)
    return phase_dynamics

def func_oscillator_approx_fourier_series(theta, K1, K2, omega, noise_scale):
    Nosc = theta.shape[0]
    phase_diff = theta.reshape(1, Nosc) - theta.reshape(Nosc, 1)
    
    if len(K1.shape)==2:
        phase_dynamics = omega + np.sum(K1 @ np.cos(phase_diff.T), axis=0) + np.sum(K2 @ np.sin(phase_diff.T), axis=0) + noise_scale * randn(Nosc)
    elif len(K1.shape)==3:
        _,_,Norder = K1.shape
        
        for n in range(Norder):
            if n == 0:
                Cos = np.sum(K1[:,:,n] @ np.cos(phase_diff.T), axis=0)
                Sin = np.sum(K2[:,:,n] @ np.sin(phase_diff.T), axis=0)
            else:
                Cos += np.sum(K1[:,:,n] @ np.cos(n * phase_diff.T), axis=0)
                Sin += np.sum(K2[:,:,n] @ np.sin(n * phase_diff.T), axis=0)
        
        phase_dynamics = omega + Cos + Sin + noise_scale * randn(Nosc)
        
    return phase_dynamics

def runge_kutta_kuramoto(h, func, theta_now, K, omega):
    k1=func(theta_now, K, omega)#omega+K*np.sin(theta_now[::-1]-theta_now)
    
    theta4k2=theta_now+(h/2)*k1
    k2=func(theta4k2, K, omega)#omega+K*np.sin(theta4k2[::-1]-theta4k2)
    
    theta4k3=theta_now+(h/2)*k2
    k3=func(theta4k3, K, omega)#omega+K*np.sin(theta4k3[::-1]-theta4k3)
    
    theta4k4=theta_now+h*k3
    k4=func(theta4k4, K, omega)#omega+K*np.sin(theta4k4[::-1]-theta4k4)
    
    theta_next=theta_now+(h/6)*(k1+2*k2+2*k3+k4)
    theta_next=np.mod(theta_next, 2*np.pi)
    
    return theta_next

def runge_kutta_oscillator_approx_fourier_series(h, func, theta_now, K1, K2, omega, noise_scale):
    k1=func(theta_now, K1, K2, omega, noise_scale)#omega+K*np.sin(theta_now[::-1]-theta_now)
    
    theta4k2=theta_now+(h/2)*k1
    k2=func(theta4k2, K1, K2, omega, noise_scale)#omega+K*np.sin(theta4k2[::-1]-theta4k2)
    
    theta4k3=theta_now+(h/2)*k2
    k3=func(theta4k3, K1, K2, omega, noise_scale)#omega+K*np.sin(theta4k3[::-1]-theta4k3)
    
    theta4k4=theta_now+h*k3
    k4=func(theta4k4, K1, K2, omega, noise_scale)#omega+K*np.sin(theta4k4[::-1]-theta4k4)
    
    theta_next=theta_now+(h/6)*(k1+2*k2+2*k3+k4)
    theta_next=np.mod(theta_next, 2*np.pi)
    
    return theta_next

def Euler_method(y0, h, f):
    yhat = y0 + h * f
    return yhat
##############################################################################
def inv_use_cholensky(M):
    L     = np.linalg.cholesky(M)
    L_inv = np.linalg.inv(L)
    M_inv = np.dot(L_inv.T, L_inv)
    
    return M_inv


def make_Dx(X, T, N, P):
    Dx   = np.zeros((N*T,  N*(N*P+1)*T), dtype=float)
    
    cnt = 0;
    
    if X.shape[0] == 1:
        order_index = np.sort(repmat(np.arange(0, T), 1, N))
        order_index = order_index[0,:]
    else:
        order_index = np.arange(0, X.shape[0])
        
    for i in order_index:#range(1, (X.shape[0])*(X.shape[1]-1) ):
        tmp_x = deepcopy(X[i, :]).reshape(-1)
        
        idx = np.arange(cnt*(N*P+1), (cnt+1)*(N*P+1), 1)
            
        Dx[cnt, idx] = tmp_x
        cnt += 1
        
    return Dx

def update_coeff(X, Y, Dx, mu_beta, Kb, sigma0, T, N):
    #### Get prior parameters 
    mu_beta0 = deepcopy(mu_beta)
    Kb0      = deepcopy(Kb)
    # Kb0_inv  = inv_use_cholensky(Kb0)
    
    ############# Estimate posterior distribution #############################
    tmp     = sigma0 * np.eye(N*T) + Dx @ Kb0 @ Dx.T # (1/sigma0) * np.eye(N*T) + Dx @ Kb0 @ Dx.T # 
    tmp_inv = inv_use_cholensky(tmp)
    
    KbDx = Kb0 @ Dx.T
    DxKb = Dx @ Kb0
    
    Yv    = Y.reshape(-1) - np.dot(Dx, mu_beta0).reshape(-1)
    #### update covariance
    Kb      = Kb0 - KbDx @ tmp_inv @ DxKb 
    #### update mean
    mu_beta = mu_beta0 + KbDx @ tmp_inv @ Yv 
    ############# Calculate log-likelihood ####################################
    # Computes the log of the marginal likelihood without constant term.
    # Kb_inv  = inv_use_cholensky(deepcopy(Kb))

    E_err   = Yv  @ Yv
    E_beta  = (mu_beta0 - mu_beta).T @ (mu_beta0 - mu_beta)
    
    loglike = ( - E_err - E_beta )/2
    
    ############# changing ratio ##############################################
    ########## KL divergence
    change_ratio = KL_div(mu_beta0, mu_beta, Kb0, Kb) 
    ###########################################################################
    return mu_beta, Kb, loglike, change_ratio

##############################################################################
def est_dynamical_kuramoto(x, P, T, h, confounder, prec_param):   
    Nt, Nosc = x.shape
    
    Kb0      = 0.1 * np.diag(abs(np.randn(Nosc*(Nosc*P+1)*T)))
    mu_beta0 = np.zeros(Nosc*(Nosc*P+1)*T)
    beta     = np.zeros((Nt-T, Nosc*Nosc, P))
    OMEGA    = np.zeros((Nt-T, Nosc, P))
    L        = np.nan * np.ones(Nt-T)
    Changes  = np.nan * np.ones(Nt-T)
    
    theta_hat   = np.nan * np.ones((Nt-T, Nosc))
    #%%
    cnt = 1
    
    sigma    = 1/prec_param * abs(randn(Nosc*T))
    
    Total_Epoch = int((Nt-T)/T)
    
    for i in range(T, Nt, T):
        #%%
        print('Epoch: (%d / %d), index: %d'%(cnt, Total_Epoch, i))
        #########################################################################
        if T ==1:
            x_train = np.sin(x[i-1, :].reshape(1, Nosc) - x[i-1, :].reshape(Nosc, 1))
            if confounder == True:
                x_train = np.concatenate((x_train, np.ones((Nosc, 1))), axis=1)
        else:
            for t in range(0, T):
                tmp = np.sin(x[(i-T)+t, :].reshape(1, Nosc) - x[(i-T)+t, :].reshape(Nosc, 1))
                if t == 0:
                    x_train = tmp
                else:
                    x_train = np.concatenate((x_train, tmp), axis=0)                
            if confounder == True:
                x_train = np.concatenate((x_train, np.ones((Nosc*T, 1))), axis=1)
        #########################################################################
        if T ==1:
            y_train = np.zeros((1, Nosc))
            for n in range(Nosc):
                theta_unwrap = np.unwrap(deepcopy(x[i-1:i+1, n]))
                y_train[:, n] = (theta_unwrap[1] - theta_unwrap[0])/h
        else:
            y_train = np.zeros((T, Nosc))
            for t in range(0, T, 1):
                for n in range(Nosc):
                    theta_unwrap = np.unwrap(deepcopy(x[(i-T)+t:(i-T)+t+2, n]))
                    y_train[t, n] = (theta_unwrap[1] - theta_unwrap[0])/h
                    
                    # tmp_idx = np.arange(i+t, i+t+2, 1)
                    # print('start:%d / end:%d'%(tmp_idx[0], tmp_idx[-1]))
        y_train = y_train.reshape(-1, order='C')
        Dx      = make_Dx(x_train, T, Nosc, P, confounder)
        #########################################################################
        #### Prediction step : Update Model covarianvce (Observation noise)  
        noise   = abs(1/prec_param)
        sigma   = np.diag(Dx @ Kb0 @ Dx.T) + noise
        
        # if i > 0:
        #     I     = np.eye(len(sigma))
        #     sigma   = (1/prec_param) * np.diag(Dx @ Kb0 @ Dx.T + I)
        #### Update step : Update posterior mean and covariance (coefficient) 
        mu_beta, Kb, loglike, change_ratio = update_coeff(x_train, y_train, Dx, mu_beta0, Kb0, sigma, T, Nosc)
    
        mu_beta0   = deepcopy(mu_beta)
        Kb0        = deepcopy(Kb)
        sigma0     = deepcopy(sigma)
        L[i-T]       = deepcopy(loglike)
        Changes[i-T] = deepcopy(change_ratio)
        
        tmp_y = (Dx @ mu_beta).reshape((T, Nosc), order='C')
        if i == T:
            y_hat = deepcopy(tmp_y)
        else:
            y_hat = np.concatenate((y_hat, deepcopy(tmp_y)), axis=0)
        
        tmp_beta = mu_beta.reshape((T*Nosc, Nosc*P+1))#B_hat.reshape((T*Nosc, Nosc*P+1))#
        for p in range(P):
            idx = np.arange(0, Nosc, 1) + p*Nosc
            beta[i-T:i, :, p]  = tmp_beta[:,idx].reshape((T, Nosc*Nosc))
        
        OMEGA[i-T:i, :, p] = tmp_beta[:,-1].reshape((T, Nosc))
        cnt += 1
    
    return beta, OMEGA, Changes, L

##############################################################################
def est_dynamical_oscillator_1st_order_fourier(x, P, T, h, prec_param):   
    Nt, Nosc = x.shape
    
    Kb0      = np.eye(Nosc*(Nosc*2*P+1)*T)
    # tmp      = np.kron(np.eye(Nosc*T), rand(Nosc*2*P+1, Nosc*2*P+1))
    # Kb0      = 0.5 * (tmp @ tmp.T)
    
    mu_beta0 = np.zeros(Nosc*(Nosc*2*P+1)*T)
    L        = np.nan * np.ones(Nt-T)
    Changes  = np.nan * np.ones(Nt-T)
    Entropy  = np.nan * np.ones(Nt-T)
    
    beta     = np.zeros((Nt-T, Nosc*Nosc, 2*P))
    OMEGA    = np.zeros((Nt-T, Nosc, P))
        
    sigma    = 1/prec_param *  abs(rand(Nosc*T))
    
    theta_hat   = np.nan * np.ones((Nt-T, Nosc))
    #%%
    cnt = 1
    
    Total_Epoch = int((Nt-T)/T)
    
    for i in range(T, Nt, T):#(0, Nt-T-1, T):
        #%%
        print('Epoch: (%d / %d), index: %d'%(cnt, Total_Epoch, i))
        #########################################################################
        if T ==1:
            tmp_sin = np.sin(x[i-1, :].reshape(1, Nosc) - x[i-1, :].reshape(Nosc, 1))
            tmp_cos = np.cos(x[i-1, :].reshape(1, Nosc) - x[i-1, :].reshape(Nosc, 1))
            tmp_cos = tmp_cos - np.eye(Nosc)
            
            x_train = np.concatenate((tmp_cos, tmp_sin), axis=1)
            x_train = np.concatenate((x_train, np.ones((Nosc, 1))), axis=1)
        else:
            for t in range(0, T):
                tmp_cos = np.cos(x[(i-T)+t, :].reshape(1, Nosc) - x[(i-T)+t, :].reshape(Nosc, 1))
                tmp_sin = np.sin(x[(i-T)+t, :].reshape(1, Nosc) - x[(i-T)+t, :].reshape(Nosc, 1))
                
                tmp_cos = tmp_cos - np.eye(Nosc)
                
                tmp     = np.concatenate((tmp_cos, tmp_sin), axis=1)
                if t == 0:
                    x_train = tmp
                else:
                    x_train = np.concatenate((x_train, tmp), axis=0)                

            x_train = np.concatenate((x_train, np.ones((Nosc*T, 1))), axis=1)
        #########################################################################
        if T ==1:
            y_train = np.zeros((1, Nosc))
            for n in range(Nosc):
                theta_unwrap = np.unwrap(deepcopy(x[i-1:i+1, n]))
                y_train[:, n] = (theta_unwrap[1] - theta_unwrap[0])/h
        else:
            y_train = np.zeros((T, Nosc))
            for t in range(0, T, 1):
                for n in range(Nosc):
                    theta_unwrap = np.unwrap(deepcopy(x[(i-T)+t:(i-T)+t+2, n]))
                    y_train[t, n] = (theta_unwrap[1] - theta_unwrap[0])/h
                    
                    # tmp_idx = np.arange(i+t, i+t+2, 1)
                    # print('start:%d / end:%d'%(tmp_idx[0], tmp_idx[-1]))
        y_train = y_train.reshape(-1, order='C')
        Dx      = make_Dx(x_train, T, Nosc, 2*P)
        #########################################################################
        #### Prediction step : Update Model covarianvce (Observation noise)  
        noise   = 1/prec_param
        sigma   = np.diag(Dx @ Kb0 @ Dx.T) + noise
        
        # if i > 0:
        #     sgm     = np.diag(deepcopy(sigma))
        #     sigma   = np.diag(Dx @ Kb0 @ Dx.T + sgm)
        # print(sigma[0])
        #### Update step : Update posterior mean and covariance (coefficient) 
        mu_beta, Kb, loglike, change_ratio = update_coeff(x_train, y_train, Dx, mu_beta0, Kb0, sigma, T, Nosc)
    
        mu_beta0   = deepcopy(mu_beta)
        Kb0        = deepcopy(Kb)
        sigma0     = deepcopy(sigma)
        L[i-T]       = deepcopy(loglike)
        Changes[i-T] = deepcopy(change_ratio)
        
        # print(Entropy[i-T])
        
        tmp_y = (Dx @ mu_beta).reshape((T, Nosc), order='C')
        if i == T:
            y_hat = deepcopy(tmp_y)
        else:
            y_hat = np.concatenate((y_hat, deepcopy(tmp_y)), axis=0)
        
        tmp_beta = mu_beta.reshape((T*Nosc, Nosc*2*P+1))#B_hat.reshape((T*Nosc, Nosc*P+1))#
        for p in range(2*P):
            idx = np.arange(0, Nosc, 1) + p*Nosc
            beta[i-T:i, :, p]  = tmp_beta[:,idx].reshape((T, Nosc*Nosc))
            
        OMEGA[i-T:i, :, 0] = tmp_beta[:,-1].reshape((T, Nosc))
        cnt += 1
    
    
    return beta, OMEGA, Changes, L, y_hat, sigma0, Kb0



#%%
def reconstruct_phase_response_curve(beta, omega, Nosc):
    Nt, Npair, Nparam = beta.shape
    phi_delta         = np.linspace(0, 2*np.pi, 10)
    PRC               = np.zeros((Nt, 10, Npair))
    
    cnt = 0
    for ref in range(Nosc):
        wn   = deepcopy(omega[:,ref])
        for osc in range(Nosc):            
            if Nparam == 2:
                a            = beta[:, cnt, 0]
                b            = beta[:, cnt, 1]
                
                PRC[:,:,cnt] = wn[:, np.newaxis] + a[:, np.newaxis] * np.cos(phi_delta[np.newaxis, :]) \
                                                 + b[:, np.newaxis] * np.sin(phi_delta[np.newaxis, :])
            elif Nparam == 4:
                a1           = beta[:, cnt, 0]
                b1           = beta[:, cnt, 1]
                a2           = beta[:, cnt, 2]
                b2           = beta[:, cnt, 3]
                PRC[:,:,cnt] = wn[:, np.newaxis] + a1[:, np.newaxis] * np.cos(    phi_delta[np.newaxis, :]) \
                                                 + b1[:, np.newaxis] * np.sin(    phi_delta[np.newaxis, :]) \
                                                 + a2[:, np.newaxis] * np.cos(2 * phi_delta[np.newaxis, :]) \
                                                 + b2[:, np.newaxis] * np.sin(2 * phi_delta[np.newaxis, :]) 
            
            cnt += 1
    return PRC, phi_delta