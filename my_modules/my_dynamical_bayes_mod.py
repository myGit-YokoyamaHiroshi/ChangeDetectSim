# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 09:54:06 2020

@author: yokoyama
"""

from copy import deepcopy
from numpy.matlib import repmat
from numpy.random import randn, rand
import numpy as np


class my_Bayesian_CP:
    def __init__(self, x, P, T, h, prec_param):
        self.x          = x
        self.P          = P
        self.T          = T
        self.h          = h
        self.prec_param = prec_param

##############################################################################
    def est_dynamical_oscillator_1st_order_fourier(self):
        x          = self.x
        P          = self.P
        T          = self.T
        h          = self.h
        prec_param = self.prec_param
        
        
        Nt, Nosc   = x.shape
        
        self.Nosc  = Nosc
        
        Total_Epoch = int((Nt-T)/T)
        # A = np.random.randn(Nosc*(Nosc*2*P+1)*T, Nosc*(Nosc*2*P+1)*T);
        # B = np.dot(A, A.transpose())
        Kb0      = np.eye(Nosc*(Nosc*2*P+1)*T)
        mu_beta0 = np.zeros(Nosc*(Nosc*2*P+1)*T)#np.random.rand(Nosc*(Nosc*2*P+1)*T)#
        L        = np.nan * np.ones(Total_Epoch)
        Changes  = np.nan * np.ones(Total_Epoch)
        
        beta     = np.zeros((Nt-T, Nosc*Nosc, 2*P))
        OMEGA    = np.zeros((Nt-T, Nosc))
        S        = np.zeros((Nt-T, Nosc))
        #%%
        cnt = 0

        for i in range(T, Nt, T):#(0, Nt-T-1, T):
            #%%
            print('Epoch: (%d / %d), index: %d'%(cnt+1, Total_Epoch, i))
            #########################################################################
            my_Bayesian_CP.make_fourier_features(self, x, i)            
            Dx      = my_Bayesian_CP.make_Dx(self)#(x_train, T, Nosc, 2*P)
            self.Dx = Dx
            
            #########################################################################
            tmp_y = (Dx @ mu_beta0).reshape((T, Nosc), order='C')
            if i == T:
                y_hat = deepcopy(tmp_y)
            else:
                y_hat = np.concatenate((y_hat, deepcopy(tmp_y)), axis=0)
                
            #### Update step : Update prior distribution (update model parameter) 
            mu_beta, Kb, loglike, change_ratio, sigma = my_Bayesian_CP.update_coeff(self, mu_beta0, Kb0, prec_param)
        
            mu_beta0   = deepcopy(mu_beta)
            Kb0        = deepcopy(Kb)
            sigma0     = deepcopy(np.diag(sigma))
            L[cnt]       = deepcopy(loglike)
            Changes[cnt] = deepcopy(change_ratio)
            S[cnt,:]     = deepcopy(np.diag(sigma)) 
            
            # print([i-T])
            
            
            
            tmp_beta = mu_beta.reshape((T*Nosc, Nosc*2*P+1))#B_hat.reshape((T*Nosc, Nosc*P+1))#
            for p in range(2*P):
                idx = np.arange(0, Nosc, 1) + p*Nosc
                beta[i-T:i, :, p]  = tmp_beta[:,idx].reshape((T, Nosc*Nosc))
                
            OMEGA[i-T:i, :] = tmp_beta[:,-1].reshape((T, Nosc))
            cnt += 1
        
        self.beta    = beta
        self.omega   = OMEGA
        self.Changes = Changes
        self.loglike = L
        self.y_hat   = y_hat
        self.sigma0  = sigma0
        self.Kb0     = Kb0
        self.S       = S
        # return beta, OMEGA, Changes, L, y_hat, sigma0, Kb0
##############################################################################    
    def make_fourier_features(self, x, idx):
        Nosc = self.Nosc
        T    = self.T
        h    = self.h
        i    = idx
        
        if T ==1:
            tmp_sin = np.sin(x[i-1, :].reshape(Nosc, 1) - x[i-1, :].reshape(1, Nosc))
            tmp_cos = np.cos(x[i-1, :].reshape(Nosc, 1) - x[i-1, :].reshape(1, Nosc))
            tmp_cos = tmp_cos - np.eye(Nosc)
            
            x_train = np.concatenate((tmp_cos, tmp_sin), axis=1)
            x_train = np.concatenate((x_train, np.ones((Nosc, 1))), axis=1)
        else:
            for t in range(0, T):
                tmp_cos = np.cos(x[(i-T)+t, :].reshape(Nosc, 1) - x[(i-T)+t, :].reshape(1, Nosc))
                tmp_sin = np.sin(x[(i-T)+t, :].reshape(Nosc, 1) - x[(i-T)+t, :].reshape(1, Nosc))
                
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
        
        self.x_train = x_train
        self.y_train = y_train
##############################################################################
    
    def make_Dx(self):#(X, T, N, P):
        X    = self.x_train
        T    = self.T
        N    = self.Nosc
        P    = 2*self.P
        
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
    
##############################################################################
    def update_coeff(self, mu_beta0, Kb0, prec_param):#(X, Y, Dx, mu_beta, Kb, sigma0, T, N):
        Y  = self.y_train
        Dx = self.Dx
        T  = self.T
        N  = self.Nosc
        
        def mylogdet(S):
            L       = np.linalg.cholesky(S)
            logdetS = 2*np.sum(np.log(np.diag(L)))
            
            return logdetS
        
        def mydet(S):
            L       = np.linalg.cholesky(S)
            detS = np.linalg.det(L)**2
            
            return detS
            
        def KL_div(mu1, mu2, K1, K2):
            D      = K1.shape[0]
            K1_inv = inv_use_cholensky(K1)
            K2_inv = inv_use_cholensky(K2)
            
            # sgn1, logdetK1     = np.linalg.slogdet(K1)
            # sgn1, logdetK2     = np.linalg.slogdet(K2)
            
            logdetK1 = mylogdet(K1)
            logdetK2 = mylogdet(K2)
            
            
            term1 = logdetK2 - logdetK1 # = logdetK1_inv - logdetK2_inv # 
            term2 = np.trace(K2_inv @ K1) - D
            term3 = (mu2 - mu1).T @ K2_inv @ (mu2 - mu1)
            
            KL    = 0.5 * (term1 + term2 + term3) # KL div ~  cross  -  
            return KL
        
            
        def inv_use_cholensky(M):
            L     = np.linalg.cholesky(M)
            L_inv = np.linalg.inv(L)
            M_inv = np.dot(L_inv.T, L_inv)
            
            return M_inv
        
        ############# Estimate posterior distribution #############################
        S       = 1/prec_param * np.eye(N*T) + Dx @ Kb0 @ Dx.T 
        tmp_inv = inv_use_cholensky(S)
        
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
    
        
        # loglike = ( - E_err - E_beta )/2
        # 0.5 * (-Ndim * np.log(2*np.pi) - mylogdet(S) - (Y.reshape(-1) - np.dot(H, mu).reshape(-1))  @ inv_use_cholensky(S) @ (Y.reshape(-1) - np.dot(H, mu).reshape(-1)) )
        Ndim    = S.shape[0]
        loglike = 0.5 * (-Ndim * np.log(2*np.pi) - mylogdet(S) - Yv @ tmp_inv @ Yv)
        ############# changing ratio ##############################################
        ########## KL divergence
        change_ratio =  KL_div(mu_beta, mu_beta0, Kb, Kb0)  #+ KL_div(mu_beta0, mu_beta, Kb0, Kb) #
        
        ###########################################################################
        return mu_beta, Kb, loglike, change_ratio, S



    