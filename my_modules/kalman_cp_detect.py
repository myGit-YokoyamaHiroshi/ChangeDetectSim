# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 09:54:06 2020

@author: yokoyama
"""

from copy import deepcopy
from numpy.matlib import repmat
from numpy.random import randn, rand
import numpy as np


class kalman_CP:
    def __init__(self, x, T, prec_param):
        self.x          = x
        self.T          = T
        self.prec_param = prec_param

##############################################################################
    def est_kalman(self):
        x          = self.x
        T          = self.T
        prec_param = self.prec_param
        
        
        Nt, Nosc   = x.shape
        
        self.Nosc  = Nosc
        
        Total_Epoch = int((Nt-T)/T)

        
        Kb0      = np.eye(Nosc*(Nosc+1)*T)
        mu_beta0 = np.zeros(Nosc*(Nosc+1)*T)
        
        # Kb0      = np.eye(Nosc*Nosc*T)
        # mu_beta0 = np.zeros(Nosc*Nosc*T)
        Changes  = np.nan * np.ones(Total_Epoch)
        
        beta     = np.zeros((Nt-T, Nosc*Nosc))
        residual = np.zeros((Nt-T, Nosc))
        #%%
        cnt = 0

        for i in range(T, Nt, T):
            #%%
            print('Epoch: (%d / %d), index: %d'%(cnt+1, Total_Epoch, i))
            #########################################################################
            kalman_CP.make_features(self, x, i)            
            Dx      = kalman_CP.make_Dx(self)#(x_train, T, Nosc, 2*P)
            self.Dx = Dx
            
            
            #########################################################################
            #### Update step : Update prior distribution (update model parameter) 
            mu_beta, Kb, change_ratio, S = kalman_CP.update_coeff(self, mu_beta0, Kb0, prec_param)
            
            
            mu_beta0   = deepcopy(mu_beta)
            Kb0        = deepcopy(Kb)
            sigma0     = deepcopy(np.diag(S))
            Changes[cnt] = deepcopy(change_ratio)
            
            # print([i-T])
            
            tmp_y = (Dx @ mu_beta).reshape((T, Nosc), order='C')
            if i == T:
                y_hat = deepcopy(tmp_y)
            else:
                y_hat = np.concatenate((y_hat, deepcopy(tmp_y)), axis=0)
            
            tmp_beta       = mu_beta.reshape((T*Nosc, Nosc+1))
            beta[i-T:i, :] = tmp_beta[:,0:-1].reshape((T, Nosc*Nosc))  
            residual[i-T:i, :] = tmp_beta[:,-1].reshape((T, Nosc))
            
            # tmp_beta       = mu_beta.reshape((T*Nosc, Nosc))
            # beta[i-T:i, :] = tmp_beta.reshape((T, Nosc*Nosc))  
            cnt += 1
        
        self.beta     = beta
        self.residual = residual
        self.Changes  = Changes
        self.y_hat    = y_hat
        self.sigma0   = sigma0
        self.Kb0      = Kb0
        
        # return beta, OMEGA, Changes, L, y_hat, sigma0, Kb0
##############################################################################    
    def make_features(self, x, idx):
        Nosc = self.Nosc
        i    = idx
        x_train = np.zeros((Nosc, Nosc))
        for n in range(Nosc):
            x_train[n,:] = x[i-1,:]
        x_train = np.concatenate((x_train, np.ones((Nosc, 1))), axis=1)
        #########################################################################
        y_train = x[i,:]
        
        self.x_train = x_train
        self.y_train = y_train
##############################################################################
    
    def make_Dx(self):#(X, T, N, P):
        X    = self.x_train
        T    = self.T
        N    = self.Nosc
        
        Dx   = np.zeros((N*T,  N*(N+1)*T), dtype=float)
        # Dx   = np.zeros((N*T,  N*N*T), dtype=float)
        
        cnt = 0;
        
        if X.shape[0] == 1:
            order_index = np.sort(repmat(np.arange(0, T), 1, N))
            order_index = order_index[0,:]
        else:
            order_index = np.arange(0, X.shape[0])
            
        for i in order_index:
            tmp_x = deepcopy(X[i, :]).reshape(-1)
            
            idx = np.arange(cnt*(N+1), (cnt+1)*(N+1), 1)
            # idx = np.arange(cnt*N, (cnt+1)*N, 1)
                
            Dx[cnt, idx] = tmp_x
            cnt += 1
            
        return Dx
    
##############################################################################
    def update_coeff(self, mu, P, prec_param):#(X, Y, Dx, mu_beta, Kb, sigma0, T, N):
        Y  = self.y_train
        H  = self.Dx
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
            
        def hoteling_T2(error, S):
            S_inv = inv_use_cholensky(S)
            T2    = error.T @ S_inv @ error
            
            return T2
        
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
        
        def loglikelihood(mu, S, H, Y):
            S_inv   = inv_use_cholensky(S)
            err     = Y - H @ mu
            loglike = 0.5 * (mylogdet(S) - err @ S_inv @ err)
            
            return loglike
            
        def inv_use_cholensky(M):
            L     = np.linalg.cholesky(M)
            L_inv = np.linalg.inv(L)
            M_inv = np.dot(L_inv.T, L_inv)
            
            return M_inv
        
        ############# Estimate posterior distribution #############################
        S       = 1/prec_param * np.eye(N*T) + H @ P @ H.T 
        S_inv   = inv_use_cholensky(S)
        #### update Kalman gain
        K       = P @ H.T @ S_inv # Kalman Gain
        #### update covariance
        P_new   = P - K @ S @ K.T 
        #### update mean
        mu_new  = mu + K  @ (Y.reshape(-1) - np.dot(H, mu).reshape(-1)) 
        ###########################################################################
        S_new        = 1/prec_param * np.eye(N*T) + H @ P_new @ H.T 
        err          = (Y.reshape(-1) - np.dot(H, mu_new).reshape(-1)) 
        change_ratio = hoteling_T2(err, S_new) # 
        return mu_new, P_new, change_ratio, S



    