# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 09:54:06 2020

@author: yokoyama
"""

from copy import deepcopy
from numpy.matlib import repmat
from numpy.random import randn, rand
import numpy as np


class AR_Kalman:
    def __init__(self, x, P, UC, flimits):
        self.x       = x
        self.P       = P
        self.UC      = UC
        self.flimits = flimits

##############################################################################
    def est_kalman(self):
        x           = self.x
        P           = self.P
        UC          = self.UC
        flimits     = self.flimits
        
        Nt, Nch     = x.shape
        
        self.Nch    = Nch
        
        Total_Epoch = int(Nt-P)
        
        Kb0         = np.eye(Nch*(Nch*P+1))
        mu_beta0    = np.zeros(Nch*(Nch*P+1))
        sigma0      = np.eye(Nch)
        
        S           = np.zeros((Nch, Nch, Total_Epoch))
        Changes     = np.nan * np.ones(Total_Epoch)
        loglike     = np.nan * np.ones(Total_Epoch)
        
        beta        = np.zeros((Nt-P, Nch*Nch, P))
        residual    = np.zeros((Nt-P, Nch))
        #%%
        cnt      = 0
        Q1       = UC * np.eye(Kb0.shape[0])
        Q2       = np.eye(Nch)
        
        for i in range(P, Nt, 1):
            #%%
            print('Epoch: (%d / %d), index: %d'%(cnt+1, Total_Epoch, i))
            #########################################################################
            AR_Kalman.make_features(self, x, i)            
            Dx      = AR_Kalman.make_Dx(self)#(x_train, T, Nosc, 2*P)
            self.Dx = Dx
            
            #########################################################################
            tmp_y = (Dx @ mu_beta0).reshape((1,Nch), order='C')
            if i == P:
                y_hat = deepcopy(tmp_y)
            else:
                y_hat = np.concatenate((y_hat, deepcopy(tmp_y)), axis=0)
            
            #### Update step : Update prior distribution (update model parameter) 
            mu_beta, Kb, change_ratio, sigma, L, Q2 = AR_Kalman.update_coeff(self, mu_beta0, Kb0, sigma0, UC, Q2)
            Kb           = Kb + Q1
            
            mu_beta0     = deepcopy(mu_beta)
            Kb0          = deepcopy(Kb)
            sigma0       = deepcopy(sigma)
            
            S[:,:,cnt]   = deepcopy(sigma)
            Changes[cnt] = deepcopy(change_ratio)
            loglike[cnt] = L
            
            tmp_beta = mu_beta.reshape((Nch, Nch*P+1))
            # tmp_beta = mu_beta.reshape((Nch, Nch*P))
            for p in range(P):
                idx = np.arange(0, Nch, 1) + p*Nch
                beta[cnt, :, p]  = tmp_beta[:,idx].reshape((Nch*Nch))
            
            residual[cnt, :] = tmp_beta[:,-1]
            
            cnt += 1
        
        PDC = AR_Kalman.calc_connectivity(beta, Nch, Nt, P, flimits)
        
        self.PDC      = PDC
        self.beta     = beta
        self.residual = residual
        self.Changes  = Changes
        self.loglike  = loglike
        self.y_hat    = y_hat
        self.S        = S
        self.Kb0      = Kb0
        
        # return beta, OMEGA, Changes, L, y_hat, sigma0, Kb0
##############################################################################
    def calc_connectivity(beta, Nch, Nt, P, flimits):
        frqs        = np.arange(flimits[0], flimits[-1]+1, 1)
        
        Nf           = len(frqs)
        Nt, Ndim, Np = beta.shape
        
        PDC          = np.zeros((Nch, Nch, Nt))
        for t in range(Nt):
            Aij        = np.zeros((Nch, Nch, Nf), dtype=complex)
            for f in range(Nf):
                Af_tmp = np.zeros((Nch, Nch, Np), dtype=complex)
                for p in range(Np):
                    Af_tmp[:,:,p] = beta[t, :, p].reshape(Nch, Nch) * np.exp(1j * 2 * np.pi * f * (p+1))
                
                Aij[:,:,f] = np.eye(Nch) - np.sum(Af_tmp, axis=2)
            
            f_width    = flimits[1] - flimits[0]
            
            term1      = abs(Aij)**2
            term2      = np.sum(abs(Aij)**2, axis=1)
            term2      = term2[:,np.newaxis]
            
            pdc        = 1/f_width * np.sum(term1/term2, axis=2)
            
            PDC[:,:,t] = pdc
        
        return PDC
##############################################################################    
    def make_features(self, x, idx):
        Nch = self.Nch
        P   = self.P
        i   = idx
        x_train = np.flipud(x[i-P:i,:]).reshape(-1)
        x_train = np.concatenate((x_train, np.ones(1)), axis=0)
        #########################################################################
        y_train = x[i,:]
        
        self.x_train = x_train
        self.y_train = y_train
##############################################################################
    
    def make_Dx(self):#(X, T, N, P):
        X    = self.x_train
        N    = self.Nch
        
        Dx   = np.kron(np.eye(N), X)
        
        return Dx
    
##############################################################################
    def update_coeff(self, mu, P, S, UC, Q2):#(X, Y, Dx, mu_beta, Kb, sigma0, T, N):
        Y  = self.y_train
        H  = self.Dx
        N  = self.Nch
        
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
        
        def loglikelihoodratio(mu1, mu2, S1, S2, y):
            err1 = y - mu1
            err2 = y - mu2
            
            logdetS1 = mylogdet(S1)
            logdetS2 = mylogdet(S2)
            
            term1 = logdetS2 - logdetS1
            term2 = 0.5 * err1 @ inv_use_cholensky(S1) @ err1
            term3 = 0.5 * err2 @ inv_use_cholensky(S2) @ err2
            
            loglike_ratio = term1 - term2 + term3
            
            return loglike_ratio
            
        def inv_use_cholensky(M):
            L     = np.linalg.cholesky(M)
            L_inv = np.linalg.inv(L)
            M_inv = np.dot(L_inv.T, L_inv)
            
            return M_inv
        
        ############# Estimate posterior distribution #############################
        err     = (Y.reshape(-1) - np.dot(H, mu).reshape(-1)) 
        
        Q2      = (1 - UC) * Q2 + UC * err[:, np.newaxis] @ err[:, np.newaxis].T
        
        S_new   = Q2 + H @ P @ H.T 
        S_new_inv   = inv_use_cholensky(S_new)
        #### update Kalman gain
        K       = P @ H.T @ S_new_inv # Kalman Gain
        #### update mean
        mu_new  = mu + K  @ (Y.reshape(-1) - np.dot(H, mu).reshape(-1)) 
        #### update covariance
        P_new   = P - K @ S_new @ K.T 
        ###########################################################################
        Ndim         = Y.shape[0]
        loglike      = 0.5 * (-Ndim * np.log(2*np.pi) - mylogdet(S) - (Y.reshape(-1) - np.dot(H, mu).reshape(-1))  @ inv_use_cholensky(S) @ (Y.reshape(-1) - np.dot(H, mu).reshape(-1)) )
        
        change_ratio = hoteling_T2(Y - np.dot(H, mu_new), S_new) 
        # change_ratio =  KL_div(mu_new, mu, P_new, P) 
        
        # change_ratio = -loglike

        return mu_new, P_new, change_ratio, S_new, loglike, Q2



    