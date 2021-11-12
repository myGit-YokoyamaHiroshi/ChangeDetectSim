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


def func_oscillator_approx_fourier_series(theta, K1, K2, omega):
    Nosc       = theta.shape[0]
    phase_diff = np.zeros((Nosc, Nosc))
    for i in range(Nosc):
        for j in range(Nosc):
            
            phase_diff[i,j] = np.mod(theta[j] - theta[i], 2*np.pi)
    
    if len(K1.shape)==2:
        Nparam = 1
    elif len(K1.shape) ==3:
        _, _, Nparam = K1.shape
    
    if Nparam==1:
        phase_dynamics = omega + np.sum(K1.T * np.cos(phase_diff), axis=0) + np.sum(K2.T * np.sin(phase_diff), axis=0)
    else:
        # phase_dynamics = omega
        
        for k in range(0, int(Nparam)):
            p = k + 1
            if k == 0:
                phase_dynamics = K1[:,:,k].T * np.cos(p * phase_diff) + K2[:,:,k].T * np.sin(p * phase_diff)
            else:
                phase_dynamics += K1[:,:,k].T * np.cos(p * phase_diff) + K2[:,:,k].T * np.sin(p * phase_diff)
    
        phase_dynamics = omega + np.sum(phase_dynamics, axis=0)
    
    return phase_dynamics


def euler_maruyama_oscillator_approx_fourier_series(h, func, theta_now, K1, K2, omega, noise_scale):
    dt = h
    p  = noise_scale
    dw = np.sqrt(dt) * np.random.randn(theta_now.shape[0])
    
    theta      = theta_now + func(theta_now, K1, K2, omega) * dt
    theta_next = theta +  p * dw
    theta_next = np.mod(theta_next, 2*np.pi)
    return theta_next
#%%
def generate_sythetic_phase(theta_init, omega, K1_tr, K2_tr, Nt, Nosc, noise, h):
    dtheta        = np.zeros((Nt, Nosc))
    theta         = np.zeros((Nt, Nosc))
    theta[0, :]   = theta_init
    
    phase_dynamics       = np.zeros((Nt, Nosc))
    phase_dynamics[0, :] = func_oscillator_approx_fourier_series(theta[0, :], K1_tr[:,:,0], K2_tr[:,:,0], omega)
    
    np.random.seed(1)
    for t in range(1, Nt):
        if t < int(Nt/3):
            Nst = 0
        elif int(Nt/3) <= t < int(Nt*2/3):
            Nst = 1
        else:
            Nst = 2
            
        noise_scale = noise[Nst]
        
        K1 = K1_tr[:,:,Nst]
        K2 = K2_tr[:,:,Nst]
        
        theta_now  = theta[t-1, :]
        theta_next = euler_maruyama_oscillator_approx_fourier_series(h, func_oscillator_approx_fourier_series, theta_now, K1, K2, omega, noise_scale)
        
        theta[t, :]          = theta_next.reshape(1, Nosc)
        phase_dynamics[t, :] = func_oscillator_approx_fourier_series(theta[t, :], K1, K2, omega)
    
        for i in range(Nosc):
            theta_unwrap = np.unwrap(deepcopy(theta[t-1:t+1, i]))
            
            dtheta[t, i] = (theta_unwrap[1] - theta_unwrap[0])/h
    
    return theta, dtheta, phase_dynamics 
#%%
def reconstruct_phase_response_curve(beta, omega, Nosc):
    Nt, Npair, Nparam = beta.shape
    phi_delta         = np.linspace(0, 2*np.pi, 30)
    PRC               = np.zeros((Nt, 30, Npair))
    
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