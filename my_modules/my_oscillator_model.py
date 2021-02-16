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

def func_kuramoto(theta, K, omega):
    Nosc = theta.shape[0]
    phase_diff = theta.reshape(1, Nosc) - theta.reshape(Nosc, 1)
    phase_dynamics = omega + np.sum(K * np.sin(phase_diff.T), axis=0) + randn(Nosc)
    return phase_dynamics

def func_oscillator_approx_fourier_series(theta, K1, K2, omega, noise_scale):
    Nosc = theta.shape[0]
    phase_diff = theta.reshape(1, Nosc) - theta.reshape(Nosc, 1)
    
    if len(K1.shape)==2:
        phase_dynamics = omega + np.sum(K1.T * np.cos(phase_diff.T), axis=0) + np.sum(K2.T * np.sin(phase_diff.T), axis=0) + noise_scale * randn(Nosc)
    elif len(K1.shape)==3:
        _,_,Norder = K1.shape
        
        for n in range(Norder):
            if n == 0:
                Cos = np.sum(K1[:,:,n].T * np.cos(phase_diff.T), axis=0)
                Sin = np.sum(K2[:,:,n].T * np.sin(phase_diff.T), axis=0)
            else:
                Cos += np.sum(K1[:,:,n].T * np.cos(n * phase_diff.T), axis=0)
                Sin += np.sum(K2[:,:,n] .T* np.sin(n * phase_diff.T), axis=0)
        
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