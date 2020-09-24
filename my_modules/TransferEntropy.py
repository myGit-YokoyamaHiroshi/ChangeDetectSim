# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.family']      = 'Arial'#"IPAexGothic"
plt.rcParams['mathtext.fontset'] = 'stix' # math fontの設定
plt.rcParams['xtick.direction']  = 'in'#x軸の目盛線が内向き('in')か外向き('out')か双方向か('inout')
plt.rcParams['ytick.direction']  = 'in'
plt.rcParams["font.size"]        = 12 # 全体のフォントサイズが変更されます。
plt.rcParams['lines.linewidth']  = 0.5
plt.rcParams['figure.dpi']       = 96
plt.rcParams['savefig.dpi']      = 300 

def TransferEntropy(X, Y, Ntwin, r,tau):
    '''
        Transfer Entropy from y to x with resolution r and time-delay tau
    '''
    x      = X[:Ntwin]
    y      = Y[:Ntwin]
    xdelay = X[tau:Ntwin+tau]
    
    del X
    del Y
    
    # the number of bins
    binsx = 1+int((max(x)-min(x))/(r))
    binsy = 1+int((max(y)-min(y))/(r))
    
    # histgrams
    # p(x(t))
    px  = np.histogram(x, bins=binsx)
    # p(x(t),y(t))
    pxy = np.histogram2d(x, y, bins=(binsx,binsy))
    # p(x(t),x(t+tau))
    pxx = np.histogram2d(xdelay, x, bins=(binsx,binsx))
    # p(x(t),x(t+tau),y(t))
    pxxy = np.histogramdd(np.c_[x, xdelay, y],bins=(binsx,binsx,binsy))
    
    # estimation of p.d.f.
    # p(x(t))
    px   = px[0]/np.sum(px[0])
    # p(x(t),y(t))
    pxy  = pxy[0]/np.sum(pxy[0])
    # p(x(t),x(t+tau))
    pxx  = pxx[0]/np.sum(pxx[0])
    # p(x(t),x(t+tau),y(t))
    pxxy = pxxy[0]/np.sum(pxxy[0])
    
    # joint entropy
    Hx   = -np.dot(px[px!=0],np.log2(px[px!=0]))
    Hxx  = -np.dot(pxx[pxx!=0],np.log2(pxx[pxx!=0])) 
    Hxy  = -np.dot(pxy[pxy!=0],np.log2(pxy[pxy!=0]))
    Hxxy = -np.dot(pxxy[pxxy!=0],np.log2(pxxy[pxxy!=0]))
    
    TE_y2x = (Hxx-Hx+Hxy-Hxxy)
    
    if (TE_y2x<0) | (np.isnan(TE_y2x) == 1) | (abs(TE_y2x)==np.inf):
        TE_y2x = 0
    
    return TE_y2x 
#%%
def NormalizedTransferEntropy(X, Y, Ntwin, r,tau):
    '''
        Transfer Entropy from y to x with resolution r and time-delay tau
    '''
    x      = X[:Ntwin]
    y      = Y[:Ntwin]
    xdelay = X[tau:Ntwin+tau]
    
    # the number of bins
    binsx = 1+int((max(x)-min(x))/(r))
    binsy = 1+int((max(y)-min(y))/(r))
    
    # histgrams
    # p(x(t))
    px  = np.histogram(x, bins=binsx)
    # p(x(t),y(t))
    pxy = np.histogram2d(x, y, bins=(binsx,binsy))
    # p(x(t),x(t+tau))
    pxx = np.histogram2d(xdelay, x, bins=(binsx,binsx))
    # p(x(t),x(t+tau),y(t))
    pxxy = np.histogramdd(np.c_[x, xdelay, y],bins=(binsx,binsx,binsy))
    
    # estimation of p.d.f.
    # p(x(t))
    px   = px[0]/np.sum(px[0])
    # p(x(t),y(t))
    pxy  = pxy[0]/np.sum(pxy[0])
    # p(x(t),x(t+tau))
    pxx  = pxx[0]/np.sum(pxx[0])
    # p(x(t),x(t+tau),y(t))
    pxxy = pxxy[0]/np.sum(pxxy[0])
    
    # joint entropy
    Hx   = -np.dot(px[px!=0],np.log2(px[px!=0]))
    Hxx  = -np.dot(pxx[pxx!=0],np.log2(pxx[pxx!=0])) 
    Hxy  = -np.dot(pxy[pxy!=0],np.log2(pxy[pxy!=0]))
    Hxxy = -np.dot(pxxy[pxxy!=0],np.log2(pxxy[pxxy!=0]))
    
    NormTerm = Hxx - Hx
    
    TE_y2x     = (Hxx-Hx+Hxy-Hxxy)
    TE_shuffle = surrogate_TransferEntropy(X, Y, Ntwin, r,tau)
    
    del X
    del Y
    
    nTE_y2x    = (TE_y2x - TE_shuffle)/NormTerm
    
    if (nTE_y2x<0) | (np.isnan(nTE_y2x) == 1) | (abs(nTE_y2x)==np.inf):
        nTE_y2x = 0
    
    return nTE_y2x 


#%%
def surrogate_TransferEntropy(X, Y, Ntwin, r,tau):
    '''
        Transfer Entropy from y to x with resolution r and time-delay tau with randomized x
    '''
    X          = np.random.permutation(X)
    surrTE_y2x = TransferEntropy(X, Y, Ntwin, r,tau)  
   
    return surrTE_y2x 

def surrogate_NormalizdTransferEntropy(X, Y, Ntwin, r,tau):
    '''
        Transfer Entropy from y to x with resolution r and time-delay tau with randomized x
    '''
    X          = np.random.permutation(X)
    surrTE_y2x = NormalizedTransferEntropy(X, Y, Ntwin, r,tau)  
   
    return surrTE_y2x 

#系列yから系列xへの系列zで条件づけたTransfer Entropy
def ConditionalMutualInformation(x,y,z,r,tau):
    '''
        Mutual information between y(t) and x(t+tau) with resolution r and time-delay tau conditioned by x(t), z(t), and z(t-1)
        MI(x(t+tau);y(t)|x(t),z(t),z(t-1))
    '''
    # 異なる定式化も可能．
    # 今回は演習課題に特化させて定式化を行った．

    # the number of bins
    binsx = 1+int((max(x)-min(x))/(r))
    binsy = 1+int((max(y)-min(y))/(r))
    binsz = 1+int((max(z)-min(z))/(r))
    
    # histgrams
    # p(x(t),z(t),z(t-1))
    pxz = np.histogramdd(np.c_[x[1:],z[1:],z[:-1]],bins=(binsx,binsz,binsz))
    # p(x(t),y(t),z(t),z(t-1))
    pxyz = np.histogramdd(np.c_[x[1:],y[1:],z[1:],z[:-1]],bins=(binsx,binsy,binsz,binsz))
    # p(x(t),x(t+tau),z(t),z(t-1))
    pxxz = np.histogramdd(np.c_[x[1:(-tau)],x[(1+tau):],z[1:(-tau)],z[:(-tau-1)]],bins=(binsx,binsx,binsz,binsz))
    # p(x(t),x(t+tau),y(t),z(t),z(t-1))
    pxxyz = np.histogramdd(np.c_[x[1:(-tau)],x[(1+tau):],y[1:(-tau)],z[1:(-tau)],z[:(-tau-1)]],bins=(binsx,binsx,binsy,binsz,binsz))
    
    # estimation of p.d.f.
    # p(x(t),z(t),z(t-1))
    pxz = pxz[0]/np.sum(pxz[0])
    # p(x(t),y(t),z(t),z(t-1))
    pxyz = pxyz[0]/np.sum(pxyz[0])
    # p(x(t),x(t+tau),z(t),z(t-1))
    pxxz = pxxz[0]/np.sum(pxxz[0])
    # p(x(t),x(t+tau),y(t),z(t),z(t-1))
    pxxyz = pxxyz[0]/np.sum(pxxyz[0])
    
    # joint entropy
    Hxz = -np.dot(pxz[pxz!=0],np.log2(pxz[pxz!=0]))
    Hxxz =-np.dot(pxxz[pxxz!=0],np.log2(pxxz[pxxz!=0])) 
    Hxyz = -np.dot(pxyz[pxyz!=0],np.log2(pxyz[pxyz!=0]))
    Hxxyz = -np.dot(pxxyz[pxxyz!=0],np.log2(pxxyz[pxxyz!=0]))
    
    return (Hxxz-Hxz+Hxyz-Hxxyz)
