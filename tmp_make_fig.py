# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 18:28:44 2020

@author: yokoyama
"""
#%%
import matplotlib.pylab as plt
import numpy as np
import seaborn as sns
plt.rcParams['font.family']      = 'Arial'#
plt.rcParams['mathtext.fontset'] = 'stix' # math font setting
plt.rcParams["font.size"]        = 18 # Font size

sns.set_style({"xtick.direction": "in","ytick.direction": "in"})

title_str = ['Segment 1\n','Segment 2\n']
fig       = plt.figure(constrained_layout = False, figsize=(6, 8));
plt.subplots_adjust(wspace=0.8, hspace=0.8);
gs        = fig.add_gridspec(State+1, 3)
ratio     = list(np.ones(State))
ratio.append(0.08)
gs.set_height_ratios( ratio)

ax_cb     = fig.add_subplot(gs[State, 0:3])

for state in range(State):
    ### coeff. a_ij
    ax1 = fig.add_subplot(gs[0, state])
    im = sns.heatmap(K1_tr[:,:,state], 
                     vmin=0, vmax=vmax, linewidths=0.01, linecolor='whitesmoke',
                     cmap='Blues', 
                     cbar = False, 
                     ax=ax1) 
    for _, spine in im.spines.items():
           spine.set_visible(True)
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_title('$a_{ij}$')
    ax1.set_xlabel('osci. $j$')
    ax1.set_ylabel('osci. $i$')
    ax1.set_aspect('equal')
    
    ### coeff. b_ij
    ax2 = fig.add_subplot(gs[1, state])
    im = sns.heatmap(K2_tr[:,:,state], 
                     vmin=0, vmax=vmax, linewidths=0.01, linecolor='whitesmoke',
                     xticklabels = [5, 10, 15],
                     yticklabels = [5, 10, 15],
                     cmap='Blues', 
                     cbar = False, 
                     ax=ax2)
    for _, spine in im.spines.items():
           spine.set_visible(True)
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.set_title('$b_{ij}$')
    ax2.set_xlabel('osci. $j$')
    ax2.set_ylabel('osci. $i$')
    ax2.set_aspect('equal')
    
    ### coupling strength K
    ax3 = fig.add_subplot(gs[2, state])
    im = sns.heatmap(K_tr[:,:,state], 
                     vmin=0, vmax=vmax, linewidths=0.01, linecolor='whitesmoke',
                     xticklabels = [5, 10, 15],
                     yticklabels = [5, 10, 15],
                     cmap='Blues', 
                     cbar = True, cbar_kws = {"orientation":"horizontal"}, 
                     ax=ax3, cbar_ax = ax_cb) 
    for _, spine in im.spines.items():
           spine.set_visible(True)
    ax3.set_xticks([])
    ax3.set_yticks([])
    ax3.set_title('$K_{ij}$')
    ax3.set_xlabel('osci. $j$')
    ax3.set_ylabel('osci. $i$')
    ax3.set_aspect('equal')

fig_save_dir = current_path + '\\figures\\' + simName + '\\'
plt.plt.savefig(fig_save_dir + 'param_setting.png')
plt.savefig(fig_save_dir + 'param_setting.svg')

plt.show()
#%%
