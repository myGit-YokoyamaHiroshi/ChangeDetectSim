# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 09:39:40 2020

@author: yokoyama
"""
from copy import deepcopy
import matplotlib.pylab as plt
import numpy as np
#%%
def vis_directed_graph(K, vmin, vmax):
    import networkx as nx
    import matplotlib as mpl
    
    weight = deepcopy(K).reshape(-1)
    weight = weight[weight != 0]
    
    G      = nx.from_numpy_matrix(K, create_using=nx.MultiDiGraph())
    G.edges(data=True)
    
    pos    = {}
    pos[0] = np.array([0.5, 1.0])
    pos[1] = np.array([0.0, 0.0])
    pos[2] = np.array([1.0, 0.0])
    labels = {i : i + 1 for i in G.nodes()}          
    
    node_sizes  = [800  for i in range(len(G))]
    M           = G.number_of_edges()
    edge_colors = np.ones(M, dtype = int)
    edge_alphas = weight/vmax
    edge_alphas[edge_alphas>1] = 1
    
    nodes       = nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color='blue')
    edges       = nx.draw_networkx_edges(G, pos, node_size=node_sizes, arrowstyle='->',
                                         connectionstyle='arc3, rad = 0.08',
                                         arrowsize=10, edge_color=edge_colors,
                                         width=4,
                                         edge_vmin=vmin, edge_vmax=vmax)
    
    nx.draw_networkx_labels(G, pos, labels, font_size=15, font_color = 'w')
    plt.axis('equal')
    # set alpha value for each edge
    if vmin < 0:       
        from matplotlib.colors import LinearSegmentedColormap
        
        cm_b = plt.get_cmap('Blues', 128)
        cm_r = plt.get_cmap('Reds', 128)
        
        color_list_b = []
        color_list_r = []
        for i in range(128):
            color_list_b.append(cm_b(i))
            color_list_r.append(cm_r(i))
        
        color_list_r = np.array(color_list_r)
        color_list_b = np.flipud(np.array(color_list_b))
        
        color_list   = list(np.concatenate((color_list_b, color_list_r), axis=0))
        
        cm = LinearSegmentedColormap.from_list('custom_cmap', color_list)
            
    elif vmin>=0:
        cm = plt.get_cmap('Reds', 256)
        
    for i in range(M):
        if vmin < 0:
            c_idx = int((weight[i]/vmax + 1)/2 * cm.N)
        elif vmin>=0:
            c_idx = int((edge_alphas[i] * cm.N))
            
        rgb = np.array(cm(c_idx))[0:3]
        # edges[i].set_alpha(edge_alphas[i])
        edges[i].set_color(rgb)
    
    
    plt.xlim([-0.24, 1.24])
    plt.ylim([-0.24, 1.24])
    ax = plt.gca()
    ax.set_axis_off()
    
    
    return edges
#%%
def vis_undirected_graph(K, vmin, vmax):
    import networkx as nx
    import matplotlib as mpl
    
    weight = deepcopy(K).reshape(-1)
    weight = weight[weight != 0]
    
    G      = nx.from_numpy_matrix(K, create_using=nx.MultiDiGraph())
    G.edges(data=True)
    
    pos    = {}
    pos[0] = np.array([0.5, 1.0])
    pos[1] = np.array([0.0, 0.0])
    pos[2] = np.array([1.0, 0.0])
    labels = {i : i + 1 for i in G.nodes()}          
    
    node_sizes  = [800  for i in range(len(G))]
    M           = G.number_of_edges()
    edge_colors = np.ones(M, dtype = int)
    edge_alphas = abs(weight/vmax)
    edge_alphas[edge_alphas>1]  =  1
    # edge_alphas[edge_alphas<-1] = -1
    
    nodes       = nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color='blue')
    edges       = nx.draw_networkx_edges(G, pos, node_size=node_sizes, arrowstyle='-',
                                         edge_color=edge_colors,
                                         edge_cmap=plt.cm.bwr, width=4,
                                         edge_vmin=vmin, edge_vmax=vmax)
    
    nx.draw_networkx_labels(G, pos, labels, font_size=15, font_color = 'w')
    plt.axis('equal')
    # set alpha value for each edge
    
    cm = plt.get_cmap('bwr')
    for i in range(M):
        if vmin < 0:
            c_idx = int((weight[i]/vmax + 1)/2 * cm.N)
        elif vmin>=0:
            c_idx = int((edge_alphas[i] * cm.N))
            
        rgb = np.array(cm(c_idx))[0:3]
        # edges[i].set_alpha(edge_alphas[i])
        edges[i].set_color(rgb)
        
    
    pc = mpl.collections.PatchCollection(edges, cmap=plt.cm.Blues)
    
    plt.xlim([-0.24, 1.24])
    plt.ylim([-0.24, 1.24])
    pc.set_array(edge_colors)
    ax = plt.gca()
    ax.set_axis_off()
#%%
def plot_PRC(theta, dtheta, phi_delta_plot, PRC, K, vmin, vmax, Nosc, ylims=[8, 24]):
    Ni, Nj = K.shape
    
    fig = plt.figure(constrained_layout = False, figsize=(10, 10));
    plt.subplots_adjust(wspace=0.8, hspace=0.8);
    gs  = fig.add_gridspec(Nosc, Nosc)
    
    cnt = 0
    for ref in range(Nosc):
        dphi = deepcopy(dtheta[:,ref])
        for osc in range(Nosc):
            if osc != ref:
                phi_delta = np.mod(deepcopy(theta[:, ref]) - deepcopy(theta[:, osc]), 2*np.pi)
                prc       = PRC[:, :, cnt]
                
                plt.subplot(gs[ref, osc])
                
                plt.scatter(phi_delta, dphi, c = 'b', marker = '.')
                # plt.plot(phi_delta_plot, prc.T, c=[0.5, 0.5, 0.5])
                plt.plot(phi_delta_plot, np.median(prc,axis=0), c='r', linewidth = 3)
                plt.xlabel('$\\theta_{%d} - \\theta_{%d} $'%(osc+1, ref+1))
                plt.ylabel('$d \\theta_{%d} / dt $'%(ref+1))
                plt.xticks([0, np.pi, 2 * np.pi], ['$0$', '$\\pi$', '$2 \\pi$'])
                
                # plt.ylim(8, 24)
                plt.ylim(ylims)
            elif (osc == 0) & (ref == 0) & (Ni == 3):
                plt.subplot(gs[0, 0])
                
                
                if (Ni != Nj) :
                    tmp_ave = (np.median(K,axis=0)).reshape((Nosc, Nosc), order='c')
                    K       = deepcopy(tmp_ave)
                  
                vis_directed_graph(K.T, vmin, vmax)
            cnt += 1

#%% 
def plot_graph(K_tr, K, vmin, vmax):
    Nosc, dummy, Nst = K_tr.shape
    
    
    fig = plt.figure(constrained_layout = False, figsize=(10, 6));
    plt.subplots_adjust(wspace=0.3, hspace=0.3);
    gs  = fig.add_gridspec(2, Nst)
    
    for state in range(Nst):
        #### True graph
        ax1 = fig.add_subplot(gs[0, state])
        vis_directed_graph(K_tr[:,:,state].T, vmin, vmax)
        ax1.set_title('Segment %d\n (true)'%(state+1), fontsize =12)
        
        #### Estimated Graph (epoch average)
        ax2 = fig.add_subplot(gs[1, state])
        vis_directed_graph(K[:,:,state].T, vmin, vmax)
        ax2.set_title('Segment %d\n (pred.)'%(state+1), fontsize =12)
#%%
def vis_heatmap(Mtrx, vmin, vmax, ax, strs, cbar_info, linewidths = 0, fontsize=18): # cbar_info = [True, {"orientation":"horizontal"}, ax_cb]
    import seaborn as sns
    if vmin < 0:       
        from matplotlib.colors import LinearSegmentedColormap
        
        cm_b = plt.get_cmap('Blues', 128)
        cm_r = plt.get_cmap('Reds', 128)
        
        color_list_b = []
        color_list_r = []
        for i in range(128):
            color_list_b.append(cm_b(i))
            color_list_r.append(cm_r(i))
        
        color_list_r = np.array(color_list_r)
        color_list_b = np.flipud(np.array(color_list_b))
        
        color_list   = list(np.concatenate((color_list_b, color_list_r), axis=0))
        
        cm = LinearSegmentedColormap.from_list('custom_cmap', color_list)
            
    elif vmin>=0:
        cm = plt.get_cmap('Reds', 256)
    
    
    title_str = strs[0]
    xlab      = strs[1]
    ylab      = strs[2]
    
    if cbar_info[0] == True:
        im = sns.heatmap(Mtrx, 
                         vmin=vmin, vmax=vmax, linewidths=linewidths, linecolor='whitesmoke',
                         cmap=cm, 
                        cbar = True, cbar_kws = cbar_info[1], 
                        ax=ax, cbar_ax = cbar_info[2]) 
    else:
        im = sns.heatmap(Mtrx, 
                         vmin=vmin, vmax=vmax, linewidths=linewidths, linecolor='whitesmoke',
                         cmap=cm, 
                         cbar = False, 
                         ax=ax) 
    for _, spine in im.spines.items():
           spine.set_visible(True)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title_str, fontsize=fontsize)
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    ax.set_aspect('equal')        
#%%
def plot_graph_heatmap(K1_tr, K2_tr, K1, K2, vmin, vmax, cmaps, twin):
    import seaborn as sns
    Nosc,_,Ncond = K1.shape
    
    
    fig       = plt.figure(constrained_layout = False, figsize=(10, 5.5));
    plt.subplots_adjust(wspace=0.8, hspace=0.8);
    gs        = fig.add_gridspec(3, Ncond+1)
    ratios    = [1,1,0.08]
    gs.set_height_ratios(ratios)
    
    ax_cb     = fig.add_subplot(gs[2, 1:Ncond])
    ############# plot exact couplings
    ### exact Kij in segment 1
    ax1 = fig.add_subplot(gs[0, 0])
    im = sns.heatmap(K1_tr, 
                     vmin=vmin, vmax=vmax, #linewidths=.1, 
                     cmap=cmaps, 
                     cbar = False, 
                     ax=ax1) 
    for _, spine in im.spines.items():
           spine.set_visible(True)
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_title('Segment 1\n(exact)', fontsize=18)
    ax1.set_xlabel('osci. $j$')
    ax1.set_ylabel('osci. $i$')
    ax1.set_aspect('equal')
    
    ### exact Kij in segment 2
    ax2 = fig.add_subplot(gs[1, 0])
    im = sns.heatmap(K2_tr, 
                     vmin=vmin, vmax=vmax, #linewidths=.1, 
                     xticklabels = [5, 10, 15],
                     yticklabels = [5, 10, 15],
                     cmap=cmaps, 
                     cbar = False, 
                     ax=ax2) 
    for _, spine in im.spines.items():
           spine.set_visible(True)
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.set_title('Segment 2\n(exact)', fontsize=18)
    ax2.set_xlabel('osci. $j$')
    ax2.set_ylabel('osci. $i$')
    ax2.set_aspect('equal')
    
    
    for i in range(Ncond):
        if i < Ncond-1:
            ### estimated Kij in segment 1
            ax3 = fig.add_subplot(gs[0, i+1])
            im = sns.heatmap(K1[:,:,i], 
                             vmin=vmin, vmax=vmax, #linewidths=.1, 
                             xticklabels = [5, 10, 15],
                             yticklabels = [5, 10, 15],
                             cmap=cmaps, 
                             cbar = False, 
                             ax=ax3) 
            for _, spine in im.spines.items():
                spine.set_visible(True)
            ax3.set_xticks([])
            ax3.set_yticks([])
            ax3.set_title('\n $T_{win}$ = %2d'%(twin[i]), fontsize=18)
            ax3.set_xlabel('osci. $j$')
            ax3.set_ylabel('osci. $i$')
            ax3.set_aspect('equal')
            
            ### estimated Kij in segment 2
            ax4 = fig.add_subplot(gs[1, i+1])
            im = sns.heatmap(K2[:,:,i], 
                             vmin=vmin, vmax=vmax, #linewidths=.1, 
                             xticklabels = [5, 10, 15],
                             yticklabels = [5, 10, 15],
                             cmap=cmaps, 
                             cbar = False, 
                             ax=ax4) 
            for _, spine in im.spines.items():
                spine.set_visible(True)
            ax4.set_xticks([])
            ax4.set_yticks([])
            ax4.set_title('\n $T_{win}$ = %2d'%(twin[i]), fontsize=18)
            ax4.set_xlabel('osci. $j$')
            ax4.set_ylabel('osci. $i$')
            ax4.set_aspect('equal')
        elif i == Ncond-1:
            ### estimated Kij in segment 1
            ax3 = fig.add_subplot(gs[0, i+1])
            im = sns.heatmap(K1[:,:,i], 
                             vmin=vmin, vmax=vmax, #linewidths=.1, 
                             xticklabels = [5, 10, 15],
                             yticklabels = [5, 10, 15],
                             cmap=cmaps, 
                             cbar = False, 
                             ax=ax3) 
            for _, spine in im.spines.items():
                spine.set_visible(True)
            ax3.set_xticks([])
            ax3.set_yticks([])
            ax3.set_title('\n $T_{win}$ = %2d'%(twin[i]), fontsize=18)
            ax3.set_xlabel('osci. $j$')
            ax3.set_ylabel('osci. $i$')
            ax3.set_aspect('equal')
            
            ### estimated Kij in segment 2
            ax4 = fig.add_subplot(gs[1, i+1])
            im = sns.heatmap(K2[:,:,i], 
                             vmin=vmin, vmax=vmax, #linewidths=.1, 
                             xticklabels = [5, 10, 15],
                             yticklabels = [5, 10, 15],
                             cmap=cmaps, 
                             cbar = True, cbar_kws = {"orientation":"horizontal"}, 
                             ax=ax4, cbar_ax = ax_cb) 
            for _, spine in im.spines.items():
                spine.set_visible(True)
            ax4.set_xticks([])
            ax4.set_yticks([])
            ax4.set_title('\n $T_{win}$ = %2d'%(twin[i]), fontsize=18)
            ax4.set_xlabel('osci. $j$')
            ax4.set_ylabel('osci. $i$')
            ax4.set_aspect('equal')