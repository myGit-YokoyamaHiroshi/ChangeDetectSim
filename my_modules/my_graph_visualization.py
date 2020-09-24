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
    pos[1] = np.array([1.0, 0.0])
    pos[2] = np.array([0.0, 0.0])
    labels = {i : i + 1 for i in G.nodes()}          
    
    node_sizes  = [800  for i in range(len(G))]
    M           = G.number_of_edges()
    edge_colors = range(2, M+2)
    edge_alphas = weight/vmax
    edge_alphas[edge_alphas>1] = 1
    
    nodes       = nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color='blue')
    edges       = nx.draw_networkx_edges(G, pos, node_size=node_sizes, arrowstyle='->',
                                         connectionstyle='arc3, rad = 0.08',
                                         arrowsize=10, edge_color=edge_colors,
                                         edge_cmap=plt.cm.Blues, width=4,
                                         edge_vmin=0.0, edge_vmax=vmax)
    
    nx.draw_networkx_labels(G, pos, labels, font_size=15, font_color = 'w')
    plt.axis('equal')
    # set alpha value for each edge
    for i in range(M):
        edges[i].set_alpha(edge_alphas[i])
    
    pc = mpl.collections.PatchCollection(edges, cmap=plt.cm.Blues)
    
    
    pc.set_array(edge_colors)
    ax = plt.gca()
    ax.set_axis_off()
#%%
def plot_PRC(theta, dtheta, phi_delta_plot, PRC, K, vmin, vmax, Nosc):
    
    fig = plt.figure(constrained_layout = False, figsize=(10, 10));
    plt.subplots_adjust(wspace=0.8, hspace=0.5);
    gs  = fig.add_gridspec(Nosc, Nosc)
    
    cnt = 0
    for ref in range(Nosc):
        dphi = deepcopy(dtheta[:,ref])
        for osc in range(Nosc):
            if osc != ref:
                phi_delta = np.mod(deepcopy(theta[:, osc]) - deepcopy(theta[:, ref]), 2*np.pi)
                prc       = PRC[:, :, cnt]
                
                plt.subplot(gs[ref, osc])
                
                plt.scatter(phi_delta, dphi, c = 'b', marker = '.')
                # plt.plot(phi_delta_plot, prc.T, c=[0.5, 0.5, 0.5])
                plt.plot(phi_delta_plot, np.median(prc,axis=0), c='r', linewidth = 3)
                plt.xlabel('$\\phi_{%d} - \\phi_{%d} $'%(osc+1, ref+1))
                plt.ylabel('$d \\phi_{%d} / dt $'%(ref+1))
                plt.xticks([0, np.pi, 2 * np.pi], ['$0$', '$\\pi$', '$2 \\pi$'])
                
                # plt.ylim(8, 24)
                plt.ylim(20, 35)
            elif (osc == 0) & (ref == 0):
                plt.subplot(gs[0, 0])
                
                Ni, Nj = K.shape
                if Ni != Nj:
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