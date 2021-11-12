% Code used in Vidaurre et al. (2016) NeuroImage
%
% Detailed documentation and further examples can be found in:
% https://github.com/OHBA-analysis/HMM-MAR
% This pipeline must be adapted to your particular configuration of files. 
%%%%%%%%%%%%%%%%%%%%%%%%%
%% SETUP THE MATLAB PATHS AND FILE NAMES
clear 
close all

mydir = './HMM_MAR/'; % adapt to yours
addpath(genpath(mydir))

load('./Nt_3000/phase_data.mat')


data = {cos(theta)};
T = {size(theta,1)};
N =  length(T);


K     = 3; 
Hz    = 100; 
order = 5; 

lgd_labels = cell(1,K);

options = struct();
options.K = K; 
options.Fs = 1/h; 
options.covtype = 'diag';
options.order = order;
options.DirichletDiag = 2; 
options.zeromean = 1;
options.verbose = 1;
options.to_do = [1, 1];


%%
P_cand = 1:12;
L      = zeros(1,length(P_cand));
for n = 1:length(P_cand)
    rng(0)
    options.order = P_cand(n);
    [hmm, Gamma, Xi, vpath, GammaInit,residuals,fehist] = hmmmar(data,T,options);

    fe = hmmfe(data,T,hmm,Gamma, Xi);
    L(1, n) = fe;
end
%%
figure(1)
plot(P_cand, L, 'o-')
%%
% rng(0)
idx_opt   = find(L==min(L));
options.order = P_cand(idx_opt);

[hmm, Gamma, Xi, vpath, GammaInit,residuals,fehist] = hmmmar(data,T,options);
Gamma = single(Gamma);

spectramar = hmmspectramar(data,T,hmm,Gamma,options);
%%
fig = figure(2);

cm     = colormap;
colors = cm(round(linspace(1,size(cm,1),K)),:);

tile = tiledlayout('flow','TileSpacing','compact');
nexttile

for k = 1:K
    plot(t(P_cand(idx_opt)+1:end), Gamma(:,k),'Color',colors(k,:),'LineWidth',3)
    lgd_labels{k} = sprintf('State %d', k);
    
    if k ==1
        hold on
    end
end
hold off
xlabel('time (s)')
ylabel('state activation')

set(gcf,'units','inch','position',[0,0, 18, 4])
set(gca, 'fontsize', 26)
lgd=legend(lgd_labels, 'location', 'northeastoutside');
xticks(linspace(0, t(end) + h, 4))
grid on
print(fig, 'HMM_state.png', '-r800','-dpng');
print(fig, 'HMM_state.eps', '-r800','-depsc');
%%
maxFO = getMaxFractionalOccupancy(Gamma,T,options); % useful to diagnose if the HMM 
FO = getFractionalOccupancy (Gamma,T,options); % state fractional occupancies per session
LifeTimes = getStateLifeTimes (Gamma,T,options); % state life times
Intervals = getStateIntervalTimes (Gamma,T,options); % interval times between state visits
SwitchingRate =  getSwitchingRate(Gamma,T,options); % rate of switching between stats
%%

figure(4) % spectra computed with the multitaper
plot_hmmspectra (spectramar)
