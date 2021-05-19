function LEiDA(path, file)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% LEADING EIGENVECTOR DYNAMICS ANALYSIS (LEiDA)
%
% This function processes, clusters and analyses BOLD data using LEiDA.
% Here the example_BOLD is a dataset containing rest and task conditions
%
% NOTE: Step 4 can be run independently once data is saved by calling
%       LEiDA('LEiDA_results.mat')
%
% 1 - Read the BOLD data from the folders and computes the BOLD phases
%   - Calculate the instantaneous BOLD synchronization matrix
%   - Compute the Leading Eigenvector at each frame from all fMRI scans
% 2 - Cluster the Leading Eigenvectors
% 3 - Compute the probability and lifetimes each cluster in each session

%   - Calculate signigifance between tasks
%   - Saves the Eigenvectors, Clusters and statistics into LEiDA_results.mat
%
% 4 - Plots FC states and errorbars for each clustering solution
%   - Adds an asterisk when results are significantly different between
%   tasks
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Joana Cabral Oct 2017
% joana.cabral@psych.ox.ac.uk
%
% First use in
% Cabral, et al. 2017 Scientific reports 7, no. 1 (2017): 5135.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


    
    %% 1 - Compute the Leading Eigenvectors from the BOLD datasets
    
    
    load([path, file])
    theta = theta.';
    
    n_Subjects = 1;
    n_Task     = 1;
    
    [N_areas, Tmax]=size(theta);
    
     
    % Preallocate variables to save FC patterns and associated information
    Leading_Eig = zeros(Tmax, N_areas); % All leading eigenvectors
    iFC_all     = zeros(N_areas, N_areas, Tmax);
    
    Time_all    = zeros(2, Tmax); % vector with subject nr and task at each t
    t_all=0; % Index of time (starts at 0 and will be updated until n_Sub*Tmax)
    
     
    % [Tmax]=size(BOLD,2); Get Tmax here, if it changes between scans
    Phase = theta;
    for t=1:Tmax

        %Calculate the Instantaneous FC (BOLD Phase Synchrony)
        iFC=zeros(N_areas);
        for n=1:N_areas
            for p=1:N_areas
                iFC(n,p)=cos(Phase(n,t)-Phase(p,t));
            end
        end
        
        % Get the leading eigenvector
        [eVec,eigVal]=eig(iFC);
        eVal=diag(eigVal);
        [val1, i_vec_1] = max(eVal);
        % Calculate the variance explained by the leading eigenvector
%         Var_Eig=val1/sum(eVal);

        % Save V1 from all frames in all fMRI sessions in Leading eig
        t_all=t_all+1; % Update time
        Leading_Eig(t_all,:) = eVec(:,i_vec_1);
        iFC_all(:,:,t_all)   = iFC;
    end
    
    
    clear BOLD tc_aal signal_filt V1 Phase_BOLD
    
    %% 2 - Cluster the Leading Eigenvectors
    
    disp('Clustering the eigenvectors into')
    % Leading_Eig is a matrix containing all the eigenvectors:
    % Collumns: N_areas are brain areas (variables)
    % Rows: Tmax*n_Subjects are all time points (independent observations)
    
    % Set maximum/minimum number of clusters
    % There is no fixed number of states the brain can display
    % Extend the range depending on the hypothesis of each work
    maxk=12;
    mink=2;
    rangeK=mink:maxk;
    
    % Set the parameters for Kmeans clustering
    Kmeans_results=cell(size(rangeK));
    
    for k=1:length(rangeK)      
        disp(['- ' num2str(rangeK(k)) ' clusters'])
        [IDX, C, SUMD, D]=kmeans(Leading_Eig,rangeK(k),'Distance','cityblock','Replicates',20,'MaxIter',1000);%,'Display','off','Options',statset('UseParallel',1));
        Kmeans_results{k}.IDX=IDX;   % Cluster time course - numeric collumn vectos
        Kmeans_results{k}.C=C;       % Cluster centroids (FC patterns)
        Kmeans_results{k}.SUMD=SUMD; % Within-cluster sums of point-to-centroid distances
        Kmeans_results{k}.D=D;       % Distance from each point to every centroid
    end
    
%     save LEiDA_results.mat Leading_Eig Time_all Kmeans_results
    %%
    distM_fcd=squareform(pdist(Leading_Eig,'cityblock'));
    dunn_score=zeros(length(rangeK),1);
    for j=1:length(rangeK)
        dunn_score(j)=dunns(rangeK(j), distM_fcd, Kmeans_results{j}.IDX);
        disp(['Performance for ' num2str(j) 'clusters'])
    end
    [~,ind_max]=max(dunn_score);
    disp(['Best clustering solution: ' num2str(rangeK(ind_max)) ' clusters']);


    clust_labels = Kmeans_results{ind_max}.IDX;
    Kopt         = rangeK(ind_max);
    FCpattern    = Kmeans_results{ind_max}.C;
    save LEiDA_results.mat Leading_Eig iFC_all clust_labels Kopt FCpattern


%% 4 - Plot FC patterns and stastistics between groups
% 
% disp(' ')
% disp('%%% PLOTS %%%%')
% disp(['Choose number of clusters between ' num2str(rangeK(1)) ' and ' num2str(rangeK(end)) ])
% Pmin_pval=min(P_pval(P_pval>0));
% LTmin_pval=min(LT_pval(LT_pval>0));
% if Pmin_pval<LTmin_pval
%    [k,~]=ind2sub([length(rangeK),max(rangeK)],find(P_pval==Pmin_pval));
% else
%    [k,~]=ind2sub([length(rangeK),max(rangeK)],find(LT_pval==LTmin_pval));
% end
% disp(['Note: The most significant difference is detected with K=' num2str(rangeK(k)) ' (p=' num2str(min(Pmin_pval,LTmin_pval)) ')'])  
% 
% % To correct for multiple comparisons, you can divide p by the number of
% % clusters considered
% 
% K = input('Number of clusters: ');
% Best_Clusters=Kmeans_results{rangeK==K};
% k=find(rangeK==K);
% 
% % Clusters are sorted according to their probability of occurrence
% ProbC=zeros(1,K);
% for c=1:K
%     ProbC(c)=mean(Best_Clusters.IDX==c);
% end
% [~, ind_sort]=sort(ProbC,'descend'); 
% 
% % Get the K patterns
% V=Best_Clusters.C(ind_sort,:);
% [~, N]=size(Best_Clusters.C);
% Order=[1:2:N N:-2:2];
% 
% figure
% colormap(jet) 
% % Pannel A - Plot the FC patterns over the cortex 
% % Pannel B - Plot the FC patterns in matrix format
% % Pannel C - Plot the probability of each state in each condition
% % Pannel D - Plot the lifetimes of each state in each condition
%    
% for c=1:K
%     subplot(4,K,c)
%     % This needs function plot_nodes_in_cortex.m and aal_cog.m
%     plot_nodes_in_cortex(V(c,:))
%     title({['State #' num2str(c)]})
%     subplot(4,K,K+c)
%     FC_V=V(c,:)'*V(c,:);  
%     li=max(abs(FC_V(:)));
%     imagesc(FC_V(Order,Order),[-li li])   
%     axis square
%     title('FC pattern') 
%     ylabel('Brain area #')
%     xlabel('Brain area #')   
%     
%     subplot(4,K,2*K+c)  
%             Rest=squeeze(P(1,:,k,ind_sort(c)));
%             Task=squeeze(P(2,:,k,ind_sort(c)));
%             bar([mean(Rest) mean(Task)],'EdgeColor','w','FaceColor',[.5 .5 .5])
%             hold on
%             % Error bar containing the standard error of the mean
%             errorbar([mean(Rest) mean(Task)],[std(Rest)/sqrt(numel(Rest)) std(Task)/sqrt(numel(Task))],'LineStyle','none','Color','k')
%             set(gca,'XTickLabel',{'Rest', 'Task'})
%             if P_pval(k,ind_sort(c))<0.05
%                 plot(1.5,max([mean(Rest) mean(Task)])+.01,'*k')
%             end             
%             if c==1
%                 ylabel('Probability')
%             end
%             box off
%             
%      subplot(4,K,3*K+c)  
%             Rest=squeeze(LT(1,:,k,ind_sort(c)));
%             Task=squeeze(LT(2,:,k,ind_sort(c)));
%             bar([mean(Rest) mean(Task)],'EdgeColor','w','FaceColor',[.5 .5 .5])
%             hold on
%             errorbar([mean(Rest) mean(Task)],[std(Rest)/sqrt(numel(Rest)) std(Task)/sqrt(numel(Task))],'LineStyle','none','Color','k')
%             set(gca,'XTickLabel',{'Rest', 'Task'})
%             if LT_pval(k,ind_sort(c))<0.05
%                 plot(1.5,max([mean(Rest) mean(Task)])+.01,'*k')
%             end             
%             if c==1
%                 ylabel('Lifetime (seconds)')
%             end
%             box off           
% end