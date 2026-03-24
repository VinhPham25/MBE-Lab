% Code to perform GC analysis on the ASTaUND dataset from https://github.com/anilkamat/GC_FLS_PC 

%% setups
clc;clear all; close all;
addpath C:\Users\_Kamat_\Desktop\RPI\ResearchWork\Papers_\Data\Data_processed_BU\MGH_bpfilter_GLM\MGH_NIRS_GLM_bpFilter % processed data directory
addpath C:\Users\_Kamat_\Desktop\RPI\ResearchWork\Papers_\Effective_Connectivity\GCCA             % toolbox directory
addpath C:\Users\_Kamat_\Desktop\RPI\ResearchWork\Papers_\Effective_Connectivity\GCCA\utilities   % toolbox utility directory
addpath C:\Users\_Kamat_\Desktop\RPI\ResearchWork\Papers_\Effective_Connectivity\bsmart           % toolbox utility directory
addpath C:\Users\_Kamat_\Desktop\RPI\ResearchWork\Papers_\Effective_Connectivity\toolbox_original\mvgc_v1.0\demo % toolbox directory
addpath (genpath('C:\\Users\\_Kamat_\\Desktop\\RPI\\ResearchWork\\Papers_\\Homer3-master')) % Add the Homer3 parent directory

Cnames = {'LPFC' 'RPFC' 'LPMC' 'RPMC' 'SMA'};
connections = cell(5,5);
for i =1:5
    for j = 1:5
        connections{j,i} = sprintf('%s --> %s ',Cnames{i},Cnames{j});
    end
end
conn_temp = reshape(connections,25,1);
conn_temp([1 7 13 19 25],:)= [];
% fpath1 = 'D:\\RPI\\ResearchWork\\Papers_\\Effective_Connectivity\\toolbox_original\\mvgc_v1.0\\Plots_TRGC_Nov_Exp\\signals';
fpath1 = 'D:\\RPI\\ResearchWork\\Papers_\\Effective_Connectivity\\toolbox_original\\mvgc_v1.0\\Plots_TRGC_Nov_Exp\\signals_with_GLM_bandpassfilter';
fpath2 = 'D:\\RPI\\ResearchWork\\Papers_\\Effective_Connectivity\\toolbox_original\\mvgc_v1.0\\Results_TRGC_Nov_Exp';

fpath = 'D:\\RPI\\ResearchWork\\Papers_\\Effective_Connectivity\\toolbox_original\\mvgc_v1.0\\Plots_TRGC\\FLS_conn_and_VBLAST_conn_win_40s';
connection_GC = {'LPFC-->RPFC','LPFC-->LPMC','LPFC-->RPMC','LPFC-->SMA',...
    'RPFC-->LPFC','RPFC-->LPMC','RPFC-->RPMC','RPFC-->SMA',...
    'LPMC-->LPFC','LPMC-->RPFC','LPMC-->RPMC','LPMC-->SMA',...
    'RPMC-->LPFC','RPMC-->RPFC','RPMC-->LPMC','RPMC-->SMA',...
    'SMA-->LPFC','SMA-->RPFC','SMA-->LPMC','SMA-->RPMC'};
Exp_sub = {'R7','R8','R9','R11','R12','A1','A2'};   % subjects
Nov_sub = {'R1','R2','R3','R4','R5','R6'};          % subjects
%% ############################################# GC Analysis  ########################################################%%%
nos.Exp = 7;  
nos.Nov = 6;  % 6-> FLS subjects, 6-> VBLaST
GC_f1 = [];
proceed = 1;  % Flag 
Wn = 1;
%n_winSize = [];
count = 1;
d  = 1;                 % counter for task duration
tic;
for m = 1:2  % 1 for Exp and 2 for Nov and exp
    if m ==1
        N = nos.Exp;
    else
        N = nos.Nov;
    end
    ss = 7;
    h = 1;
    hh = 1;
    for s = 1:N  % no of Subjects (N)
        % NOTE: for physical and virtual analysis the sub_trial of both
        % cohorts needs to be changed ex: sub_trial      =
        % sprintf('Exp_%d-FLS.mat',s); or sub_trial      = sprintf('Exp_%d-VBLAST.mat',s);
        if m == 1
            sub_trial      = sprintf('Exp_%d-VBLAST.mat',s);
            sub_trial_nirs = sprintf('Exp_%d-VBLAST.nirs',s);
            sub_trial_name = sprintf('Exp_%d_Exp',s);
        else
            sub_trial      = sprintf('Nov_%d-VBLAST.mat',s);
            sub_trial_nirs = sprintf('Nov_%d-VBLAST.nirs',s);
            sub_trial_name = sprintf('Nov_%d_Nov',s);
        end
        fprintf('Analyzing %s \n',sub_trial_name);
        if exist(sub_trial , 'file') == 2
            DD = load(sub_trial);
            da = load(sub_trial_nirs,'-mat');
        else
            fprintf('Subject %s doesnot exist in the directory',sub_trial)
            continue;
        end
        st = find(da.s);    % for trials, stimulus trigger points
        size_st = size(st,1);
        %Data = (DD.output.dc.dataTimeSeries)'; % (:,1:33)
        %             Data = (DD.output.misc.dcNew.dataTimeSeries(:,1,:));
        %             Data = permute(Data,[1,3,2])';
        Data = (DD.output.dc.dataTimeSeries)';
        k = 1;
        D = [];
        channel_sd = [];
        for i = 1:3:96
            D(k,:) = Data(i,:);
            channel_sd(1,k) = DD.output.dc.measurementList(1, i).sourceIndex;
            channel_sd(2,k) = DD.output.dc.measurementList(1, i).detectorIndex;
            k      = k+1;
        end
        D = Data;
        LPFC    = D([1 2],:);           % select channels belonging to the ROIs
        LPFC( any(isnan(LPFC),2),:) = [];
        LPFC    = mean(LPFC,1);
        RPFC    = D([7 8],:);           
        RPFC( any(isnan(RPFC),2),:) = [];
        RPFC    = mean(RPFC,1);
        LPMC    = D([ 10 11 ],:);                
        LPMC( any(isnan(LPMC),2),:) = [];
        LPMC    = mean(LPMC,1);
        RPMC    = D([27 28],:);                 
        RPMC( any(isnan(RPMC),2),:) = [];
        RPMC    = mean(RPMC,1);
        SMA     = D([29 30 31],:);
        SMA( any(isnan(SMA),2),:) = [];
        SMA     = mean(SMA,1);
        Data_Reg = [LPFC;RPFC;LPMC;RPMC;SMA];
        valid_Ch_names = Cnames;
        NaN_rows = find(all(isnan(Data_Reg),2));
        valid_Ch_names(:,NaN_rows) = [];
        Data_Reg( any(isnan(Data_Reg),2),:) = [];
        %D_task = Data_Reg(:,w(1):w(2));   % data select for task depends on w
        %D_rest = Data_Reg(:,1:w(1));   % data select for first resting state
        k = rank(D');
        Z = null(D');
        freq = [0:0.01:0.1];           % Frequency range for GC computation
        Tr = 1;                        % Trial counter
        for stt = 1:2:(size_st-1)      % original 1:2:(size_st-1) 
            duration_task(d) = (st(stt+1)-st(stt))/25;      % duration in seconds
            trial = sprintf('Tr_%d',Tr);
            d = d+1;
            Y = Data_Reg(:,st(stt):st(stt+1));      % Task data
            %Y = Data_Reg(:,st(stt+1):st(stt+2));      % Rest data
            LY = size(Y,2);
            fprintf('length of data %d',LY);
            window_size = 54*25;            % 54 sec with 25 sampling frequency
            sT = window_size;               % stride of 27 sec i.e. 50% overlap
            Wn = 0;                         % window number
            for w = window_size+1:sT:LY         
                close all;
                Y_windowed = Y(:,w-window_size:w);                
                W = sprintf('Win_%d',Wn+1)
                fprintf('Win_%d\n',Wn);
                NL = size(Y_windowed,2);
                [GW,COH,pp,waut,cons]= cca_pwcausal(Y_windowed,1,NL,20,20,freq, 1);
                %fprintf('Reached hear')
                GC_f1.(sub_trial_name) = GW;     % for freq1
                %mean in the neurophysiology frequency band.
                idx1 = find(freq == 0.01);
                idx2 = find(freq == 0.07);
                if m == 1
                    GC_fqmean.FLS.(sub_trial_name).(trial).(W)   = mean(GW(:,:,idx1:idx2),3);
                    sz = numel(GC_fqmean.FLS.(sub_trial_name).(trial).(W));
                    B_Exp.(trial).Exp(h,:,Wn+1)  = reshape(GC_fqmean.FLS.(sub_trial_name).(trial).(W),1,sz);   % B_phy -> sub x connection x window
                else
                    GC_fqmean.VBLaST.(sub_trial_name).(trial).(W) = mean(GW(:,:,idx1:idx2),3);
                    sz = numel(GC_fqmean.VBLaST.(sub_trial_name).(trial).(W));
                    B_Nov.(trial).Nov(hh,:,Wn+1)  = reshape(GC_fqmean.VBLaST.(sub_trial_name).(trial).(W),1,sz);
                end
                Wn = Wn+1;
            end
            Tr = Tr+1;
        end
        h = h+1;
        hh = hh+1;
    end
    fprintf('%s \n',sub_trial);
    if m ==1
        nWinFLS(s) = Wn;
    else
        nWinVBLAST(s) = Wn;
    end
end
toc;
%%
for i = 1:size(fieldnames(B_Exp),1)
    trial_temp = sprintf('Tr_%d',i);
    B_Exp.(trial_temp).Exp(:,[1 7 13 19 25],:) = [];       % Remove all the diagonals
end
for i = 1:size(fieldnames(B_Nov),1)
    trial_temp = sprintf('Tr_%d',i);
    B_Nov.(trial_temp).Nov(:,[1 7 13 19 25],:) = [];       % Remove all the diagonals
end
%% ################  GC--ANOVA test  ############
% Saphiro-Wilks normality test for each connection in Exp and Nov
nnn = size(fieldnames(B_Exp),1);
for k = 1:nnn           % trials
    trial_temp = sprintf('Tr_%d',k);
    nn = size(B_Exp.(trial_temp).Exp,3);
    for j = 1:nn        % windows
        for i = 1:20        % channels
            [~,NORM_p.(trial_temp).ExpFLS(i,1,j)] = swtest(B_Exp.(trial_temp).Exp(:,i,j));
        end
    end
end
% NovFLS
nnn = size(fieldnames(B_Nov),1);
for k = 1:nnn
    trial_temp = sprintf('Tr_%d',k);
    nn = size(B_Nov.(trial_temp).Nov,3);
    for j = 1:nn
        for i = 1:20
            [~,NORM_p.(trial_temp).NovFLS(i,1,j)] = swtest(B_Nov.(trial_temp).Nov(:,i,j));
        end
    end
end
%% ##########################  HIII:inter and intra connectivity ############################################
p11 = [];
nn = 2; %size(B_Exp,3);
p_interHemisExp = [];
p_interHemisNov = [];
for m = 1:2
    if m == 1
        qq = size(fieldnames(B_Exp),1);
    else
        qq = size(fieldnames(B_Nov),1);
    end
    for q = 1:qq
        trial = sprintf('Tr_%d',q)
        for j = 1:nn
            l = 1;
            for i = 1:20
                close all;
                for k = 1:20
                    if i ~= k
                        if m == 1
                            X = [B_Exp.(trial).Exp(:,i,j),B_Exp.(trial).Exp(:,k,j)];
                            p_interHemisExp.(trial).Exp(j,1:2) =  nanmean(X); % Mean1 mean of first column i.e. connection ij
                            p_interHemisExp.(trial).Exp(j,3:4) =  std(X); % Mean1 mean of first column i.e. connection ij
                            [p_interHemisExp.(trial).Exp(j,6),tbl,~] = anova1(X);     % p11 -> window x inter connection
                            p_interHemisExp.(trial).Exp(j,5) =  tbl{2,5}; % F-Statistics
                            l = l+1;
                        else
                            %label{temp} = [connection_GC{i},' <=>', connection_GC{k}];
                            X = [B_Nov.(trial).Nov(:,i,j),B_Nov.(trial).Nov(:,k,j)];
                            p_interHemisNov.(trial).Nov(j,1:2) =  nanmean(X); % Mean1 mean of first column i.e. connection ij
                            p_interHemisNov.(trial).Nov(j,3:4) =  std(X); % Mean1 mean of first column i.e. connection ij
                            [p_interHemisNov.(trial).Nov(j,6),tbl,~] = anova1(X);     % p11 -> window x inter connection
                            p_interHemisNov.(trial).Nov(j,5) =  tbl{2,5}; % F-Statistics
                            l = l+1;
                        end
                        %                 title('One way ANOVA between all GC_connection in Physical')
                        %                 ylabel('GC freq[0.01,0.07]')
                        %                 xticklabels(connection_GC);
                        %                 xtickangle(45)
                    end
                end
            end
        end
    end
end
% l = 1;
% for i = 1:20    % label generation
%     for j = 1:20
%         if i ~= j
%             label{l} = [connection_GC{i},' <=>', connection_GC{k}];
%             l = l+1;
%         end
%     end
% end
%%
%With in VBLaST;
nn = size(B_Nov,3);
for j = 1:nn
    for i = 1:20
        close all;
        [p22(i,1,j),tbl2,stats22] = anova1(B_Nov(:,i,j));
        %title('One way ANOVA between all GC_connection in virtual')
        %ylabel('GC freq[0.01,0.07]')
        %xticklabels(connection_GC);
        %xtickangle(45)
    end
end
%%
%With in FLS;
nn = size(B_Exp,3);
for j = 1:nn
    for i = 1:20
        close all;
        [p11(i,1,j),tbl1,stats11] = anova1(B_Exp(:,i,j));
        %         title('One way ANOVA between all GC_connection in Physical')
        %         ylabel('GC freq[0.01,0.07]')
        %         xticklabels(connection_GC);
        %         xtickangle(45)
    end
end

%With in VBLaST;
nn = size(B_Nov,3);
for j = 1:nn
    for i = 1:20
        close all;
        [p22(i,1,j),tbl2,stats22] = anova1(B_Nov(:,i,j));
        %         title('One way ANOVA between all GC_connection in virtual')
        %         ylabel('GC freq[0.01,0.07]')
        %         xticklabels(connection_GC);
        %         xtickangle(45)
    end
end

%% ################ H1: pair wise anova test between Novice and Expert on fls or virtual simulator ########################
group = {'FLS';'VBLaST'};
%n = min(nos.Exp, nos.Nov);
nnn = size(fieldnames(B_Exp),1);  % no. of common trial between nov and exp
X = [];
p1 = [];
baseFileName = 'H2_Nov_Exp_Phy.xlsx';          % excel write
fpath2 = 'D:\RPI\ResearchWork\Papers_\Effective_Connectivity\toolbox_original\mvgc_v1.0\Results_TRGC_Nov_Exp_Nature\P_value_corrected_FDR';
fullFileName = fullfile(fpath2, baseFileName);
sheet = [];
col_name = 'A':'Z';
for k = 1:1  % nnn loop over trial
    col = 1;
    trial_temp = sprintf('Tr_%d',k)
    %     if k > 4
    %        n = 5; % after five trail, no.of subj in VBLAST drops to 5
    %     end
    nn = min(size(B_Nov.(trial_temp).Nov,3),size(B_Exp.(trial_temp).Exp,3));
    for j = 1:nn    % nn window no.
        sheet = sprintf('Phy_T%dWin_%d',k,j);
        close all;
        for i = 1:20    % 20 connection
            n = min(size(B_Exp.(trial_temp).Exp(:,i,j),1), size(B_Nov.(trial_temp).Nov(:,i,j),1));
            X = [B_Exp.(trial_temp).Exp(1:n,i,j), B_Nov.(trial_temp).Nov(1:n,i,j)];
            %tbl = ['tbl',num2str(i)];
            [p1.(trial_temp)(i,6,j),tbl,~] = anova1(X);
            p1.(trial_temp)(i,1,j) =  nanmean(X(:,1)); % Mean mean of first column i.e. connection ij
            p1.(trial_temp)(i,2,j) =  std(X(:,1)); % std of first column i.e. connection ij
            p1.(trial_temp)(i,3,j) =  nanmean(X(:,2)); % Mean mean of second column i.e. connection ij
            p1.(trial_temp)(i,4,j) =  std(X(:,2)); % std of second column i.e. connection ij
            p1.(trial_temp)(i,5,j) =  tbl{2,5}; % F-Statistics
            %figure(10)
            %baseFileName = sprintf('FLS vs VBLaST Aonva Conn:-%s.png',connection_GC{i});
            %             title(baseFileName)
            %             xticklabels(group);
            %             xtickangle(45)
            %             fullFileName = fullfile(fpath, baseFileName);
            %saveas(figure(10),fullFileName)
        end
        [FDR_temp,Q_temp] = mafdr(p1.(trial_temp)(:,6,j));  % FDR correction on p-value
        p1.(trial_temp)(:,6,j) = FDR_temp;
        p1.(trial_temp)(:,7,j) = Q_temp;
        
        %fnPlot(p1.(trial_temp)(:,1,j), trial_temp,j,Cnames,fpath2);
        %xlswrite(fullFileName,p1.(trial_temp)(:,:,j),sheet)
        col = col+6;
        pause;
    end
end
close all;
%% ################ H0: Connectivity doesn't changes during single trial ###########################################
for m = 1:2  % 1-> expert, 2-> Novice
    P_inTrial = [];
    if m == 1
        nnn = size(fieldnames(B_Exp),1);  % no. of trial in Expert
        %n = nos.Exp;
    else
        nnn = size(fieldnames(B_Nov),1);  % no. of trial in Novice
        %n = nos.Nov;
    end
    for k = 1:nnn  % nnn loop over trial
        close all;
        col = 1;
        trial_temp = sprintf('Tr_%d',k);
        %nn = min(size(B_Nov.(trial_temp).Nov,3),size(B_Exp.(trial_temp).Exp,3)); % no. of windows having subjects >= 5
        if m == 1
            nn = size(B_Exp.(trial_temp).Exp,3);
            n = size(B_Exp.(trial_temp).Exp,1);
            sheet = sprintf('vir_Exp_T%d',k);
        else
            nn = size(B_Nov.(trial_temp).Nov,3);
            n = size(B_Nov.(trial_temp).Nov,1);
            sheet = sprintf('vir_Nov_T%d',k);
        end
        for j = 1:nn
            if m == 1
                zer_col = sum(B_Exp.(trial_temp).Exp(1:n,1,j) == 0); % no of zeros in each column
            else
                zer_col = sum(B_Nov.(trial_temp).Nov(1:n,1,j) == 0); % no of zeros in each column
            end
            if zer_col > 2
                nn = j-1;
                break;
            end
        end
        for i = 1:20    % 20 connection
            if m ==1
                X = squeeze(B_Exp.(trial_temp).Exp(1:n,i,1:nn)); % groups all the windows <= 2 zeros
                kkk = 1;
                for iii = 1:nn % to store the statistics over windows
                    P_inTrial.(trial_temp)(i,iii+kkk) =  nanmean(X(:,iii)); % Mean mean of first column i.e. connection ij
                    P_inTrial.(trial_temp)(i,iii+kkk+1) =  std(X(:,iii)); % std of first column i.e. connection ij
                    kkk = kkk+1;
                end
                [P_inTrial.(trial_temp)(i,2*kkk+1),tbl,~] = anova1(X);    % pval -> trial x connection
                P_inTrial.(trial_temp)(i,2*kkk) =  tbl{2,5}; % F-Statistics
            else
                X = squeeze(B_Nov.(trial_temp).Nov(1:n,i,1:nn));
                kkk = 1;
                for iii = 1:nn % to store the statistics over windows
                    P_inTrial.(trial_temp)(i,iii+kkk) =  nanmean(X(:,iii)); % Mean mean of first column i.e. connection ij
                    P_inTrial.(trial_temp)(i,iii+kkk+1) =  std(X(:,iii)); % std of first column i.e. connection ij
                    kkk = kkk+1;
                end
                [P_inTrial.(trial_temp)(i,2*kkk+1),tbl,~] = anova1(X);    % pval -> trial x connection
                P_inTrial.(trial_temp)(i,2*kkk) =  tbl{2,5}; % F-Statistics
            end
            %fnPlot(p1.(trial_temp)(:,1,j), trial_temp,j,Cnames,fpath2);
        end
        pause;
        [FDR_temp,Q_temp] = mafdr(P_inTrial.(trial_temp)(:,2*kkk+1));  % FDR correction on p-value
        P_inTrial.(trial_temp)(:,2*kkk+1) = FDR_temp;
        P_inTrial.(trial_temp)(:,2*kkk+2) = Q_temp;
        baseFileName = 'H0_Nov_Exp.xlsx';          % excel write
        fpath2 = 'D:\RPI\ResearchWork\Papers_\Effective_Connectivity\toolbox_original\mvgc_v1.0\Results_TRGC_Nov_Exp_Nature\P_value_corrected_FDR';
        fullFileName = fullfile(fpath2, baseFileName);
        %xlswrite(fullFileName,P_inTrial.(trial_temp),sheet)
    end
end
close all;
%% ################ H4 : inter and intra connectivity are same #####################
iN = 1; iL = 1; iR = 1;
Inter = []; Intra = [];
for i =1:20
    if (connection_GC{i}(1) == 'L' && connection_GC{i}(1) ==connection_GC{i}(8))
        Intra.left(iL) = i;
        iL = iL+1;
    elseif (connection_GC{i}(1) == 'R' && connection_GC{i}(1) ==connection_GC{i}(8))
        Intra.right(iR) = i;
        iR = iR+1;
    else
        Inter(iN) = i;
        iN = iN+1;
    end
end

for m = 1:1  % 1-> expert, 2-> Novice
    if m == 1
        nnn = size(fieldnames(B_Exp),1);  % no. of trial in Expert
        %n = nos.Exp;
    else
        nnn = size(fieldnames(B_Nov),1);  % no. of trial in Novice
        %n = nos.Nov;
    end
    for k = 1:nnn  % nnn loop over trial
        close all;
        col = 1;
        trial_temp = sprintf('Tr_%d',k);
        %nn = min(size(B_Nov.(trial_temp).Nov,3),size(B_Exp.(trial_temp).Exp,3)); % no. of windows having subjects >= 5
        if m == 1
            nn = size(B_Exp.(trial_temp).Exp,3);
            n  = size(B_Exp.(trial_temp).Exp,1);
        else
            nn = size(B_Nov.(trial_temp).Nov,3);
            n  = size(B_Nov.(trial_temp).Nov,1);
        end
        for j = 1:nn    % loop for considering only the windows with no more than 2 zeros connectivity
            if m == 1
                zer_col = sum(B_Exp.(trial_temp).Exp(1:n,1,j) == 0); % no of zeros in each column
            else
                zer_col = sum(B_Nov.(trial_temp).Nov(1:n,1,j) == 0); % no of zeros in each column
            end
            if zer_col > 2
                nn = j-1;
                break;
            end
        end
        % intra and inter connectivity
        for mm = 1:1    % 1-> left, 2-> Right and 3 -> inter
            if mm ==1
                nI = size(Intra.left,2);        %nI -> no. of inter/intra connections
            elseif mm == 2
                nI = size(Intra.right,2);
            else
                nI = size(Inter,2);
            end
            for i = 1:nI
                if m == 1
                    if mm == 1
                        X = squeeze(B_Exp.(trial_temp).Exp(1:n,Intra.left(i),1:nn)); % groups all the windows <= 2 zeros
                        [P_intraLeft.Exp(k,i),tbl,~] = anova1(X);
                        %                         kkk = 1;
                        %                         N = size(X,2);
                        %                         for iii = 1:N % to store the statistics over each trial
                        %                             P_intraLeft.(Win).Exp(i,iii+kkk) =  nanmean(X(:,iii)); % Mean mean of first column i.e. connection ij
                        %                             P_intraLeft.(Win).Exp(i,iii+kkk+1) =  std(X(:,iii)); % std of first column i.e. connection ij
                        %                             kkk = kkk+1;
                        %                         end
                        %                         [P_intraLeft.(Win).Exp(i,2*kkk+2),tbl,~] = anova1(X);    % pval -> connection x windows
                        %                         P_intraLeft.(Win).Exp(i,2*kkk) =  tbl{2,5}; % F-Statistics
                        %                         sheet = sprintf('Nov_Win%d',j);
                    elseif mm == 2
                        X = squeeze(B_Exp.(trial_temp).Exp(1:n,Intra.right(i),1:nn));
                        [P_intraRight.Exp(k,i),tbl,~] = anova1(X);
                    else
                        X = squeeze(B_Exp.(trial_temp).Exp(1:n,Inter(i),1:nn));
                        [P_inter.Exp(k,i),tbl,~] = anova1(X);
                    end
                else
                    if mm == 1
                        X = squeeze(B_Nov.(trial_temp).Nov(1:n,Intra.left(i),1:nn));
                        [P_intraLeft.Nov(k,i),tbl,~] = anova1(X);
                    elseif mm == 2
                        X = squeeze(B_Nov.(trial_temp).Nov(1:n,Intra.right(i),1:nn));
                        [P_intraRight.Nov(k,i),tbl,~] = anova1(X);
                    else
                        X = squeeze(B_Nov.(trial_temp).Nov(1:n,Inter(i),1:nn));
                        [P_inter.Nov(k,i),tbl,~] = anova1(X);
                    end
                end
            end
        end
        %sheet = sprintf('vir_T%d',k+1);
    end
end
%% ################ H4 : inter an intra connectivity are same --> modified (don't use)#####################
iN = 1; iL = 1; iR = 1;
Inter = []; Intra = [];
for i =1:20
    if (connection_GC{i}(1) == 'L' && connection_GC{i}(1) ==connection_GC{i}(8))
        Intra.left(iL) = i;
        iL = iL+1;
    elseif (connection_GC{i}(1) == 'R' && connection_GC{i}(1) ==connection_GC{i}(8))
        Intra.right(iR) = i;
        iR = iR+1;
    else
        Inter(iN) = i;
        iN = iN+1;
    end
end

for m = 1:2  % 1-> expert, 2-> Novice
    if m == 1
        nnn = size(fieldnames(B_Exp),1);  % no. of trial in Expert
        %n = nos.Exp;
    else
        nnn = size(fieldnames(B_Nov),1);  % no. of trial in Novice
        %n = nos.Nov;
    end
    for k = 1:nnn  % nnn loop over trial
        close all;
        col = 1;
        trial_temp = sprintf('Tr_%d',k);
        %nn = min(size(B_Nov.(trial_temp).Nov,3),size(B_Exp.(trial_temp).Exp,3)); % no. of windows having subjects >= 5
        if m == 1
            nn = size(B_Exp.(trial_temp).Exp,3);
            n  = size(B_Exp.(trial_temp).Exp,1);
        else
            nn = size(B_Nov.(trial_temp).Nov,3);
            n  = size(B_Nov.(trial_temp).Nov,1);
        end
        for j = 1:nn    % loop for considering only the windows with no more than 2 zeros connectivity
            if m == 1
                zer_col = sum(B_Exp.(trial_temp).Exp(1:n,1,j) == 0); % no of zeros in each column
            else
                zer_col = sum(B_Nov.(trial_temp).Nov(1:n,1,j) == 0); % no of zeros in each column
            end
            if zer_col > 2
                nn = j-1;
                break;
            end
        end
        % intra and inter connectivity
        for mm = 1:3    % 1-> left, 2-> Right and 3 -> inter
            if mm ==1
                nI = size(Intra.left,2)-1;        %nI -> no. of inter/intra connections
            elseif mm == 2
                nI = size(Intra.right,2)-1;
            else
                nI = size(Inter,2);
            end
            for i = 1:nI
                if m == 1
                    if mm == 1
                        X = [B_Exp.(trial_temp).Exp(1:n,Intra.left(i),1), B_Exp.(trial_temp).Exp(1:n,Intra.left(i+1),1)]; % groups all the windows <= 2 zeros
                        [P_intraLeft.Exp(k,i),tbl,~] = anova1(X);
                    elseif mm == 2
                        X = [B_Exp.(trial_temp).Exp(1:n,Intra.right(i),1), B_Exp.(trial_temp).Exp(1:n,Intra.right(i+1),1)];;
                        [P_intraRight.Exp(k,i),tbl,~] = anova1(X);
                    else
                        X = squeeze(B_Exp.(trial_temp).Exp(1:n,Inter(i),1:nn));
                        [P_inter.Exp(k,i),tbl,~] = anova1(X);
                    end
                else
                    if mm == 1
                        X = squeeze(B_Nov.(trial_temp).Nov(1:n,Intra.left(i),1:nn));
                        [P_intraLeft.Nov(k,i),tbl,~] = anova1(X);
                    elseif mm == 2
                        X = squeeze(B_Nov.(trial_temp).Nov(1:n,Intra.right(i),1:nn));
                        [P_intraRight.Nov(k,i),tbl,~] = anova1(X);
                    else
                        X = squeeze(B_Nov.(trial_temp).Nov(1:n,Inter(i),1:nn));
                        [P_inter.Nov(k,i),tbl,~] = anova1(X);
                    end
                end
            end
        end
        %sheet = sprintf('vir_T%d',k+1);
    end
end
%% ####################3 HIV: Connectivity stays same across trials ###############
baseFileName = 'H4_Nov_Exp_Phy.xlsx';          % on physical simulator
%baseFileName = 'H4_Nov_Exp.xlsx';          % on virtual simulator
for m = 1:2  % 1-> expert, 2-> Novice
    P_AcrossTrial = [];
    if m == 1
        nnn = size(fieldnames(B_Exp),1);  % no. of trial in Expert
        %n = nos.Exp;
    else
        nnn = size(fieldnames(B_Nov),1);  % no. of trial in Novice, last trial of novice only has 1 subj
        %n = nos.Nov;
    end
    for j = 1:3     %for j = 1:2; % window number
        Win = sprintf('Win_%d',j);
        X = [];
        for i = 1:20    % 20 connection
            t_count = 1;    % counts trials
            for k = 1:nnn  % nnn loop over trial
                close all;
                trial_temp = sprintf('Tr_%d',k)
                if m == 1
                    nn = size(B_Exp.(trial_temp).Exp,3);
                    n  = size(B_Exp.(trial_temp).Exp,1)
                    if n >=5
                        X(:,k) = B_Exp.(trial_temp).Exp(1:n,i,j); % for first windows of all trials only
                    end
                else
                    nn = size(B_Nov.(trial_temp).Nov,3);
                    n  = size(B_Nov.(trial_temp).Nov,1)
                    if n == 6                                % for VBLASt n == 6, FLS n >=5
                        X(:,k) = B_Nov.(trial_temp).Nov(1:n,i,j);
                    end
                end
                t_count = t_count+1;
                %fnPlot(p1.(trial_temp)(:,1,j), trial_temp,j,Cnames,fpath2);
                %xlswrite(fullFileName,p1.(trial_temp)(:,1,j),sheet,col_name(col+2))
            end
            if m == 1
                [P_AcrossTrial_1.Exp(i,j),~,~] = anova1(X);    % pval -> trial x connection
                N = size(X,2);
                kkk = 1;
                for iii = 1:N % to store the statistics over each trial
                    P_AcrossTrial.(Win)(i,iii+kkk) =  nanmean(X(:,iii)); % Mean mean of first column i.e. connection ij
                    P_AcrossTrial.(Win)(i,iii+kkk+1) =  std(X(:,iii)); % std of first column i.e. connection ij
                    kkk = kkk+1;
                end
                [P_AcrossTrial.(Win)(i,2*kkk+2),tbl,~] = anova1(X);    % pval -> connection x windows
                P_AcrossTrial.(Win)(i,2*kkk) =  tbl{2,5}; % F-Statistics
                sheet = sprintf('Exp_Win%d',j);
            else
                [P_AcrossTrial_1.Nov(i,j),~,~] = anova1(X);
                N = size(X,2);
                kkk = 1;
                for iii = 1:N % to store the statistics over each trial
                    P_AcrossTrial.(Win)(i,iii+kkk) =  nanmean(X(:,iii)); % Mean mean of first column i.e. connection ij
                    P_AcrossTrial.(Win)(i,iii+kkk+1) =  std(X(:,iii)); % std of first column i.e. connection ij
                    kkk = kkk+1;
                end
                [P_AcrossTrial.(Win)(i,2*kkk+3),tbl,~] = anova1(X);    % pval -> connection x windows
                P_AcrossTrial.(Win)(i,2*kkk) =  tbl{2,5}; % F-Statistics
                sheet = sprintf('Nov_Win%d',j);
            end
        end
        pause;
        [FDR_temp,Q_temp] = mafdr(P_AcrossTrial.(Win)(:,2*kkk+2));  % FDR correction on p-value
        P_AcrossTrial.(Win)(:,2*kkk+1) = FDR_temp;
        P_AcrossTrial.(Win)(:,2*kkk+2) = Q_temp;
        fpath2 = 'D:\RPI\ResearchWork\Papers_\Effective_Connectivity\toolbox_original\mvgc_v1.0\Results_TRGC_Nov_Exp_Nature\P_value_corrected_FDR';
        fullFileName = fullfile(fpath2, baseFileName);
        %xlswrite(fullFileName,P_AcrossTrial.(Win),sheet)
    end
end
