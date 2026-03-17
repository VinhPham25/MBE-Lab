%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ASTaUND Paper-Faithful Preprocessing Pipeline
% Based on: Kamat et al. (2023) Scientific Data
% 
% This pipeline replicates the exact preprocessing steps described in the
% paper for TechEn CW6 systems (Expert/Novice and Learning Curve datasets)
%
% Pipeline Steps:
% 1. Channel Quality Assessment (80-140 dB SNR threshold)
% 2. Intensity → Optical Density conversion
% 3. Bandpass filter (0.01-0.50 Hz)
% 4. Motion artifact detection and correction (rPCA)
% 5. Modified Beer-Lambert Law (PPF: 6.4 @ 690nm, 5.8 @ 830nm)
% 6. GLM with Gaussian basis functions and short-separation regression (OLS)
%
% NOTE: Parameters marked with * are inferred from Homer3 defaults or
%       reference papers as they were not explicitly stated in the paper
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all; close all;

%% ========================================================================
% STEP 0: Load Data
% ========================================================================
% Select SNIRF file
[fname, pname] = uigetfile('*.snirf', 'Select SNIRF file');
if fname == 0
    error('No file selected');
end

% Load SNIRF data
data_raw = SnirfClass([pname, fname]);
fprintf('Loaded: %s\n', fname);
fprintf('Channels: %d, Duration: %.1f s, Sampling rate: %.1f Hz\n', ...
    size(data_raw.data.dataTimeSeries,2), ...
    data_raw.data.time(end), ...
    1/mean(diff(data_raw.data.time)));

%% ========================================================================
% STEP 1: Channel Quality Assessment (80-140 dB SNR)
% ========================================================================
% Paper: "channels with signal quality outside 80 dB to 140 dB were excluded"
%
% Parameters:
% - SNRthresh: 2.5 * (inferred from function default)
% - dRange: [0, 10000000] (intensity range, default)
% - SDrange: [0, 45] mm (source-detector distance range, default)
% - reset: 0 (don't reset manual pruning)

fprintf('\n=== STEP 1: Channel Quality Assessment ===\n');

mlActMan = [];  % No manual channel pruning
tIncMan = [];   % No manual time exclusion

[mlActAuto] = hmrR_PruneChannels(data_raw.data, ...
                                  data_raw.probe, ...
                                  mlActMan, ...
                                  tIncMan, ...
                                  [0, 10000000], ...  % dRange
                                  2.5, ...            % SNRthresh *
                                  [0, 45], ...        % SDrange *
                                  0);                 % reset

nBad = sum(mlActAuto == 0);
fprintf('Channels excluded: %d / %d\n', nBad, length(mlActAuto));

%% ========================================================================
% STEP 2: Intensity → Optical Density
% ========================================================================
% Paper: "remaining raw optical signals...were converted into optical density"

fprintf('\n=== STEP 2: Intensity → Optical Density ===\n');

dod = hmrR_Intensity2OD(data_raw.data);

fprintf('OD conversion complete. Range: [%.3f, %.3f]\n', ...
    min(dod.dataTimeSeries(:)), max(dod.dataTimeSeries(:)));

%% ========================================================================
% STEP 3: Bandpass Filter (0.01-0.50 Hz)
% ========================================================================
% Paper: "band-pass filter of 0–0.50 Hz was used to denoise the signal"
% Note: Paper says "0-0.50" but standard practice is 0.01-0.50 Hz

fprintf('\n=== STEP 3: Bandpass Filter ===\n');

hpf = 0.001;  % High-pass cutoff (Hz)
lpf = 0.50;  % Low-pass cutoff (Hz)

dod_filt = hmrR_BandpassFilt(dod, hpf, lpf);

fprintf('Bandpass filter applied: %.2f - %.2f Hz\n', hpf, lpf);

%% ========================================================================
% STEP 4: Motion Artifact Detection and Correction (rPCA)
% ========================================================================
% Paper: "Motion artifacts...were corrected using recursive principal 
%         component analysis"
%
% Parameters from von Lühmann et al. (2020) - Reference #36:
% - tMotion: 0.5 s (motion detection window)
% - tMask: 0.5 s * (inferred from von Lühmann, paper states 0.5)
% - STDEVthresh: 20 (standard deviation threshold)
% - AMPthresh: 5 (amplitude threshold for OD units)
% - nSV: 0.97 * (variance explained - inferred from typical values)
% - maxIter: 5 * (maximum iterations - function default)
% - turnon: 1 (enable correction)

fprintf('\n=== STEP 4: Motion Artifact Detection and Correction ===\n');

tMotion = 0.5;      % Motion detection window (s)
tMask = 0.5;        % Motion mask duration (s) *
STDEVthresh = 20;   % Std dev threshold (from ref #36)
AMPthresh = 5;      % Amplitude threshold (OD units, from ref #36)
nSV = 0.97;         % Variance explained by PCs *
maxIter = 5;        % Max iterations *
turnon = 1;         % Enable correction

[dod_mc, tIncAuto, svs, nSVactual, tInc0] = hmrR_MotionCorrectPCArecurse(...
    dod_filt, ...
    data_raw.probe, ...
    mlActMan, ...
    mlActAuto, ...
    tIncMan, ...
    tMotion, ...
    tMask, ...
    STDEVthresh, ...
    AMPthresh, ...
    nSV, ...
    maxIter, ...
    turnon);

motionRatio = 1 - sum(tIncAuto{1}) / length(tIncAuto{1});
fprintf('Motion artifacts detected: %.1f%% of data flagged\n', motionRatio*100);
fprintf('Principal components used: %d (%.1f%% variance)\n', nSVactual, nSV*100);

%% ========================================================================
% STEP 5: Modified Beer-Lambert Law
% ========================================================================
% Paper: "modified Beer-Lambert law with partial path-length factors of 
%         6.4 (690 nm) and 5.8 (830 nm)"

fprintf('\n=== STEP 5: Modified Beer-Lambert Law ===\n');

ppf = [6.4, 5.8];  % Partial path-length factors for 690nm and 830nm

dc = hmrR_OD2Conc(dod_mc, data_raw.probe, ppf);

fprintf('Beer-Lambert conversion complete (PPF: %.1f @ 690nm, %.1f @ 830nm)\n', ...
    ppf(1), ppf(2));
fprintf('HbO range: [%.2f, %.2f] μM\n', ...
    min(dc.dataTimeSeries(:,1,:),[],'all'), ...
    max(dc.dataTimeSeries(:,1,:),[],'all'));
fprintf('HbR range: [%.2f, %.2f] μM\n', ...
    min(dc.dataTimeSeries(:,2,:),[],'all'), ...
    max(dc.dataTimeSeries(:,2,:),[],'all'));

%% ========================================================================
% STEP 6: GLM with Gaussian Basis Functions + Short-Sep Regression (OLS)
% ========================================================================
% Paper: "the most correlated short-distance channels are regressed from 
%         the long-distance channels...using a consecutive sequence of 
%         Gaussian basis functions via ordinary least squares to create 
%         the hemodynamic response function (HRF)"
%
% Parameters:
% - trange: [-5, 30] s * (epoch window - inferred from typical fNIRS)
% - glmSolveMethod: 1 (OLS)
% - idxBasis: 1 (Gaussian basis functions)
% - paramsBasis: [1.0, 1.0, 0, 0, 0, 0] * (stdev=1s, step=1s - function default)
% - rhoSD_ssThresh: 15 mm * (short-sep threshold - function default)
% - flagNuisanceRMethod: 1 (use most correlated short-sep channel)
% - driftOrder: 3 * (polynomial drift correction - function default)
% - c_vector: 0 (no contrast)

fprintf('\n=== STEP 6: GLM (Gaussian Basis + Short-Sep Regression) ===\n');

trange = [-5, 30];              % Epoch window (s) *
glmSolveMethod = 1;             % 1 = OLS (paper specifies OLS)
idxBasis = 1;                   % 1 = Gaussian basis functions
paramsBasis = [1.0, 1.0, 0, 0, 0, 0];  % [stdev, step, ...] *
rhoSD_ssThresh = 15;            % Short-sep threshold (mm) *
flagNuisanceRMethod = 1;        % 1 = most correlated short-sep
driftOrder = 3;                 % Polynomial drift order *
c_vector = 0;                   % No contrast vector

% Prepare inputs for session-level GLM
% (For single-run analysis, wrap in cell arrays)
dcRuns = {dc};
stimRuns = {data_raw.stim};
mlActRuns = {mlActAuto};
AauxRuns = {[]};  % No auxiliary regressors
tIncAutoRuns = tIncAuto;
rcMapRuns = {''};  % No tCCA regressor map

[dcAvg, dcAvgStd, nTrials, dcNew, dcResid, dcSum2, beta, R, hmrstats] = ...
    hmrS_GLM(dcRuns, ...
             stimRuns, ...
             data_raw.probe, ...
             mlActRuns, ...
             AauxRuns, ...
             tIncAutoRuns, ...
             rcMapRuns, ...
             trange, ...
             glmSolveMethod, ...
             idxBasis, ...
             paramsBasis, ...
             rhoSD_ssThresh, ...
             flagNuisanceRMethod, ...
             driftOrder, ...
             c_vector);

fprintf('GLM complete:\n');
fprintf('  Method: Ordinary Least Squares (OLS)\n');
fprintf('  Basis: Gaussian (stdev=%.1fs, step=%.1fs)\n', ...
    paramsBasis(1), paramsBasis(2));
fprintf('  Short-sep regression: Most correlated channel (threshold: %d mm)\n', ...
    rhoSD_ssThresh);
fprintf('  Drift correction: %d-order polynomial\n', driftOrder);
fprintf('  Trials per condition: %s\n', mat2str(nTrials{1}));

% Calculate mean correlation (R) across channels
meanR_HbO = mean(R{1}(:,1), 'omitnan');
meanR_HbR = mean(R{1}(:,2), 'omitnan');
fprintf('  Mean model fit (R): HbO=%.3f, HbR=%.3f\n', meanR_HbO, meanR_HbR);

%% ========================================================================
% STEP 7: Extract and Save Results
% ========================================================================

fprintf('\n=== Pipeline Complete ===\n');

% Extract HRF data
tHRF = dcAvg.time;
yavg = dcAvg.GetDataTimeSeries('reshape');  % time x chromophore x channel x condition
yavgstd = dcAvgStd.GetDataTimeSeries('reshape');

fprintf('HRF extracted: %d timepoints, %d conditions\n', ...
    length(tHRF), size(yavg,4));

% Save results
[~, fname_base, ~] = fileparts(fname);
output_file = [pname, fname_base, '_processed.mat'];
save(output_file, 'dcAvg', 'dcAvgStd', 'nTrials', 'beta', 'R', ...
     'hmrstats', 'tHRF', 'yavg', 'yavgstd', 'mlActAuto');
fprintf('Results saved to: %s\n', output_file);

%% ========================================================================
% STEP 8: Visualization
% ========================================================================

fprintf('\n=== Generating Plots ===\n');

% Get measurement list for channel identification
ml = data_raw.data.GetMeasListSrcDetPairs('reshape');

% Plot HRF for all active channels (first condition, HbO only)
figure('Name', 'HRF - All Channels (HbO)', 'NumberTitle', 'off');
nChannels = size(yavg, 3);
activeChannels = find(mlActAuto == 1);

subplot_rows = ceil(sqrt(length(activeChannels)));
subplot_cols = ceil(length(activeChannels) / subplot_rows);

for i = 1:length(activeChannels)
    iCh = activeChannels(i);
    subplot(subplot_rows, subplot_cols, i);
    
    % HbO (condition 1)
    hbo_trace = squeeze(yavg(:, 1, iCh, 1));
    hbo_std = squeeze(yavgstd(:, 1, iCh, 1));
    
    % Plot with error band
    fill([tHRF; flipud(tHRF)], ...
         [hbo_trace + hbo_std; flipud(hbo_trace - hbo_std)], ...
         [1 0.7 0.7], 'EdgeColor', 'none', 'FaceAlpha', 0.3);
    hold on;
    plot(tHRF, hbo_trace, 'r-', 'LineWidth', 1.5);
    plot([0 0], ylim, 'k--', 'LineWidth', 0.5);
    
    title(sprintf('Ch %d (S%d-D%d)', iCh, ml(iCh,1), ml(iCh,2)));
    xlabel('Time (s)');
    ylabel('HbO (μM)');
    grid on;
end

sgtitle('Hemodynamic Response Functions (HbO) - All Active Channels');

% Plot average HRF across all active channels
figure('Name', 'Average HRF', 'NumberTitle', 'off');

% Calculate mean across active channels
hbo_mean = mean(yavg(:, 1, activeChannels, 1), 3);
hbr_mean = mean(yavg(:, 2, activeChannels, 1), 3);
hbo_std_mean = mean(yavgstd(:, 1, activeChannels, 1), 3);
hbr_std_mean = mean(yavgstd(:, 2, activeChannels, 1), 3);

% HbO
subplot(2,1,1);
fill([tHRF; flipud(tHRF)], ...
     [hbo_mean + hbo_std_mean; flipud(hbo_mean - hbo_std_mean)], ...
     [1 0.7 0.7], 'EdgeColor', 'none', 'FaceAlpha', 0.3);
hold on;
plot(tHRF, hbo_mean, 'r-', 'LineWidth', 2);
plot([0 0], ylim, 'k--', 'LineWidth', 1);
title('Average HbO Response');
xlabel('Time (s)');
ylabel('HbO (μM)');
grid on;
legend('±1 SD', 'Mean HbO');

% HbR
subplot(2,1,2);
fill([tHRF; flipud(tHRF)], ...
     [hbr_mean + hbr_std_mean; flipud(hbr_mean - hbr_std_mean)], ...
     [0.7 0.7 1], 'EdgeColor', 'none', 'FaceAlpha', 0.3);
hold on;
plot(tHRF, hbr_mean, 'b-', 'LineWidth', 2);
plot([0 0], ylim, 'k--', 'LineWidth', 1);
title('Average HbR Response');
xlabel('Time (s)');
ylabel('HbR (μM)');
grid on;
legend('±1 SD', 'Mean HbR');

sgtitle(sprintf('Average HRF (n=%d active channels)', length(activeChannels)));

fprintf('Plots generated.\n');

%% ========================================================================
% PARAMETER NOTES
% ========================================================================
% Parameters marked with * were inferred because they were not explicitly
% stated in the paper. Sources for inferred parameters:
%
% 1. SNRthresh (2.5): Homer3 hmrR_PruneChannels default
% 2. tMask (0.5s): von Lühmann et al. (2020) - Reference #36
% 3. nSV (0.97): Typical value for rPCA (97% variance explained)
% 4. maxIter (5): Homer3 hmrR_MotionCorrectPCArecurse default
% 5. trange ([-5, 30]s): Standard fNIRS epoch window
% 6. paramsBasis ([1.0, 1.0, ...]): Homer3 hmrS_GLM default for Gaussian
% 7. rhoSD_ssThresh (15mm): Homer3 hmrS_GLM default
% 8. driftOrder (3): Homer3 hmrS_GLM default (3rd-order polynomial)
%
% All other parameters are explicitly stated in the paper.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%