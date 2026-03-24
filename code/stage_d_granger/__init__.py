"""
Stage D: Granger Causality Analysis
====================================
Frequency-domain pairwise conditional Granger Causality pipeline for the
ASTaUND Expert/Novice fNIRS dataset.

Reference MATLAB script : matlab_references/Linear GC/GC_freq_main_TRGC_NovExp.m
Input                   : outputs/stage_b_processed/processed_data/expert-novice/
                          sub-{id}_task-ExpNovPatternCutting_desc-preprocessed_nirs_raw.fif
Output root             : outputs/stage_d_granger/linear_gc/expert-novice/

Modules
-------
utils_gc              — shared loaders, subject-group lookup, connection labels
01_extract_roi_signals — Data_Reg (5×T) extraction from preprocessed .fif
02_trial_segmentation  — trial onset/offset sample indices from MNE annotations
03_linear_gc           — VAR fitting, spectral GC, sliding-window loop
04_subject_gc_pipeline — per-subject orchestrator + group array builder
05_gc_statistics       — normality, H0/H1/HIV ANOVA + FDR correction
06_gc_visualization    — connectivity matrices, group comparison plots
07_run_linear_gc_pipeline — CLI batch runner

See docs/LINEAR_GC_PROGRESS.md for full task breakdown and design notes.
"""
