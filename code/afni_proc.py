#!/bin/tcsh
# =============================================================================
# MRIeat - AFNI preprocessing (afni_proc.py command)
# =============================================================================
# This is the afni_proc.py call I used to build the single-subject
# preprocessing + first-level regression pipeline. It is included here for
# reference only. All real subject IDs and local directories have been
# replaced with the generic placeholder /path/to/data, so this will NOT run
# as-is. To use it, point the paths at your own BIDS-style data and stimulus
# timing files, and loop it over your subjects.
#
# Task: block/event design, 6 runs, picture cue + juice outcome.
# Conditions modelled: ero, mut, neu, food (food+ / food-), juice (Y / N).
# =============================================================================

set subj    = sub-XX
set top_dir = /path/to/data
set anat    = $top_dir/$subj/anat/${subj}_T1w.nii.gz
set epi_dir = $top_dir/$subj/func
set stim_dir = /path/to/data/stim_times/$subj

afni_proc.py                                                            \
    -subj_id            $subj                                           \
    -script             proc.$subj  -scr_overwrite                      \
    -out_dir            /path/to/data/derivatives/$subj.results         \
    -dsets                                                              \
        $epi_dir/${subj}_task-mrieat_run-1_bold.nii.gz                  \
        $epi_dir/${subj}_task-mrieat_run-2_bold.nii.gz                  \
        $epi_dir/${subj}_task-mrieat_run-3_bold.nii.gz                  \
        $epi_dir/${subj}_task-mrieat_run-4_bold.nii.gz                  \
        $epi_dir/${subj}_task-mrieat_run-5_bold.nii.gz                  \
        $epi_dir/${subj}_task-mrieat_run-6_bold.nii.gz                  \
    -copy_anat          $anat                                           \
    -blocks             tshift align tlrc volreg blur mask scale regress \
    -tcat_remove_first_trs  2                                           \
    -align_opts_aea     -cost lpc+ZZ -giant_move -check_flip            \
    -tlrc_base          MNI152_2009_template_SSW.nii.gz                 \
    -tlrc_NL_warp                                                       \
    -volreg_align_to    MIN_OUTLIER                                     \
    -volreg_align_e2a                                                   \
    -volreg_tlrc_warp                                                   \
    -blur_size          4.0                                             \
    -mask_epi_anat      yes                                             \
    -regress_stim_times                                                 \
        $stim_dir/ero.1D                                                \
        $stim_dir/mut.1D                                                \
        $stim_dir/neu.1D                                                \
        $stim_dir/foodplus.1D                                           \
        $stim_dir/foodminus.1D                                          \
        $stim_dir/juiceY.1D                                             \
        $stim_dir/juiceN.1D                                             \
    -regress_stim_labels                                                \
        ero mut neu foodplus foodminus juiceY juiceN                    \
    -regress_basis      'BLOCK(2,1)'                                    \
    -regress_motion_per_run                                            \
    -regress_censor_motion  0.3                                         \
    -regress_censor_outliers 0.05                                       \
    -regress_apply_mot_types demean deriv                               \
    -regress_opts_3dD                                                   \
        -jobs 4                                                         \
    -regress_make_ideal_sum  sum_ideal.1D                               \
    -regress_est_blur_epits                                             \
    -regress_est_blur_errts                                             \
    -regress_run_clustsim    no                                         \
    -html_review_style       pythonic

# After this runs, execute the generated proc.$subj script:
#   tcsh -xef proc.$subj |& tee output.proc.$subj
