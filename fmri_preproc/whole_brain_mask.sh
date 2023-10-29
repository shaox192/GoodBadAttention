#!/bin/tcsh -xef



set subj = GBAI01
set prefix = ../${subj}/${subj}_exp

# assign output directory name
set in_dir = ${prefix}/reg_data
set out_dir = ${prefix}/masks

if ( ! -d $out_dir ) then
    mkdir $out_dir
endif

# *** CHANGE THIS!!
set base_run = ${in_dir}/${subj}_r5_topup_st_vr

# ================================== mask ==================================
# create 'full_mask' dataset
# This uses out lab's pipeline, extracting automask from one of the runs.
3dAutomask  -prefix ${out_dir}/whole_brain_mask \
            -dilate 1 \
            -overwrite \
            ${base_run}+orig








