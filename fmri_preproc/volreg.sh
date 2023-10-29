#!/bin/tcsh -xef

# ********
# CHANGE: base run, base volume for align and 3dvolreg!!
# ********

set subj = GBAI01
set prefix = ../${subj}/${subj}_exp

set in_dir = ${prefix}/st_data

set out_data_dir = ${prefix}/reg_data
set out_transform_mat_dir = ${prefix}/volreg_transform_mat
set out_param_dir = ${prefix}/params
foreach out_dir ($out_data_dir $out_transform_mat_dir $out_param_dir)
    if (! -d ${out_dir}) then
            mkdir $out_dir
            echo making $out_dir
        endif
end

set base_run = ${in_dir}/${subj}_r5_topup_st


# ================================= volreg =================================
# align each dset to base volume, to anat

# register and warp
foreach run (`count -digits 1 0 7`)
    
    #### register each volume to the base image
    # input: original run files;
    # output 0: run files that are volume registered to the EPI base.
    # output 1: dfile: motion parameters for later regression
    # output 2: 1Dmatrix_save: transformation matrix for the warping
    3dvolreg    -zpad 4 \
                -prefix ${out_data_dir}/${subj}_r${run}_topup_st_vr \
                -dfile ${out_param_dir}/${subj}_r${run}_vr_mp.1D \
                -base ${base_run}+orig'[145]' \
                -1Dmatrix_save ${out_transform_mat_dir}/${subj}_r${run}_volreg_mat.1D   \
                -verbose \
                ${in_dir}/${subj}_r${run}_topup_st+orig

end

# #######dfile is the mc param file
# make a single file of registration params
cat ${out_param_dir}/${subj}_r*_vr_mp.1D > ${out_param_dir}/${subj}_all_run_vr_mp.1D


