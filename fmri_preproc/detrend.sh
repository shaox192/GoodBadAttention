#!/bin/tcsh -xef

# only done for experimental runs. for functional localizers, this was replaced with a polynomial regressors in 3ddeconvolve (GLM)

set num_subj = 1

foreach sub_n (`count -digit 2 1 ${num_subj} `)
    
    set subj = GBAI${sub_n}
    set prefix = ../${subj}/${subj}_exp

    set in_dir = ${prefix}/norm_data

    # assign output directory name
    set out_dir = ${prefix}/detrended_data
    if ( ! -d $out_dir ) then
        mkdir $out_dir
    endif

    # ================================= scale ==================================
    # scale each voxel time series to have a mean of 100
    # (be sure no negatives creep in)
    # (subject to a range of [0,200])

    ###### use our lab's %signal change.

    foreach run (`count -digits  1 0 7`)
        3dDetrend -session $out_dir \
                  -prefix ${subj}_r${run}_topup_st_vr_sm_norm_detrend \
                  -polorts 3 \
                  ${in_dir}/${subj}_r${run}_topup_st_vr_sm_norm+orig
    
    end
    
end






