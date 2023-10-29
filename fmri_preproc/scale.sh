#!/bin/tcsh -xef


set num_subj = 1

foreach sub_n (`count -digit 2 1 ${num_subj} `)
    
    set subj = GBAI${sub_n}
    set prefix = ../${subj}/${subj}_exp

    set in_dir = ${prefix}/blur_data

    # assign output directory name
    set out_dir = ${prefix}/norm_data
    if ( ! -d $out_dir ) then
        mkdir $out_dir
    endif

    # ================================= scale ==================================
    # scale each voxel time series to have a mean of 100
    # (be sure no negatives creep in)
    # (subject to a range of [0,200])

    ###### use our lab's %signal change.

    foreach run (`count -digits  1 0 7`)
        
        3dTstat -prefix rm_scale_r${run}_mean \
                ${in_dir}/${subj}_r${run}_topup_st_vr_sm+orig

        3dcalc  -datum float \
                -a ${in_dir}/${subj}_r${run}_topup_st_vr_sm+orig \
                -b rm_scale_r${run}_mean+orig \
                -expr "((a-b)/b*100)" \
                -overwrite \
                -prefix ${out_dir}/${subj}_r${run}_topup_st_vr_sm_norm
    end
    
    rm rm*

end






