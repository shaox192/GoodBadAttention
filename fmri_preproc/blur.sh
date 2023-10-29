#!/bin/tcsh -xef

# 1blur_fwhm kernel size is set to 3 for functional localizers

set num_subj = 1

foreach sub_n (`count -digit 2 1 ${num_subj} `)

    set subj = GBAI${sub_n}
    set prefix = ../${subj}/${subj}_exp
    set mask_f = ${prefix}/masks/whole_brain_mask

    set in_dir = ${prefix}/reg_data
    set out_dir = ${prefix}/blur_data
    if (! -d ${out_dir}) then
        mkdir ${out_dir}
    endif

    foreach run (`count -digit 1 0 7`)
        3dmerge -doall \
                -1blur_fwhm 2 \
                -1fmask ${mask_f}+orig \
                -prefix ${out_dir}/${subj}_r${run}_topup_st_vr_sm \
                ${in_dir}/${subj}_r${run}_topup_st_vr+orig
                 
    end

end




