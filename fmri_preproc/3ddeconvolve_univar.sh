#!/bin/tcsh -xef

# 2nd step of regression: 3dDeconvolve to output the bucket files.
# 3dDeconvolve cannot write the 1D files if the folder is NOT creaed ahead of time.

set num_subj = 1

foreach sub_n (`count -digit 2 1 ${num_subj}`)
    set subj = GBAI${sub_n}
    set prefix = ../${subj}/${subj}_exp

    set regressor_dir = ${prefix}/regressor_final_uni
    set mp_file = ${prefix}/params/${subj}_all_run_vr_mp.1D
    set censor_file = ${prefix}/outcounts/censor_all_run.1D
    set mask_dir = ${prefix}/masks
    set in_dir = ${prefix}/norm_data

    set out_dir = ${prefix}/regression_results_uni
    if (! -d ${out_dir}) then
        mkdir ${out_dir}
    endif

    ##================ run the regression analysis ===================
    ## -polorts: 1 order per 150 secs is the default
    3dDeconvolve    -input ${in_dir}/${subj}_r*_topup_st_vr_sm_norm+orig.HEAD \
                    -mask ${mask_dir}/whole_brain_mask+orig \
                    -censor ${censor_file} \
                    -CENSORTR '*:0-2' \
                    -CENSORTR '*:238' \
                    -polort -1 \
                    -num_stimts 10 \
                    -stim_file 1 ${regressor_dir}/bad_fixation_final.1D -stim_label 1 bad_fixation_final \
                    -stim_file 2 ${regressor_dir}/bad_scenes_final.1D  -stim_label 2 bad_scenes_final \
                    -stim_file 3 ${regressor_dir}/good_fixation_final.1D -stim_label 3 good_fixation_final \
                    -stim_file 4 ${regressor_dir}/good_scenes_final.1D  -stim_label 4 good_scenes_final \
                    -stim_file 5 ${mp_file}'[1]' -stim_label 5 mc_params1 -stim_base 5 \
                    -stim_file 6 ${mp_file}'[2]' -stim_label 6 mc_params2 -stim_base 6 \
                    -stim_file 7 ${mp_file}'[3]' -stim_label 7 mc_params3 -stim_base 7 \
                    -stim_file 8 ${mp_file}'[4]' -stim_label 8 mc_params4 -stim_base 8 \
                    -stim_file 9 ${mp_file}'[5]' -stim_label 9 mc_params5 -stim_base 9 \
                    -stim_file 10 ${mp_file}'[6]' -stim_label 10 mc_params6 -stim_base 10 \
                    -num_glt 2 \
                    -gltsym 'SYM: 0.5*bad_scenes_final 0.5*good_scenes_final -0.5*bad_fixation_final -0.5*good_fixation_final' \
                    -glt_label 1 attention_scenes-fix \
                    -gltsym 'SYM: 0.5*bad_scenes_final 0.5*bad_fixation_final -0.5*good_scenes_final -0.5*good_fixation_final' \
                    -glt_label 2 SR_bad-good \
                    -bucket ${out_dir}/${subj}_bucket \
                    -fout \
                    -tout \
                    -xjpeg ${out_dir}/${subj}_regressor.jpg


end



