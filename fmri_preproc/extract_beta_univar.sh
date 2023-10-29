#!/bin/tcsh -xef

# extract beta values corresponding to eavh trial.


set roi = PPA

set sub = 1
foreach sub_n (`count -digit 2 1 $sub`)
       set subj = GBAI${sub_n}
       set prefix = ../${subj}/${subj}_exp

       set out_dir = ../${subj}/${subj}_beta
       if (! -d ${out_dir}) then
              mkdir ${out_dir}
       endif

       set mask_f = ${prefix}/masks/${subj}_${roi}_final
       foreach run (`count -digit 1 0 7`)
              set bucket_f = ${prefix}/regression_results_mvpa/run${run}/${subj}_bucket
              
              set out_dir_run = ${out_dir}/run${run}
              if (! -d $out_dir_run) then
                     mkdir $out_dir_run
              endif

              foreach blk (`count -digit 1 0 19`)
                     @ count = 3 * $blk + 1
                     echo 'Current used value, '"${count}"'!'   
                     3dmaskdump -noijk -xyz \
                                -mask ${mask_f}+orig \
                                -o ${out_dir_run}/${subj}_${roi}_blk$count.txt \
                                ${bucket_f}+orig'['"${count}"']'
              end
       end
end




