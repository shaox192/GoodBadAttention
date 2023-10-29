
## Use of Topup for distortion correction

Distortion correction was done using topup program from FSL. Conversion between different data formats 
(BRIK/HEAD <-> NIFTI) was therefore necessary. We followed the guide 
[here](https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/topup/TopupUsersGuide).

Our acquisition parameters in *acqparams.txt*: 

```angular2html
0 1 0 0.05656  
0 1 0 0.05656 
0 1 0 0.05656
0 -1 0 0.05656
0 -1 0 0.05656
0 -1 0 0.05656
```

topup and apply: 
```angular2html
fslmerge  -t my_b0_images AP.nii PA.nii

topup   --imain=my_b0_images \
        --datain=acqparams.txt \
        --config=b02b0.cnf \
        --out=my_topup_results \
        --fout=my_field \
        --iout=my_unwarped_images

applytopup  --imain=sub1_r1.nii \
            --topup=my_topup_results \
            --datain=acqparams.txt \
            --inindex=1  \
            --out=sub1_r1_topup \
            --method=jac


```
