# GoodBadAttention

Code base for [*Is Attention Necessary for the Representational Advantage of Good Exemplars over Bad Exemplars?*](https://doi.org/10.1111/ejn.16291)
(Shao & Beck (2024), European Journal of Neuroscience, 59(9), 2129-2415.)

### Absract


### Publicly available data:
- BOLD beta estimates from each ROI (PPA, OPA, MPA) are available on [OSF](https://osf.io/yc97n/). 
There are outputs after finishing the preprocessing pipeline. Data are stored in __HDF5__ formats.
Behavioral files not only contain subjects' responses and reactions times, but also has the order and timing
of images presented. 
- Images used in the experiment and their representativeness scores are available [here](). 
These ratings scores and images were obtained from [Torralbo et al., 2013](https://doi.org/10.1371/journal.pone.0058594). 

- Image rating experiment pre-registration details are also available [here](https://osf.io/4mqpj/).

***
### fMRI data preprocessing scripts
All data preprocessing scripts are in [fmri_preproc/](./fmri_preproc). We show all preprocessing scripts used to process
our fMRI data. Details are in our [*Methods*]() section. These scripts are written in tcsh, and needs manual adjustments, 
and visual inspections of processing results regularly, therefore are __not directly runnable__ but serve as a demonstration
of our procedures.

Similar pipeline was applied to both functional localizer for ROI extractions and main experimental runs for actual data
with some differences. Below we also outline which steps we took for each type.

**Code dependencies**:
 1. FSL: FLIRT version 6.0
 2. AFNI: Version AFNI_20.0.4

**Notes**
1. GLMs are in 3ddeconvolve* files. The one with .py extension was used to avoid manually input all trial regressors
because we need one estimate for each trial for later MVPA analysis.
2. ROI masks were manully isolated using te bucket files output after 3ddeconvolve on functional localizers. These masks
do need to be intersected with whole brain mask, then warped to the experimental run space using T1 images because these
two sessions take place on different days.


***
### Main data processing scripts

Subsequent data processing that produced all of our results are in [main_proc/](./main_proc). We included 
multiple analysis to understand the role of attention in the neural effects of statistical regularities in scene images.
Below is a breakdown of code files used for each analysis.

**> Prerequisite data files**: Please download both neural and behavioral data files from [here](https://osf.io/yc97n/). 
and put them respectively into *main_proc/neural_data/* and *main_proc/behav_data/*.

**> Code dependencies**: Use [requirements.txt](./requirements.txt) to install all dependencies.




