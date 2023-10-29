
import os
POLORT = -1
import subprocess

def get_regressors(mp_f, regressor_dir, run):

    # -stim_file 1 ${regressor_dir}/face_final.1D -stim_label 1 face_final
    cnt = 1
    stim_file_ls = []
    for f in sorted(os.listdir(regressor_dir)):

        if (not f.startswith(f"run{run}")) or (not f.endswith(".1D")) or (f.startswith('.')):
            continue

        stim_file_ls+=["-stim_file", str(cnt), f"{regressor_dir}/{f}", 
                       "-stim_label", str(cnt), f.split('.')[0]]
        cnt += 1

    # -stim_file 5 ${mp_file}'[1]' -stim_label 5 mc_params1 -stim_base 5 \
    motion_ls = []
    for i in range(6):
        motion_ls += ["-stim_file", str(cnt), f"{mp_f}'[{i + 1}]'",
                      "-stim_label", str(cnt), f"mc_params{i + 1}",
                      "-stim_base", str(cnt)]
        cnt += 1
    return cnt - 1, stim_file_ls + motion_ls


def deconvolve(sub, run_num, data_fs, mask, censor_f, mp_f, regressor_dir, results_dir):
    cmd_ls = ["3dDeconvolve", 
              "-input", data_fs,
              "-mask", mask,
              "-censor", censor_f,
              "-CENSORTR", "'*:0-2'",
              "-CENSORTR", "'*:238'",  \
              "-polort", str(POLORT)]

    num_stim, regressors = get_regressors(mp_f, regressor_dir, run_num)
    cmd_ls += ["-num_stimts", str(num_stim)]
    cmd_ls += regressors
    cmd_ls += ["-bucket", f"{results_dir}/{sub}_bucket", "-fout", "-tout",
               "-xjpeg", f"{results_dir}/{sub}_regressor.jpg"]

    return cmd_ls


def dry_run_cmd(cmd):
    for ln in cmd:
        if ln.startswith("-") and ("stim_label" not in ln) and ("stim_base" not in ln):
            print(f'\n{ln}', end=' ')
        else:
            print(ln, end=' ')

def main(dry_run):

    for i in range(1, 2):
        for r in range(8):
            sub_str = f"GBAI0{i}" if i < 10 else f"GBAI{i}"
            prefix = f"../{sub_str}/{sub_str}_exp"

            data_fs = f"{prefix}/detrended_data/{sub_str}_r{r}_topup_st_vr_sm_norm_detrend+orig.HEAD"
            mask = f"{prefix}/masks/whole_brain_mask+orig"
            censor_f = f"{prefix}/outcounts/out_censor_r{r}.1D"
            mp_f = f"{prefix}/params/{sub_str}_r{r}_vr_mp.1D"

            regressor_dir = f"{prefix}/regressor_final_mvpa"
            results_dir = f"{prefix}/regression_results_mvpa/run{r}"
            if not os.path.exists(results_dir):
                print(f"Making results directory: {results_dir}")
                os.makedirs(results_dir)

            cmd = deconvolve(sub_str, r, data_fs, mask, censor_f, mp_f, regressor_dir, results_dir)
            if dry_run:
                dry_run_cmd(cmd)
            else:
                subprocess.call(' '.join(cmd), shell=True)

    
if __name__ == "__main__":
    dry_run = False
    main(dry_run)



