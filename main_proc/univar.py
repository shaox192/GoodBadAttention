import time

import utils.params as helper
import utils.plots as plots
from fMRIflow.postproc.mvpa.load_data import load_data_time_series, load_data_beta
from fMRIflow.postproc.mvpa.base_obj.utils import subselect_cond

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.stats.anova import AnovaRM
from scipy.stats import ttest_rel

from typing import List


def _plot_ts(dat, cond_order, label):
    for i in range(len(cond_order)):
        arr = np.asarray(dat[cond_order[i]])
        lb = label[i]
        shape = "--" if "distracted" in lb else '-'

        mean_ts = arr[:, 0]
        err_ts = arr[:, 1]

        x = [f"TR{i}" for i in range(9)]

        plt.errorbar(x=x, y=mean_ts, yerr=err_ts, color=helper.PLOT_COLOR_COND[i], linestyle=shape,
                     markersize=5, marker='o', label=lb)
    plt.legend(loc='upper left')
    plt.show()


def _plot_ts_multiple(dat: dict, cond_order: List[str], label: List[str]) -> None:
    x = np.arange(9).astype(str)
    fig, axes = plt.subplots(3, 5, figsize=(8, 6), dpi=1000)
    for i in range(len(cond_order)):
        arr = np.asarray(dat[cond_order[i]])
        lb = label[i]
        shape = "--" if "distracted" in lb else '-'

        for sub in range(arr.shape[0]):
            ts = arr[sub, :]
            axes[int(sub / 5), sub % 5].errorbar(x=x, y=ts, color=helper.PLOT_COLOR_COND[i], linestyle=shape,
                                                 markersize=5, marker='o', label=lb)
    for i in range(15):
        axes[int(i / 5), i % 5].set_title(f"GBAI{i + 1}" if i >= 9 else f"GBAI0{i + 1}")
        if i < 10:
            axes[int(i / 5), i % 5].set_xticklabels([])
    fig.tight_layout()
    # plt.savefig("test_univar_300.png", dpi=300)
    # plt.savefig("test_univar_600.png", dpi=600)
    plt.show()


def grp_plot_ts(roi: str, plot_avg: bool = True, plot_sep: bool = False):
    cond_name = ['_'.join([q, t]) for q in helper.IM_QUALITY for t in helper.EXP_TASK]

    def _load_all_sub(roi):
        for sub in sorted(helper.SUB_LAB + helper.SUB_OUT):
            lb_dir = f"{sub}/{sub}_exp_behav/perception_behav_output"
            bold_dir = f"{sub}/{sub}_norm"
            dat, lb = load_data_time_series(sub, roi, lb_dir, bold_dir)

            dat = np.asarray(dat)
            lb = np.asarray(lb)

            for cond in cond_name:
                mask = (lb[:, 1] == cond.split('_')[0]) * (lb[:, 0] == cond.split('_')[1])
                all_sub[cond].append(np.mean(dat[mask], axis=0))

    def _2meansem(dt: dict):
        mat = []
        for _, val in dt.items():
            mat.append(np.asarray([v for v in val]).T)
        mean_sem = helper.within_sub_error(np.vstack(mat))

        count = 0
        for key, _ in dt.items():
            dt[key] = mean_sem[count: count + 9, :]
            count += 9
        return dt

    all_sub = {cond: [] for cond in cond_name}
    _load_all_sub(roi)

    if plot_sep:
        _plot_ts_multiple(all_sub, helper.PLOT_ORDER_COND, helper.PLOT_LB_COND)
    if plot_avg:
        _plot_ts(_2meansem(all_sub), helper.PLOT_ORDER_COND, helper.PLOT_LB_COND)

    return all_sub


# ========================== group beta plots =====================================
def _beta_anova(df, within):
    aovrm2way = AnovaRM(df, 'data', 'sub_id', within=within)
    res2way = aovrm2way.fit()
    print(res2way)


def _beta_ttest(df):
    a = (df.query('quality == "good"')).query("task == 'scenes'")["data"]
    b = (df.query('quality == "bad"')).query("task == 'scenes'")["data"]
    print("******************* scenes **********************")
    print("good mean: ", np.mean(np.asarray(a)))
    print("bad mean: ", np.mean(np.asarray(b)))
    print("t test results for scenes: ")
    print(ttest_rel(a, b))

    a = (df.query('quality == "good"')).query("task == 'fixation'")["data"]
    b = (df.query('quality == "bad"')).query("task == 'fixation'")["data"]
    print("******************* fixation **********************")
    print("good mean: ", np.mean(np.asarray(a)))
    print("bad mean: ", np.mean(np.asarray(b)))
    print("t test results for fixation: ")
    print(ttest_rel(a, b))


def _2pd_dataframe(data_dict: dict, var_name: list, num_sub: int, roi: str):
    df_dict = {"data": [],
               "sub_id": np.tile(np.arange(0, num_sub), len(data_dict))
               }
    cond_dict = {var: [] for var in var_name}
    for key, val in data_dict.items():
        cond = key.split('_')
        for dat in val:
            df_dict["data"].append(dat)
            for i in range(len(var_name)):
                cond_dict[var_name[i]].append(cond[i])

    df_dict.update(cond_dict)
    df = pd.DataFrame(df_dict)
    # df.to_csv(f"grp_perm/univar_grp_{roi}_forR.csv")
    return df


def grp_plot_beta(roi):
    # beta value for bar plot
    cond_ls = ["_".join([q, t]) for t in helper.EXP_TASK for q in helper.IM_QUALITY]
    all_beta = {}
    for cond in cond_ls:
        beta_curr_cond = []
        roi_size = []
        for sub in sorted(helper.SUB_LAB + helper.SUB_OUT):
            d_arr = np.loadtxt(f"../GBAI2022_2/{sub}/{sub}_beta_grp/{sub}_{roi}_{cond}.txt", dtype=float)[:, -1]
            beta_curr_cond.append(np.mean(d_arr))
            roi_size.append(d_arr.shape[0])
        print(roi, np.mean(np.asarray(roi_size) * 0.008), np.std(np.asarray(roi_size) * 0.008))
        exit()
        all_beta[cond] = beta_curr_cond

    df = _2pd_dataframe(all_beta, ["quality", "task"], 15, roi)
    _beta_ttest(df)
    _beta_anova(df, within=["quality", "task"])

    full_data = np.vstack([np.asarray(val).reshape(1, -1) for _, val in all_beta.items()])
    data_mean_sem = helper.within_sub_error(full_data)

    beta4plot = {}
    beta4plot_individual = {}
    cnt = 0
    for key, _ in all_beta.items():
        print(key, data_mean_sem[cnt, :])
        beta4plot[key] = data_mean_sem[cnt, :]
        beta4plot_individual[key] = np.asarray(full_data[cnt])
        cnt += 1

    return beta4plot, beta4plot_individual


def _2pd_dataframe_overrun(data_dict: dict, num_sub: int, roi: str):
    var_name = ["quality", "task", "run"]
    num_run = 4
    df_dict = {"data": [],
               "sub_id": []
               }
    cond_dict = {var: [] for var in var_name}
    for key, val in data_dict.items():
        cond = key.split('_')
        for i in range(val.shape[0]):
            for j in range(val.shape[1]):
                cond_dict["quality"].append(cond[0])
                cond_dict["task"].append(cond[1])
                cond_dict["run"].append(f"run{j}")
                df_dict["sub_id"].append(i)
                df_dict["data"].append(val[i, j])
    df_dict.update(cond_dict)
    df = pd.DataFrame(df_dict)
    print(df)
    # df.to_csv(f"grp_perm/univar_byrun_{roi}_forR_{num_sub}.csv")
    return df


def grp_plot_beta_by_run(roi, fname):
    # num_runs = 4  # 4 for each scenes/fixation condition, but the order was random.
    sub_ls = sorted(helper.SUB_LAB + helper.SUB_OUT)
    # sub_ls.pop(sub_ls.index("GBAI08"))

    # beta value for bar plot
    all_sub = {'_'.join(k): np.empty((len(sub_ls), len(helper.KEY_LS))) for k in helper.KEY_LS}

    for sub_i in range(len(sub_ls)):
        sub = sub_ls[sub_i]
        tic = time.time()
        #  =========== perception data load =========
        lb_dir = f"{sub}/{sub}_exp_behav/perception_behav_output"
        beta_dir = f"../GBAI2022_2/{sub}/{sub}_beta"
        beta_ls = load_data_beta(sub, roi, lb_dir, beta_dir)
        #  ==========================================

        for k in range(len(helper.KEY_LS)):
            curr_cond_ls = subselect_cond(beta_ls, helper.ATTR_LS, helper.KEY_LS[k])
            curr_cond_ls.sort(key=lambda x: x.run)
            all_run = []
            prev_run = -1
            for obj in curr_cond_ls:
                if obj.run == prev_run:
                    all_run[-1].append(np.mean(obj.bold))
                else:
                    all_run.append([np.mean(obj.bold)])
                    prev_run = obj.run

            curr_cond_by_run = np.asarray([np.mean(arr) for arr in all_run])
            all_sub['_'.join(helper.KEY_LS[k])][sub_i] = curr_cond_by_run

        print(f"{sub} takes: {time.time() - tic}")

    # for saving the data in dataframe csv to run r stuff
    df = _2pd_dataframe_overrun(all_sub, len(sub_ls), roi)
    _beta_anova(df, ["quality", "task", "run"])
    mean_sem = helper.within_sub_error(np.hstack([v for _, v in all_sub.items()]).T)
    count = 0
    dt = {}
    for i in range(len(helper.KEY_LS)):
        dt['_'.join(helper.KEY_LS[i])] = mean_sem[count: count + 4, :]
        count += 4

    helper.pickle_dump(fname, dt)
    return dt
    # _plot_runs(dt, PLOT_ORDER_COND, PLOT_LB_COND)


def main_group_bar():
    dat_dict, dat_individual_dict = {}, {}
    for roi in ["RSC"]: # helper.ROI:
        print("\n======>>", roi)
        beta4plot, beta4plot_individual = grp_plot_beta(roi)

        dat_dict[roi] = beta4plot
        dat_individual_dict[roi] = beta4plot_individual
    # helper.plot_bar_roi(dat_dict, dat_individual_dict, helper.PLOT_COLOR_COND, helper.PLOT_LB_COND, helper.PLOT_ORDER_COND)
    plots.plot_bar_PPA(dat_dict, dat_individual_dict,
                       helper.PLOT_COLOR_COND, helper.PLOT_LB_PLOT, helper.PLOT_ORDER_COND,
                       clf=False, fname="univar_ppa_new.png")


def main_overrun():
    # plot the over run adaptation figure
    # only one for each ROI
    for roi in ["PPA"]:  # helper.ROI:
        fname = f"temp_univar_overrun_dt_{roi}.pkl"
        print("\n======>>", roi)
        try:
            dt4plot = helper.pickle_load(fname)
        except FileNotFoundError:
            dt4plot = grp_plot_beta_by_run(roi, fname)

        plots.plot_runs_univar_PPA(dt4plot,
                                   helper.PLOT_COLOR_COND, helper.PLOT_ORDER_COND, helper.PLOT_LB_COND,
                                   fname="univar_ppa_overrun.png")
        exit()


if __name__ == "__main__":
    os.chdir(helper.DATA_DIR)
    # main_overrun()
    main_group_bar()
