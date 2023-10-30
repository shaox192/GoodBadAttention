import sys
import os
import numpy as np
from fMRIflow.postproc.mvpa.load_data import *
from fMRIflow.postproc.mvpa import CategoryBoundaryEffect
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel
from statsmodels.stats.anova import AnovaRM

import utils.params as helper
import utils.plots as plots
font = {'family': 'Times New Roman', 'size': 26}
plt.rc('font', **font)


def main_cbe():
    for roi in helper.ROI:
        all_sub = []
        pulling_apart = []
        for sub in helper.SUB_OUT+helper.SUB_LAB:
            print("cbe: ", roi, sub)
            #  =========== perception data load =========
            beta_dir = f"../GBAI2022_2/{sub}/{sub}_beta"
            lb_dir = f"{sub}/{sub}_exp_behav/perception_behav_output"
            data_ls = load_data_beta(sub, roi, lb_dir, beta_dir)

            cbe = CategoryBoundaryEffect(data_ls)
            pulling_apart.append(list(cbe.calculate_category_boundary_effect()))
            all_sub.append(cbe.category_boundary_effect)

        save_dir = f"../GBAI2022_2/grp_cbe"
        helper.check_dir(save_dir)
        save_mat = os.path.join(save_dir, f"cbe_{roi}_euclidean_all_cond") # original data
        np.save(save_mat, np.asarray(all_sub))
        save_mat = os.path.join(save_dir, f"cbe_{roi}_euclidean_all_cond_apart") # pulling apart data
        np.save(save_mat, np.asarray(pulling_apart))


# =========================== Group =============================


key_ls = []
for q in ["good", "bad"]:
    for t in ["scenes", "fixation"]:
        for c in ["cities", "mountains"]:
            key_ls.append('\n'.join([q, t, c]))

key_ls2 = ["X".join(i) for i in helper.KEY_LS]


def cbe_plot_sep(cbe_all_sub, err):
    df = pd.DataFrame(cbe_all_sub.flatten(order='F'), columns=["CBE_scores"])
    df["full_conditions"] = np.repeat([0, 4, 2, 6, 1, 5, 3, 7], 15) # this was in gsc 0, gsm 4, gfc 2, gfm 6, bsc 1, bsm 5, bfc 3, bfm 7 order,
    # needs to be in the gsc 0 bsc 1 gfc 2 bfc 3 gsm 4 bsm 5 gfm 6 bfm 7
    df["cond"] = np.repeat(key_ls2, 30)
    df["category"] = np.tile(np.repeat(["cities", "mountains"], 15), 4)
    sns.set_style('darkgrid')
    sns.set(rc={'figure.figsize': (8, 6)})

    sns.barplot(x='full_conditions', y='CBE_scores', data=df,
                palette=["blue", "red", "steelblue", "salmon", "blue", "red", "steelblue", "salmon"],
                ci=None, yerr=err, capsize=0.03, errwidth=1)
    plt.ylim(0.9, 0.95)
    plt.show()


def cbe_plot_comb(data_dict, plot_order):

    x_pos = np.arange(4)
    fig, ax = plt.subplots()
    ax.bar(x_pos, [data_dict[i][0] for i in plot_order], yerr=[data_dict[i][1] for i in plot_order], align='center',
           color=helper.PLOT_COLOR_COND, alpha=0.6, width=0.7,
           ecolor='black', capsize=5)

    ax.set_ylabel('CBE scores')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(helper.PLOT_LB_COND)
    ax.set_xlabel("Conditions")
    ax.yaxis.grid(True)
    plt.ylim(0, 1.0)
    plt.tight_layout()
    plt.savefig("test_cbe_600.png", dpi=600)
    plt.show()


def plot_bar_roi(dat, dat_individual, plot_order, bar_width):
    data_grp = []
    sem = []
    data_indiv = []
    for lb in plot_order:
        data_grp.append(dat[lb][0])
        sem.append(dat[lb][1])
        data_indiv.append(dat_individual[lb])

    bar_width = bar_width + 0.1
    x_pos = np.arange(1, 1 + len(helper.PLOT_DICT) * bar_width, bar_width)
    return data_grp, sem, data_indiv, x_pos


def plot_bar(dat: dict, dat_individual: dict, color_custom, label_custom, plot_order, fname=""):
    randgen = np.random.default_rng(1024)
    bar_width = 0.28

    data_grp_full, sem_full, data_indiv_full, x_pos_full = [], [], [], []
    x_tick_middle = []
    for i, k in enumerate(dat):
        data_grp, sem, data_indiv, x_pos = plot_bar_roi(dat[k], dat_individual[k], plot_order, bar_width=bar_width)
        data_grp_full.append(data_grp)
        sem_full.append(sem)
        data_indiv_full.append(data_indiv)
        x_pos_full.append(x_pos + i * 0.5 * len(PLOT_DICT))
        x_tick_middle.append(np.mean(x_pos_full[-1]))

    data_grp_full = np.concatenate(data_grp_full)
    sem_full = np.concatenate(sem_full)
    data_indiv_full = np.concatenate(data_indiv_full)
    x_pos_full = np.concatenate(x_pos_full) + 0.5
    x_tick_middle = np.asarray(x_tick_middle) + 0.5

    color_ = np.tile(color_custom, len(dat))
    x_labels = np.tile(label_custom, len(dat))

    fig, ax = plt.subplots(figsize=(14, 9.3))
    ax.bar(x_pos_full, data_grp_full, yerr=sem_full, align='center',
           color=color_, alpha=0.6, width=bar_width, linewidth=1.5,
           edgecolor='black', capsize=10)

    # plot scattered dots
    for i in range(x_pos_full.shape[0]):
        x_pos_jittered = np.repeat(x_pos_full[i], len(data_indiv_full[i])) \
                         + randgen.uniform(-0.12, 0.12, len(data_indiv_full[i]))
        ax.scatter(x_pos_jittered, data_indiv_full[i],
                   edgecolors="black", s=60, facecolors=color_[i], linewidth=1)

    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in color_[:len(color_custom)]]
    plt.legend(handles, x_labels[:len(label_custom)], borderpad=0.5)
        # , loc='lower center', bbox_to_anchor=(0.5, -0.18), ncol=4)

    ax.set_ylabel('CBE score')
    # ax.set_xticks(x_pos_full)
    # ax.set_xticklabels(x_labels)
    ax.set_xticks(x_tick_middle)
    ax.set_xticklabels(list(dat.keys()))
    ax.set_xlim(x_pos_full[0] - 0.6, x_pos_full[-1] + 0.6)
    # ax.set_xlabel("ROIs")
    # ax.yaxis.grid(True)
    plt.ylim(0, 1.6)
    plt.tight_layout()
    plt.savefig(f"cbe_600.png", dpi=600)
    plt.show()


def beta_anova(m_arr):
    m = m_arr.flatten()
    quality = np.repeat(["good", "bad"], 30)
    task = np.tile(np.repeat(["scenes", "fixation"], 15), 2)
    id = np.tile(np.arange(0, 15), 4)

    df = pd.DataFrame({"beta": m,
                       "quality": quality,
                       "task": task,
                       "sub_id": id})

    aovrm2way = AnovaRM(df, 'beta', 'sub_id', within=['quality', 'task'])
    res2way = aovrm2way.fit()

    a = (df.query('quality == "good"')).query("task == 'scenes'")["beta"]
    b = (df.query('quality == "bad"')).query("task == 'scenes'")["beta"]
    print("attended, between good bad: ", ttest_rel(a, b))

    a = (df.query('quality == "good"')).query("task == 'fixation'")["beta"]
    b = (df.query('quality == "bad"')).query("task == 'fixation'")["beta"]
    print("distracted, between good bad: ", ttest_rel(a, b))

    a = (df.query('quality == "good"')).query("task == 'scenes'")["beta"]
    b = (df.query('quality == "good"')).query("task == 'fixation'")["beta"]
    print("good, between attended and distracted: ", ttest_rel(a, b))

    a = (df.query('quality == "bad"')).query("task == 'scenes'")["beta"]
    b = (df.query('quality == "bad"')).query("task == 'fixation'")["beta"]
    print("bad, between attended and distracted: ", ttest_rel(a, b))

    print(res2way)


def grp_cbe():
    dat_all = {}
    dat_individual_all = {}
    for roi in helper.ROI:
        print("\n\n=====>> ", roi)
        cbe_all_sub = np.load(f"../GBAI2022_2/grp_cbe/cbe_{roi}_euclidean_all_cond.npy")

        # sep_err = within_sub_error(cbe_all_sub)

        # cbe_plot_sep(cbe_all_sub, sep_err)
        cbe_combined = []
        for i in range(0, cbe_all_sub.shape[1], 2):
            cbe_combined.append(np.mean(cbe_all_sub[:, i:i + 2], axis=1))
        cbe_combined = np.asarray(cbe_combined)
        beta_anova(cbe_combined)

        individual_dat = {'_'.join(helper.KEY_LS[i]): cbe_combined[i] for i in range(len(helper.KEY_LS))}
        comb_err = helper.within_sub_error(cbe_combined)

        dat = {'_'.join(helper.KEY_LS[i]): comb_err[i, :] for i in range(len(helper.KEY_LS))}
        dat_all[roi] = dat
        dat_individual_all[roi] = individual_dat
    print(dat_all)

    # plots.plot_bar_PPA(dat_all, dat_individual_all,
    #                    helper.PLOT_COLOR_COND, helper.PLOT_LB_PLOT, helper.PLOT_ORDER_COND,
    #                    mode="cbe", fname="raw_cbe_ppa_600.png")


def grp_cbe_apart():
    # pulling apart within versus between category dissimilarity
    plt_grp = ["good-attended-wn", "good-attended-bt", "good-distracted-wn", "good-distracted-bt",
               "bad-attended-wn", "bad-attended-bt", "bad-distracted-wn", "bad-distracted-bt"]

    for roi in helper.ROI:
        print(roi)
        cbe_all_sub = np.load(f"../GBAI2022_2/grp_cbe/cbe_{roi}_euclidean_all_cond_apart.npy")
        cbe_all_sub = np.transpose(cbe_all_sub, axes=(0, 2, 1))
        cbe_all_sub = np.reshape(cbe_all_sub, (15, 8))

       # cbe_plot_sep(cbe_all_sub, sep_err)
        # cbe_combined = []
        # for i in range(0, cbe_all_sub.shape[1], 2):
        #     cbe_combined.append(np.mean(cbe_all_sub[:, i:i + 2], axis=1))
        # cbe_combined = np.asarray(cbe_combined)
        comb_err = helper.within_sub_error(cbe_all_sub.T)
        dat = {plt_grp[i]: comb_err[i, :] for i in range(len(plt_grp))}
        print(dat)
        # continue


def plot_img_score():
    corr_all_sub = np.load("grp_cbe/corradv_PPA_all_cond.npy")

    all_score = np.sort(corr_all_sub[0, :, -1].flatten())

    df = pd.DataFrame(all_score, columns=["img"])

    df["qua"] = np.repeat(["bad", "good"], 80)

    sns.set_style('darkgrid')
    sns.set(rc={'figure.figsize': (10, 1.5)})
    sns.boxplot(data=df, x="img", y="qua", palette=["red", "blue"], width=0.5)
    sns.stripplot(data=df, x="img", y="qua", alpha = 0.3, palette=["red", "blue"])
    plt.xlim(0, 5)
    # plt.show()
    good = df[df["qua"] == "good"]
    bad = df[df["qua"] == "bad"]
    print(good.mean(), good.std())
    print(bad.mean(), bad.std())
    """
    good    4.92463 +- 0.080044
    bad     1.437336 +- 0.448287
    """


if __name__ == "__main__":
    os.chdir(helper.DATA_DIR)
    # main_corr_adv()
    # main_cbe()
    grp_cbe()
    # grp_cbe_apart()
    # plot_img_score()
