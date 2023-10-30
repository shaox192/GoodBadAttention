###################################
# IMPORTANT!!: lag folder are fro -1 to 7, but in reality it was from 0 to 8. The
# python code for 
###################################

import sys
import time
from typing import Dict, List, Optional
from scipy.stats import percentileofscore, norm
from statsmodels.stats.anova import AnovaRM
from pandas import DataFrame
import numpy as np
from copy import deepcopy
import utils.params as helper
import utils.plots as plots

from fMRIflow.postproc.mvpa import ClfMod, ClfPerm
from fMRIflow.postproc.mvpa.load_data import *
from fMRIflow.postproc.mvpa.base_obj.utils import subselect_cond

import matplotlib.pyplot as plt
import pickle as pkl

font = {'family': 'Times New Roman', 'size': 26}
plt.rc('font', **font)


class ClassificationWrapper:
    def __init__(self, sub="", roi=""):
        self.sub = sub
        self.roi = roi

    def _decode_within(self, data_ls: List, save_mat=None, save_report=None, plot=True):
        clf = ClfMod(data_ls)
        accuracy = clf.clf_loro_linear()
        print(f"Accuracy: {accuracy}")

        clf.calc_confusion_mat(save_mat)
        if save_mat is not None:
            print(f"saving confusion mat to: {save_mat}")
        clf.clf_report(save_report)
        if save_report is not None:
            print(f"saving classification report to {save_report}")
        if plot:
            print("Plotting confusion mat...")
            clf.clf_plot_confusion_mat()

        return clf.confusion_mat

    def _decode_cross(self, data_ls_train, data_ls_test, save_mat=None, save_report=None, plot=False):

        clf_train_d = ClfMod(data_ls_train)
        clf_test_d = ClfMod(data_ls_test)

        clf_test_d.bold_dat, *_ = clf_test_d.scale_pca_data(clf_test_d.bold_dat, scale_on=True, pca_on=False)

        accuracy = clf_train_d.clf_test(clf_test_d)
        print(f"original accuracy: {accuracy}")

        clf_train_d.calc_confusion_mat(save_mat)
        if save_mat is not None:
            print(f"saving confusion mat to: {save_mat}")
        clf_train_d.clf_report(save_report)
        if save_report is not None:
            print(f"saving classification report to {save_report}")
        if plot:
            print("Plotting confusion mat...")
            clf_train_d.clf_plot_confusion_mat()

        return clf_train_d.confusion_mat

    def clf_within_cond_wrapper(self, save_dir: str, lb_dir: str, beta_dir: str):
        helper.check_dir(save_dir)
        for key in helper.KEY_LS:
            save_mat_fpath = os.path.join(save_dir, f"{self.sub}_{self.roi}_confusion_mat_{'_'.join(key)}")
            save_report_fpath = os.path.join(save_dir, f"{self.sub}_{self.roi}_clf_report_{'_'.join(key)}")

            data_ls = load_data_beta(self.sub, self.roi, lb_dir, beta_dir, helper.ATTR_LS, key)

            self._decode_within(data_ls, save_mat=save_mat_fpath, save_report=save_report_fpath, plot=False)

    def clf_cross_cond_wrapper(self, save_dir: str, lb_dir: str, beta_dir: str):
        helper.check_dir(save_dir)
        data_ls = load_data_beta(self.sub, self.roi, lb_dir, beta_dir)

        # train on attended, test on fixation separately for good and bad
        for train_k, test_k in [(helper.KEY_LS[0], helper.KEY_LS[2]), (helper.KEY_LS[1], helper.KEY_LS[3])]:
            save_mat_fpath = os.path.join(save_dir,
                                          f"{self.sub}_{self.roi}_confusion_mat_"
                                          f"train_{'_'.join(train_k)}_"
                                          f"test_{'_'.join(test_k)}")
            save_report_fpath = os.path.join(save_dir, f"{self.sub}_{self.roi}_clf_report_"
                                                       f"train_{'_'.join(train_k)}_"
                                                       f"test_{'_'.join(test_k)}")

            train_dset = utils.subselect_cond(data_ls, helper.ATTR_LS, train_k)
            test_dset = utils.subselect_cond(data_ls, helper.ATTR_LS, test_k)
            self._decode_cross(train_dset, test_dset, save_mat=save_mat_fpath, save_report=save_report_fpath)


def main_clf(anal: str):
    """anal: can be within, cross """
    clf = ClassificationWrapper()

    for sub in helper.SUB_OUT + helper.SUB_LAB:
        clf.sub = sub
        lb_dir = f"{sub}/{sub}_exp_behav/perception_behav_output"
        for roi in helper.ROI:
            clf.roi = roi
            if anal == "within":
                beta_dir = f"../GBAI2022_2/{sub}/{sub}_beta"
                # beta_dir = f"~/Documents/GBAI2022/temp/{sub}_beta_typeb"
                save_dir = f"../GBAI2022_2/{sub}/{sub}_clf/loro_within_cond"
                # save_dir = "~/Documents/GBAI2022/clf_typeb"
                clf.clf_within_cond_wrapper(save_dir, lb_dir, beta_dir)
            elif anal == "cross":
                # beta_dir = f"{sub}/{sub}_beta"
                beta_dir = f"../GBAI2022_2/{sub}/{sub}_beta"
                # save_dir = f"{sub}/{sub}_clf/loro_cross_cond"
                save_dir = f"../GBAI2022_2/{sub}/{sub}_clf/loro_cross_cond"
                clf.clf_cross_cond_wrapper(save_dir, lb_dir, beta_dir)


#### ======================================= PERMUTATION not organized ===================================
def permutate_within(sub, roi, num_perm, perm_id):
    print(sub, roi)
    lb_dir = f"{sub}/{sub}_exp_behav/perception_behav_output"
    # beta_dir = f"{sub}/{sub}_beta"
    beta_dir = f"../GBAI2022_2/{sub}/{sub}_beta"

    data_ls = load_data_beta(sub, roi, lb_dir, beta_dir)
    for key in range(4):
        dat = subselect_cond(data_ls, helper.ATTR_LS, helper.KEY_LS[key])
        orig_lb = [obj.category for obj in dat]  # len: 40
        clf = ClfPerm(dat)
        clf.setup_skeleton_within()

        def _perm_clf(i):
            print("iteration:", i)
            clf.label = np.random.permutation(orig_lb)
            acc = clf.clf_perm()
            return acc

        # start = time.time()
        accuracy = [_perm_clf(i) for i in range(num_perm)]
        # print(time.time() - start)
        print(sub, helper.KEY_LS[key], len(accuracy))

        save_dir = f"../GBAI2022_2/{sub}/{sub}_clf/perm"
        helper.check_dir(save_dir)
        np.save(f"{save_dir}/{sub}_{roi}_perm{perm_id}_{'_'.join(helper.KEY_LS[key])}.npy", np.asarray(accuracy))


def permutate_cross_decoding(sub, roi, num_perm, perm_id):
    print(sub, roi)
    lb_dir = f"{sub}/{sub}_exp_behav/perception_behav_output"
    beta_dir = f"{sub}/{sub}_beta"
    data_ls = load_data_beta(sub, roi, lb_dir, beta_dir)

    save_dir = f"{sub}/{sub}_clf/perm"
    helper.check_dir(save_dir)

    def _perm_clf(dat_train, dat_test):
        orig_lb = [o.category for o in dat_train] + [b.category for b in dat_test]

        clf_train_d = ClfPerm(dat_train)
        clf_test_d = ClfPerm(dat_test)
        clf_test_d.bold_dat, *_ = clf_test_d.scale_pca_data(clf_test_d.bold_dat, scale_on=True, pca_on=False)
        clf_train_d.setup_skeleton_cross(clf_test_d)

        accuracy = []
        # s = time.time()
        for i in range(num_perm):
            print("iteration: ", i)
            lb = np.random.permutation(orig_lb)
            clf_train_d.label = lb[:(len(lb) // 2)]
            clf_test_d.label = lb[(len(lb) // 2):]
            accuracy.append(clf_train_d.clf_perm_cross(clf_test_d))
        # print(time.time() - s)
        return accuracy

    # cross decode 1 for good:
    data_train = subselect_cond(data_ls, helper.ATTR_LS, helper.KEY_LS[0])
    data_test = subselect_cond(data_ls, helper.ATTR_LS, helper.KEY_LS[2])
    acc = _perm_clf(data_train, data_test)
    print("cross good: ", len(acc), end=', ')
    np.save(f"{save_dir}/{sub}_{roi}_perm{perm_id}_train_{'_'.join(helper.KEY_LS[0])}_test_{'_'.join(helper.KEY_LS[2])}.npy",
            np.asarray(acc))

    # cross decode 2 for bad:
    data_train = subselect_cond(data_ls, helper.ATTR_LS, helper.KEY_LS[1])
    data_test = subselect_cond(data_ls, helper.ATTR_LS, helper.KEY_LS[3])
    acc = _perm_clf(data_train, data_test)
    np.save(f"{save_dir}/{sub}_{roi}_perm{perm_id}_train_{'_'.join(helper.KEY_LS[1])}_test_{'_'.join(helper.KEY_LS[3])}.npy",
            np.asarray(acc))
    print("cross bad: ", len(acc))


#### ======================================= PERMUTATION not organized ===================================


def main_perm():
    # # ======= permutation within =======
    num_perm = 2000
    for sub in sorted(helper.SUB_LAB + helper.SUB_OUT):
        for roi in helper.ROI:
            permutate_within(sub, roi, num_perm, perm_id=103)

    # # combine all perm files to one with 10,000 iterations
    # for roi in ROI:
    #     for key in range(len(KEY_LS)):
    #         save_dir = "grp_perm"
    #         check_dir(save_dir)
    #         full_perm = []
    #         for sub in sorted(SUB_LAB + SUB_OUT):
    #             sub_dir = f"{sub}/{sub}_clf/perm"
    #             full_perm.append(np.concatenate([np.load(f"{sub_dir}/{sub}_{roi}_perm{i}_{'_'.join(KEY_LS[key])}.npy")
    #                                              for i in [101, 102, 103]]))
    #         full_perm = np.vstack(full_perm)
    #         np.save(f"{save_dir}/{roi}_{'_'.join(KEY_LS[key])}_fullperm.npy", full_perm)

    # # ======= permutation cross =======
    # num_perm = 10000
    # for sub in SUB_LAB + SUB_OUT:
    #     for roi in ROI:
    #         permutate_cross_decoding(sub, roi, num_perm, perm_id=101)

    # for roi in helper.ROI:
    #     full_perm_good = []
    #     full_perm_bad = []
    #     for sub in sorted(helper.SUB_LAB + helper.SUB_OUT):
    #         save_dir = f"{sub}/{sub}_clf/perm"
    #         full_perm_good.append(np.load(f"{save_dir}/{sub}_{roi}_perm101_"
    #                                       f"train_{'_'.join(helper.KEY_LS[0])}_test_{'_'.join(helper.KEY_LS[2])}.npy"))
    #
    #         full_perm_bad.append(np.load(f"{save_dir}/{sub}_{roi}_perm101_"
    #                                      f"train_{'_'.join(helper.KEY_LS[1])}_test_{'_'.join(helper.KEY_LS[3])}.npy"))
    #
    #     np.save(f"grp_perm/{roi}_train_{'_'.join(helper.KEY_LS[0])}_test_{'_'.join(helper.KEY_LS[2])}_fullperm.npy",
    #             np.vstack(full_perm_good))
    #     np.save(f"grp_perm/{roi}_train_{'_'.join(helper.KEY_LS[1])}_test_{'_'.join(helper.KEY_LS[3])}_fullperm.npy",
    #             np.vstack(full_perm_bad))


#### ======================================== Post analysis ==============================================
def _beta_anova(dat_dict, roi):
    dat = []
    for i in ["good", "bad"]:
        for j in ["scenes", "fixation"]:
            dat += dat_dict['_'.join([i, j])]

    sub_id = np.tile(np.arange(15), 4)
    quality = np.repeat(["good", "bad"], 30)
    task = np.tile(np.repeat(["scenes", "fixation"], 15), 2)
    df = DataFrame({"dat": dat, "sub_id": sub_id, "quality": quality, "task": task})
    df.to_csv(f"grp_perm/clf_{roi}_forR.csv")
    aovrm2way = AnovaRM(df, 'dat', 'sub_id', within=["quality", "task"])
    res2way = aovrm2way.fit()
    print(res2way)


def grp_plot_bar(roi, plot=True, perm=None):
    cond_name = ['_'.join([q, t]) for q in helper.IM_QUALITY for t in helper.EXP_TASK]

    def _load_all_sub(roi, key):
        accuracy_cond = []
        for sub in helper.SUB_OUT + helper.SUB_LAB:
            # save_dir = f"{sub}/{sub}_clf/loro_within_cond"
            save_dir = f"../GBAI2022_2/{sub}/{sub}_clf/loro_within_cond"
            save_mat = os.path.join(save_dir, f"{sub}_{roi}_confusion_mat_{key}.npy")
            cm = np.load(save_mat)
            accuracy_cond.append(np.mean(cm.diagonal()))

        return accuracy_cond

    def _2meansem(dt: dict):
        mat = [val for _, val in dt.items()]
        mean_sem = helper.within_sub_error(np.vstack(mat))

        count = 0
        for key, _ in dt.items():
            dt[key] = mean_sem[count]
            count += 1
        return dt

    all_cond_individual = {i: _load_all_sub(roi, i) for i in cond_name}

    _beta_anova(deepcopy(all_cond_individual), roi)
    all_cond = _2meansem(deepcopy(all_cond_individual))
    print("Accuracy scores and sem: ", all_cond)
    for i in [3]: # range(4):
        curr_cond = '_'.join(helper.KEY_LS[i])
        # print(perm[curr_cond].shape)
        # # print(all_cond[curr_cond][0])
        # plt.hist(perm[curr_cond], bins=30)
        # plt.show()
        # exit()
        p = np.sum(all_cond[curr_cond][0] > perm[curr_cond])/perm[curr_cond].shape[0]
        print("perm results for each: ", curr_cond, 1 - p) # percentileofscore(perm[curr_cond], all_cond[curr_cond][0]))
    # exit()
    if perm is not None:
        all_cond_pairs = [['_'.join(helper.KEY_LS[0]), '_'.join(helper.KEY_LS[2])],
                          ['_'.join(helper.KEY_LS[1]), '_'.join(helper.KEY_LS[3])],
                          ['_'.join(helper.KEY_LS[0]), '_'.join(helper.KEY_LS[1])],
                          ['_'.join(helper.KEY_LS[2]), '_'.join(helper.KEY_LS[3])]]

        ## ============== between good and bad ================
        for cond1, cond2 in all_cond_pairs:
            print("perm results for between", cond1, cond2, end=': ')
            print(percentileofscore(perm[cond1] - perm[cond2], all_cond[cond1][0] - all_cond[cond2][0]))

        # added to see if attention benefits good more than bad:
        null_dist = perm["good_scenes"] - perm["good_fixation"] - (perm["bad_scenes"] - perm["bad_fixation"])
        actual_dat = all_cond["good_scenes"] - all_cond["good_fixation"] - (
                    all_cond["bad_scenes"] - all_cond["bad_fixation"])
        print("attention benefits good more than bad?", percentileofscore(null_dist, actual_dat[0]))

    return all_cond, all_cond_individual


def grp_plot_cross(roi, within_cond: dict, within_cond_individual, perm=None, plot=True):
    """
    answer whether attended and fixation are similar in good and bad:
    bs -> bf;
    gs -> gf;
    :return:
    """
    cond_name = ["train_bad_scenes_test_bad_fixation", "train_good_scenes_test_good_fixation"]
    wn_name = ["bad_fixation", "good_fixation"]

    def _load_all_sub(roi, key):
        accuracy_cond = []
        for sub in helper.SUB_OUT + helper.SUB_LAB:
            # save_dir = f"{sub}/{sub}_clf/loro_cross_cond"
            save_dir = f"../GBAI2022_2/{sub}/{sub}_clf/loro_cross_cond"
            save_mat = os.path.join(save_dir, f"{sub}_{roi}_confusion_mat_{key}.npy")
            cm = np.load(save_mat)
            accuracy_cond.append(np.mean(cm.diagonal()))

        return accuracy_cond

    def _2meansem(dt: dict):
        mat = [val for _, val in dt.items()]
        mean_sem = helper.within_sub_error(np.vstack(mat))

        count = 0
        for key, _ in dt.items():
            dt[key] = mean_sem[count]
            count += 1
        return dt

    all_cond_individual = {i: _load_all_sub(roi, i) for i in cond_name}
    all_cond_individual.update({i: within_cond_individual[i] for i in wn_name})

    all_cond = _2meansem(deepcopy(all_cond_individual))
    all_cond.update({i: within_cond[i] for i in wn_name})
    print(all_cond)
    print("train_good_test_good_fixation", percentileofscore(perm[:, 0], all_cond["train_good_scenes_test_good_fixation"][0]))
    print("train_bad_test_bad_fixation", percentileofscore(perm[:, 2], all_cond["train_bad_scenes_test_bad_fixation"][0]))

    if perm is not None:
        print("train_good scenes - fix", end='')
        print(percentileofscore(perm[:, 0] - perm[:, 1],
                                all_cond["train_good_scenes_test_good_fixation"][0] - all_cond["good_fixation"][0]))
        print("train_bad scenes - fix", end='')
        print(percentileofscore(perm[:, 2] - perm[:, 3],
                                all_cond["train_bad_scenes_test_bad_fixation"][0] - all_cond["bad_fixation"][0]))

    return all_cond, all_cond_individual


def load_perm_within(roi: str) -> Dict[str, np.ndarray]:
    perm_dict = {}
    for i in range(len(helper.KEY_LS)):
        dat = np.load(f"grp_perm/{roi}_{'_'.join(helper.KEY_LS[i])}_fullperm.npy")
        perm_dict['_'.join(helper.KEY_LS[i])] = np.mean(dat, axis=0)
    return perm_dict


def load_perm_across(roi):
    cond_good = np.load(f"grp_perm/{roi}_train_{'_'.join(helper.KEY_LS[0])}_test_{'_'.join(helper.KEY_LS[2])}_fullperm.npy")
    cond_bad = np.load(f"grp_perm/{roi}_train_{'_'.join(helper.KEY_LS[1])}_test_{'_'.join(helper.KEY_LS[3])}_fullperm.npy")

    all_sub = np.vstack([np.mean(cond_good, axis=0), np.mean(cond_bad, axis=0)])
    all_sub = all_sub.T

    return all_sub


def pop_indices(d: np.ndarray, val2pop: List[int]) -> np.ndarray:
    """

    Args:
        data_ls: A list of the TR list [[1,2,3, ...], [1,1,1, ...], []]
        val2pop: values that should be popped, e.g. -1

    Returns:
    """
    mask = np.empty(d.shape, dtype=bool)
    for i in range(d.shape[1]):
        mask[:, i] = True if all([np.any(val not in d[:, i]) for val in val2pop]) else False
    return d[mask].reshape(d.shape[0], -1)


def main_anal():
    all_data_dict = {}
    for roi in helper.ROI:
        print("---->> ", roi)
        # ================== grp plot
        # ## bar for within condition
        perm_w = load_perm_within(roi)
        within_cond_dict, within_cond_dict_individual = grp_plot_bar(roi, plot=False, perm=perm_w)
        print("\n\n")
        # ## bar for across condition
        perm_c = load_perm_across(roi)
        perm_c = np.insert(perm_c, 1, perm_w["good_fixation"], axis=1)
        perm_c = np.insert(perm_c, 3, perm_w["bad_fixation"], axis=1)
        cross_cond_dict, cross_cond_dict_individual = grp_plot_cross(roi, within_cond_dict, within_cond_dict_individual,
                                                                     perm=perm_c, plot=False)

        all_data_dict[roi] = {"within_cond_dict": within_cond_dict,
                              "within_cond_dict_individual": within_cond_dict_individual,
                              "cross_cond_dict": cross_cond_dict,
                              "cross_cond_dict_individual": cross_cond_dict_individual}

    # with open("../GBAI2022_2/clf_results4plot.pkl", 'wb') as f:
    #     pkl.dump(all_data_dict, f)


def plot_reshape(all_data_dict, which_clf):
    bar_dict, scatter_dict = {}, {}
    for k, v in all_data_dict.items():
        bar_dict[k] = v[f"{which_clf}_cond_dict"]
        scatter_dict[k] = v[f"{which_clf}_cond_dict_individual"]
    return bar_dict, scatter_dict


def main_plot():
    # load data
    all_data_dict = helper.pickle_load("../GBAI2022_2/clf_results4plot.pkl")
    roi = "PPA"

    ### PLOT within condition decoding
    # bar_data, individual_data = plot_reshape(all_data_dict, "within")
    # plots.plot_bar_PPA({roi: bar_data[roi]}, {roi: individual_data[roi]},
    #                    helper.PLOT_COLOR_COND, helper.PLOT_LB_PLOT, helper.PLOT_ORDER_COND,
    #                    clf=True, fname="clf_within_ppa.png")
    # sys.exit()

    ### PLOT cross condition decoding
    bar_data, individual_data = plot_reshape(all_data_dict, "cross")
    plot_order = ["train_good_scenes_test_good_fixation", "good_fixation", "train_bad_scenes_test_bad_fixation",
                  "bad_fixation"]
    label_cus = ["train: good-attended\ntest: good-distracted",
                 "train: good-distracted\ntest: good-distracted",
                 "train: bad-attended\ntest: bad-distracted",
                 "train: bad-distracted\ntest: bad-distracted"]
    color_cus = ["blue", "steelblue", "red", "salmon"]
    plots.plot_bar_PPA({roi: bar_data[roi]}, {roi: individual_data[roi]},
                       color_cus, label_cus, plot_order,
                       mode="clf", fname="clf_cross_ppa.png")


if __name__ == "__main__":
    os.chdir(helper.DATA_DIR)
    # main_plot()
    main_anal()
    # main_clf("cross")
    # main_perm()
