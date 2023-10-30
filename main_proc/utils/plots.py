import numpy as np
import matplotlib.pyplot as plt
from . import params as helper


def plot_setup():
    font = {'family': 'Times New Roman', 'size': 22}
    plt.rc('font', **font)

def _plot_bar_roi_format(dat, dat_individual, plot_order, bar_width):
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


def plot_bar_roi(dat: dict, dat_individual: dict, color_custom, label_custom, plot_order, fname=None):
    """plot data from each roi on the same graph, for VSS poster"""

    randgen = np.random.default_rng(1024)
    bar_width = 0.28

    data_grp_full, sem_full, data_indiv_full, x_pos_full = [], [], [], []
    x_tick_middle = []
    for i, k in enumerate(dat):
        data_grp, sem, data_indiv, x_pos = _plot_bar_roi_format(dat[k], dat_individual[k], plot_order,
                                                                bar_width=bar_width)
        data_grp_full.append(data_grp)
        sem_full.append(sem)
        data_indiv_full.append(data_indiv)
        x_pos_full.append(x_pos + i * 0.5 * len(helper.PLOT_DICT))
        x_tick_middle.append(np.mean(x_pos_full[-1]))

    data_grp_full = np.concatenate(data_grp_full)
    sem_full = np.concatenate(sem_full)
    data_indiv_full = np.concatenate(data_indiv_full)
    x_pos_full = np.concatenate(x_pos_full) + 0.5
    x_tick_middle = np.asarray(x_tick_middle) + 0.5

    color_ = np.tile(color_custom, len(dat))
    x_labels = np.tile(label_custom, len(dat))

    fig, ax = plt.subplots(figsize=(18, 9.3))
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
    plt.legend(handles, x_labels[:len(label_custom)], loc='lower center', bbox_to_anchor=(0.5, -0.18), ncol=4)

    plt.axhline(y=0, color="black", linewidth=2, linestyle="dashed")

    ax.set_ylabel('%Signal Change')
    # ax.set_xticks(x_pos_full)
    # ax.set_xticklabels(x_labels)
    ax.set_xticks(x_tick_middle)
    ax.set_xticklabels(list(dat.keys()))
    ax.set_xlim(x_pos_full[0] - 0.6, x_pos_full[-1] + 0.6)
    plt.ylim(-0.26, 2)
    plt.yticks(np.arange(-0.25, 2, 0.25))
    plt.tight_layout()
    if fname:
        plt.savefig(f"univar_within_600.png", dpi=600)
    plt.show()


def plot_bar_PPA(dat: dict, dat_individual: dict, color_custom, label_custom, plot_order, mode="univar", fname=None):
    plot_setup()

    """plot only PPA data, this is for manuscript"""

    randgen = np.random.default_rng(1020)
    bar_width = 0.3

    data_grp_full, sem_full, data_indiv_full, x_pos_full = [], [], [], []
    for i, k in enumerate(dat):
        data_grp, sem, data_indiv, x_pos = _plot_bar_roi_format(dat[k], dat_individual[k], plot_order,
                                                                bar_width=bar_width)
        data_grp_full.append(data_grp)
        sem_full.append(sem)
        data_indiv_full.append(data_indiv)
        x_pos_full.append(x_pos + i * 0.5 * len(helper.PLOT_DICT))

    data_grp_full = np.concatenate(data_grp_full)
    sem_full = np.concatenate(sem_full)
    data_indiv_full = np.concatenate(data_indiv_full)
    x_pos_full = np.concatenate(x_pos_full) + 0.2
    x_tick_middle = np.asarray(x_pos_full)

    color_ = np.tile(color_custom, len(dat))
    x_labels = np.tile(label_custom, len(dat))

    fig, ax = plt.subplots(figsize=(8, 9.5))
    ax.bar(x_pos_full, data_grp_full, yerr=sem_full, align='center',
           color=color_, alpha=0.6, width=bar_width, linewidth=1.5,
           edgecolor='black', capsize=10)

    # plot scattered dots
    for i in range(x_pos_full.shape[0]):
        x_pos_jittered = np.repeat(x_pos_full[i], len(data_indiv_full[i])) \
                         + randgen.uniform(-0.12, 0.12, len(data_indiv_full[i]))
        ax.scatter(x_pos_jittered, data_indiv_full[i],
                   edgecolors="black", s=60, facecolors=color_[i], linewidth=1)

    if mode == "clf":
        plt.axhline(y=0.5, color="black", linewidth=2, linestyle="dashed")
        ax.set_ylabel("Accuracy")
        plt.ylim(0.3, 1)
        plt.yticks(np.arange(0.3, 1, 0.1))
    elif mode == "univar":
        ax.set_ylabel("%Signal Change")
        plt.ylim(0, 2.1)
        plt.yticks(np.arange(0.0, 2.1, 0.25))
    else:
        ax.set_ylabel("Category Boundary Effect Scores")
        plt.ylim(0, 1.6)
        plt.yticks(np.arange(0.0, 1.6, 0.25))

    ax.set_xticks(x_tick_middle)
    ax.set_xticklabels(x_labels)
    ax.set_xlim(x_pos_full[0] - 0.3, x_pos_full[-1] + 0.3)

    plt.tight_layout()
    if fname:
        plt.savefig(fname, dpi=600)

    plt.show()


def plot_runs_univar_PPA(dat_indiv, color_custom, cond_order, label, fname=None):
    plot_setup()

    plt.figure(figsize=(14, 9.3))

    x_lb = [f"Run{i}" for i in range(4)]
    x_pos = np.arange(4)
    for i in range(len(cond_order)):
        dat = dat_indiv[cond_order[i]]  # 4 runs * 2 (mean, sem)
        lb = label[i]
        shape = "--" if "distracted" in lb else '-'
        color = color_custom[i]

        mean_ts = dat[:, 0]
        err_ts = dat[:, 1]

        plt.errorbar(x=x_pos, y=mean_ts, yerr=err_ts, label=lb, color=color,
                     markerfacecolor=color, markeredgewidth=1.5, markeredgecolor="black",
                     linestyle=shape, markersize=12, capsize=7, linewidth=2,
                     marker='o')

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))

    plt.legend(by_label.values(), by_label.keys())

    plt.yticks(np.arange(0.6, 2, 0.2))
    plt.xticks(x_pos, x_lb)
    plt.xlim(x_pos[0] - 0.4, x_pos[-1] + 0.4)
    plt.grid(True, axis='y')
    plt.ylabel("%Signal Change")

    if fname:
        plt.savefig(fname, dpi=600)
    plt.show()
