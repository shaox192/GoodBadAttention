import sys
from utils.params import *

from fMRIflow.postproc.mvpa import RsaRdm
from fMRIflow.postproc.mvpa.load_data import *
from scipy.stats import spearmanr, permutation_test, percentileofscore
import plotly.express as px
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns


def plot_mds(X_trans):
    df = pd.DataFrame(X_trans, columns = ['x_cor', 'y_cor'])
    df["full_condition"] = PLOT_ORDER
    df["quality"] = np.repeat(["good", "bad"], 80)
    df["task"] = np.tile(np.repeat(["scenes", "fixation"], 40), 2)
    df["category"] = np.tile(np.repeat(["cities", "mountains"], 20), 4)

    df = df[df["quality"] == "good"]
    # df = df[df["task"] == "scenes"]
    sns.scatterplot(data=df, x="x_cor", y="y_cor", hue="category")
    # plt.legend()
    plt.show()


def extract_diag(mat):
    ls = []
    for i in range(4):
        start = i * 40
        end = (i + 1) * 40
        arr = np.triu(mat[start:end, start:end], 1)
        arr = arr[arr != 0].flatten()
        ls.append(arr)
    return ls


def model_rdm():
    m0 = np.full((20, 20), 0.1)
    m1 = np.full((20, 20), 1)
    m = np.vstack((np.hstack((m0, m1)), np.hstack((m1, m0))))
    m = np.triu(m, 1)
    m = m[m != 0].flatten()
    m[m == 0.1] = 0
    return m


def generate_dist_mat(ls: list):
    dist_mat = np.empty(shape=(4, 4))
    for i in range(len(ls)):
        tri_i = np.triu(ls[i], k=1)
        tri_i = (tri_i[tri_i != 0]).flatten()
        for j in range(len(ls)):
            tri_j = np.triu(ls[j], k=1)
            tri_j = (tri_j[tri_j != 0]).flatten()
            dist_mat[i, j] = 1 - spearmanr(tri_i, tri_j)[0]
            dist_mat[j, i] = dist_mat[i, j]
    return dist_mat


def rsa_cross_condition():
    for roi in ROI:
        for dist in ["euclidean"]:  # , "pearsonsR", "cosine"
            dmat_all_sub = []
            for sub in SUB_OUT + SUB_LAB:
                save_dir = f"../../{sub}/{sub}_rsa"
                save_mat = os.path.join(save_dir, f"{sub}_{roi}_{dist}_rsd_mat_full.npy")
                rdm = np.load(save_mat)
                mat_cond = extract_diag(rdm)
                dmat_all_sub.append(generate_dist_mat(mat_cond))
            fig = px.imshow(np.mean(np.asarray(dmat_all_sub), axis=0),
                            x=['X'.join(k) for k in KEY_LS],
                            y=['X'.join(k) for k in KEY_LS],
                            text_auto=True, color_continuous_scale="balance")
            fig.data[0].update(zmin=0.5)
            fig.show()
            sys.exit()



def find_rdm_mean(rdm_ls):
    """account for the dropped miniblock for sub GBAI08@PPA-run4-blk16-bad-scenes-cities,
    this should be idx 48, (city_00328), need to manually pull out the order in Rdm class"""
    gbai08_idx = sorted(SUB_OUT + SUB_LAB).index("GBAI08")
    gbai08_rdm = rdm_ls[gbai08_idx]
    assert gbai08_rdm.shape == (159, 159)
    rdm_ls_14 = [rdm_ls[i] for i in range(len(rdm_ls)) if i != gbai08_idx]
    mean_rdm_14 = np.mean(rdm_ls_14, axis=0)

    gbai08_rdm = np.insert(gbai08_rdm, 48, np.zeros(gbai08_rdm.shape[1]), axis=0)
    gbai08_rdm = np.insert(gbai08_rdm, 48, np.zeros(gbai08_rdm.shape[0]), axis=1)

    gbai08_rdm[48, :] = mean_rdm_14[48, :]
    gbai08_rdm[:, 48] = mean_rdm_14[:, 48]

    mean_rdm = np.mean(rdm_ls_14 + [gbai08_rdm], axis=0)
    return mean_rdm


# def perm_spearmanr(x, y):
#     def get_rho(a, b, axis):
#         return spearmanr(a, b)[0]
#     rng = np.random.default_rng(1024)
#     res = permutation_test((x, y), spearmanr, n_resamples=15000, permutation_type="pairings", random_state=rng)
#     return res


def perm_spearmanr(x, y):
    r_actual = spearmanr(x, y)[0]

    r = []
    rng = np.random.default_rng(1020)
    for i in range(15000):
        y_new = rng.choice(y, size=y.shape)
        r.append(spearmanr(x, y_new)[0])

    r = np.asarray(r)
    p1 = np.sum(r > r_actual)/r.shape[0]
    p2 = np.sum(r < r_actual)/r.shape[0]

    return r_actual, np.min([p1, p2]) * 2, np.asarray(r)


def rsa_grp(roi):
    rdm_ls = []
    for sub in sorted(SUB_OUT + SUB_LAB):
        # print(sub)
        # save_dir = f"{sub}/{sub}_rsa"
        save_dir = f"../GBAI2022_2/{sub}/{sub}_rsa"
        save_mat = os.path.join(save_dir, f"{sub}_{roi}_euclidean_rsd_mat_full.npy")
        rdm = np.load(save_mat)
        rdm_ls.append(rdm)

    mean_rdm = find_rdm_mean(rdm_ls)
    diag_ls = extract_diag(mean_rdm)
    for d in diag_ls:
        res = perm_spearmanr(model_rdm(), d)
        print(res[:2])
        # print(res.statistic, res.pvalue)
        # print(res.null_distribution.shape)
        # plt.hist(res[-1], bins=30)
        # plt.show()
    # exit()
    # # load one batch of data for labeling
    # lb_dir = f"GBAI01/GBAI01_exp_behav/perception_behav_output"
    # # beta_dir = f"GBAI01/GBAI01_beta"
    # beta_dir = f"../GBAI2022_2/GBAI01/GBAI01_beta"
    # data_ls = load_data_beta("GBAI01", "PPA", lb_dir, beta_dir)
    # data_ls.sort()
    # RsaRdm().plot_rsa(rdm_mat=mean_rdm, data_ls=data_ls, save_fig=f"new_rdm_{roi}_600")

    # mds = MDS(dissimilarity="precomputed")
    # X_transformed = mds.fit_transform(rdm_ls)
    # plot_mds(X_transformed)
    # sys.exit()


def main_orig():

    for roi in ROI:
        for sub in SUB_OUT + SUB_LAB:
            #  =========== perception data load =========
            lb_dir = f"{sub}/{sub}_exp_behav/perception_behav_output"
            # beta_dir = f"{sub}/{sub}_beta"
            beta_dir = f"../GBAI2022_2/{sub}/{sub}_beta"
            data_ls = load_data_beta(sub, roi, lb_dir, beta_dir)
            rdm = RsaRdm(data_ls)
            rdm.rdm("euclidean")  # euclidean, pearsonsR
            print(sub, roi, rdm.rdm_mat.shape)

            # save_dir = f"{sub}/{sub}_rsa"
            save_dir = f"../GBAI2022_2/{sub}/{sub}_rsa"
            check_dir(save_dir)
            save_mat = os.path.join(save_dir, f"{sub}_{roi}_euclidean_rsd_mat_full")
            np.save(save_mat, rdm.rdm_mat)


def main():
    for roi in ROI:
        print("\n\n=======>>", roi)
        rsa_grp(roi)
        # sys.exit()


if __name__ == "__main__":
    os.chdir(DATA_DIR)
    main()
    # main_orig()
    # rsa_cross_condition()
