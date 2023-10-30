import numpy as np
import os
import pickle

DATA_DIR = "/Volumes/WD_BLACK/Good_bad_Followup/GBAI2022"

# exp params
FMRI_TR = 1.35
FMRI_RUN = 8
FMRI_BLK = 20

SUB_LAB = ["GBAI01", "GBAI02", "GBAI05", "GBAI06", "GBAI11"]
SUB_OUT = ["GBAI03", "GBAI04", "GBAI07", "GBAI08", "GBAI09", "GBAI10", "GBAI12", "GBAI13", "GBAI14", "GBAI15"]
ROI = ["PPA", "OPA", "RSC"]
ROI_PLOT = ["PPA", "MPA/RSC", "OPA"]

ATTR_LS = ["quality", "task"]
OLD_KEY_LS = [["good", "scenes"], ["good", "fixation"], ["bad", "scenes"], ["bad", "fixation"]]

# This was the previous 4 conditions for reference
KEY_LS = [["good", "scenes"], ["bad", "scenes"], ["good", "fixation"], ["bad", "fixation"]]

IM_CATEGORY = ["cities", "mountains"]
EXP_TASK = ["fixation", "scenes"]
IM_QUALITY = ["bad", "good"]

PLOT_ORDER = ["good_scenes_cities", "good_scenes_mountains",
              "bad_scenes_cities", "bad_scenes_mountains",
              "good_fixation_cities", "good_fixation_mountains",
              "bad_fixation_cities", "bad_fixation_mountains"]
PLOT_COLOR = ["blue", "blue", "red", "red", "steelblue", "steelblue", "salmon", "salmon"]
PLOT_LB = ['\n'.join(i.split('_')) for i in PLOT_ORDER]

PLOT_ORDER_COND = ["good_scenes", "bad_scenes", "good_fixation", "bad_fixation"]
PLOT_COLOR_COND = ["blue", "red", "steelblue", "salmon"]
PLOT_LB_COND = ["good-attended", "bad-attended", "good-distracted", "bad-distracted"]
PLOT_LB_PLOT = ["Good\nAttended", "Bad\nAttended", "Good\nDistracted", "Bad\nDistracted"]


PLOT_DICT = {"good_scenes": {"color": "blue", "label": "Good-Attended"},
             "bad_scenes": {"color": "red", "label": "Bad-Attended"},
             "good_fixation": {"color": "steelblue", "label": "Good-Distracted"},
             "bad_fixation": {"color": "salmon", "label": "Bad-Distracted"},}

def within_sub_error(arr: np.ndarray, num_cond: int =None, num_sub: int =None) -> np.ndarray:
    """
    Args:
        arr: condition * subjects

    Returns:
        A np array where first column is mean and second column is the sem
    """
    num_cond = num_cond if num_cond else arr.shape[0]
    num_sub = num_sub if num_sub else arr.shape[1]

    mean_cond = np.mean(arr, axis=1)
    correction_factor = np.sqrt(num_cond / (num_cond - 1))  # J: number of conditions

    for j in range(arr.shape[0]):
        for i in range(arr.shape[1]):
            arr[j, i] = correction_factor * (arr[j, i] - mean_cond[j]) + mean_cond[j]

    sem = np.std(arr, axis=1) / np.sqrt(num_sub)
    return np.hstack((mean_cond.reshape(-1, 1), sem.reshape(-1, 1)))


def check_dir(f):
    if not os.path.exists(f):
        print(f"path: {f} does not exist, creating ...")
        os.makedirs(f)


def clear_dir(dir_rm):
    if os.path.isfile(dir_rm) and os.path.exists(dir_rm):
        print(f"deleting a file: {dir_rm} ...")
        os.remove(dir_rm)
    elif os.path.isdir(dir_rm):
        print(f"deleting directory: {dir_rm} ...")
        for sub_f in os.listdir(dir_rm):
            if os.path.isdir(sub_f):
                print("sub directory exists!")
            else:
                if not sub_f.startswith('.'):
                    os.remove(os.path.join(dir_rm, sub_f))


def pickle_load(fname:str):
    with open(fname, 'rb') as f:
        c = pickle.load(f)
    return c


def pickle_dump(fname:str, item):
    with open(fname, 'wb') as f:
        pickle.dump(item, f)
