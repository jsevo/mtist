import os
import glob

from mtist import mtist_utils as mu
from mtist import master_dataset_generation as mdg
from mtist import assemble_mtist as am
from mtist import infer_mtist as im

import numpy as np
import matplotlib.pyplot as plt

###############################
# GENERATE MTIST              #
###############################

mu.GLOBALS.MASTER_DATASET_DIR = "master_datasets_extended"
mu.GLOBALS.MTIST_DATASET_DIR = "mtist_datasets_extended"
mu.GLOBALS.GT_DIR = "ground_truths_extended"
mu.GLOBALS.GT_NAMES = [f"10_sp_aij_{i}" for i in range(50)]

am.ASSEMBLE_MTIST_DEFAULTS.N_TIMESERIES_PARAMS = np.arange(1, 31, 1)
am.ASSEMBLE_MTIST_DEFAULTS.SAMPLING_FREQ_PARAMS = [5, 10, 15]
am.ASSEMBLE_MTIST_DEFAULTS.SAMPLING_SCHEME_PARAMS = ["even", "seq", "random"]

mdg.MASTER_DATASET_DEFAULTS.NOISE_SCALES = [0.1, 0.01]

mdg.generate_mtist_master_datasets(save_example_figures=False)

plt.close("all")

am.assemble_mtist()
