import os
import glob

from mtist import mtist_utils as mu
from mtist import master_dataset_generation as mdg
from mtist import assemble_mtist as am
from mtist import infer_mtist as im

from matplotlib import pyplot as plt

###############################
# GENERATE MTIST              #
###############################

mdg.generate_mtist_master_datasets()

plt.close("all")

am.assemble_mtist()
