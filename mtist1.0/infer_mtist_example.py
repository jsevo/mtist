import os
import glob

from mtist import mtist_utils as mu
from mtist import master_dataset_generation as mdg
from mtist import assemble_mtist as am
from mtist import infer_mtist as im

from matplotlib import pyplot as plt

################################
# INFER 3-SPECIES MTIST 4 WAYS #
################################

inference_names = [
    "default",
    "ridge_CV",
    "lasso_CV",
    "elasticnet_CV",
]

prefixes = [f"{name}_" for name in inference_names]

inference_fxn_handles = [
    im.infer_from_did,
    im.infer_from_did_ridge_cv,
    im.infer_from_did_lasso_cv,
    im.infer_from_did_elasticnet_cv,
]


for inference_type, prefix, handle in zip(inference_names, prefixes, inference_fxn_handles):
    print(inference_type)
    im.INFERENCE_DEFAULTS.INFERENCE_PREFIX = prefix
    im.INFERENCE_DEFAULTS.INFERENCE_FUNCTION = handle
    _ = im.infer_and_score_all(save_inference=True, save_scores=True)