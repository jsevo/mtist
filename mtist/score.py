# Author: Jonas Schluter <jonas.schluter+github@gmail.com>
#        www.ramenlabs.science
#
# License: MIT

import pandas as pd
import numpy as np


def calculate_ED_score(truth: pd.DataFrame, inferred: pd.DataFrame) -> float:
    """
    Calculate the ecological direction (EDₙ) score (n := number of species in ecosystem).
    
    Parameters
    ===============
    truth: pandas.DataFrame(index=species_names, columns=species_names), the ecosystem coefficient matrix used to generate data
    inferred: pandas.DataFrame(index=species_names, columns=species_names), the inferred ecosystem coefficient matrix

    Returns
    ===============
    ED_score: float
    """
    # consider inferred coefficients
    mask = inferred != 0
    # compare sign: agreement when == -2 or +2, disagreement when 0
    nonzero_sign = np.sign(inferred)[mask] + np.sign(truth)[mask]
    corr_sign = (np.abs(nonzero_sign) == 2).sum().sum()
    opposite_sign = (np.abs(nonzero_sign) == 0).sum().sum()
    # count incorrect non-zero coefficients
    wrong_nz = (truth[mask] == 0).sum().sum()

    # combine
    unscaled_score = corr_sign - 2 * opposite_sign - 0.5 * wrong_nz

    # scale by theoretical extrema
    truth_nz_counts = (truth != 0).sum().sum()
    truth_z_counts = len(truth.index) ** 2 - truth_nz_counts
    theoretical_min = -2 * truth_nz_counts - 0.5 * truth_z_counts
    theoretical_max = truth_nz_counts

    # EDₙ score scaled between 0 and 1
    ED_score = (simple_score - theoretical_min) / (theoretical_max - theoretical_min)

    return ED_score
