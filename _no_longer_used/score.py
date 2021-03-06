# Author: Jonas Schluter <jonas.schluter+github@gmail.com>
#        www.ramenlabs.science
#
# License: MIT

import pandas as pd
import numpy as np


def calculate_ES_score(truth: pd.DataFrame, inferred: pd.DataFrame) -> float:
    """
    Calculate the ecological sign (ESₙ) score (n := number of species in ecosystem).
    
    Parameters
    ===============
    truth: pandas.DataFrame(index=species_names, columns=species_names), the ecosystem coefficient matrix used to generate data
    inferred: pandas.DataFrame(index=species_names, columns=species_names), the inferred ecosystem coefficient matrix
    """
    # consider inferred coefficients
    mask = inferred!=0
    # compare sign: agreement when == -2 or +2, disagreement when 0
    nonzero_sign = np.sign(inferred)[mask] + np.sign(truth)[mask]
    corr_sign = (np.abs(nonzero_sign[mask])==2).sum().sum()
    opposite_sign = (np.abs(nonzero_sign[mask])==0).sum().sum()
    # count incorrect non-zero coefficients
    wrong_nz = (truth[mask]==0).sum().sum()
    # missed interactions
    missed_nz = (truth[~mask]!=0).sum().sum()

    # scale by theoretical extrema
    truth_nz_counts = (truth!=0).sum().sum()
    truth_z_counts = len(truth.index)**2 - truth_nz_counts
    theoretical_min = -truth_nz_counts
    theoretical_max = truth_nz_counts
    
    # combine 
    unscaled_score = corr_sign - opposite_sign
    
    # ESₙ score scaled between 0 and 1
    ES_score = (unscaled_score - theoretical_min )/ (theoretical_max-theoretical_min)
    return ES_score
