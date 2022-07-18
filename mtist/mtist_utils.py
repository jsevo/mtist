import os
from collections import OrderedDict

import numpy as np
import pandas as pd

from mtist import lvsimulator

from mtist import assemble_mtist as am
from mtist import master_dataset_generation as mdg

from functools import reduce
import operator


class GLOBALS:

    names_100_sp = [
        "species_0",
        "species_1",
        "species_2",
        "species_3",
        "species_4",
        "species_5",
        "species_6",
        "species_7",
        "species_8",
        "species_9",
        "zz_random_sp0",
        "zz_random_sp1",
        "zz_random_sp2",
        "zz_random_sp3",
        "zz_random_sp4",
        "zz_random_sp5",
        "zz_random_sp6",
        "zz_random_sp7",
        "zz_random_sp8",
        "zz_random_sp9",
        "zz_random_sp10",
        "zz_random_sp11",
        "zz_random_sp12",
        "zz_random_sp13",
        "zz_random_sp14",
        "zz_random_sp15",
        "zz_random_sp16",
        "zz_random_sp17",
        "zz_random_sp18",
        "zz_random_sp19",
        "zz_random_sp20",
        "zz_random_sp21",
        "zz_random_sp22",
        "zz_random_sp23",
        "zz_random_sp24",
        "zz_random_sp25",
        "zz_random_sp26",
        "zz_random_sp27",
        "zz_random_sp28",
        "zz_random_sp29",
        "zz_random_sp30",
        "zz_random_sp31",
        "zz_random_sp32",
        "zz_random_sp33",
        "zz_random_sp34",
        "zz_random_sp35",
        "zz_random_sp36",
        "zz_random_sp37",
        "zz_random_sp38",
        "zz_random_sp39",
        "zz_random_sp40",
        "zz_random_sp41",
        "zz_random_sp42",
        "zz_random_sp43",
        "zz_random_sp44",
        "zz_random_sp45",
        "zz_random_sp46",
        "zz_random_sp47",
        "zz_random_sp48",
        "zz_random_sp49",
        "zz_random_sp50",
        "zz_random_sp51",
        "zz_random_sp52",
        "zz_random_sp53",
        "zz_random_sp54",
        "zz_random_sp55",
        "zz_random_sp56",
        "zz_random_sp57",
        "zz_random_sp58",
        "zz_random_sp59",
        "zz_random_sp60",
        "zz_random_sp61",
        "zz_random_sp62",
        "zz_random_sp63",
        "zz_random_sp64",
        "zz_random_sp65",
        "zz_random_sp66",
        "zz_random_sp67",
        "zz_random_sp68",
        "zz_random_sp69",
        "zz_random_sp70",
        "zz_random_sp71",
        "zz_random_sp72",
        "zz_random_sp73",
        "zz_random_sp74",
        "zz_random_sp75",
        "zz_random_sp76",
        "zz_random_sp77",
        "zz_random_sp78",
        "zz_random_sp79",
        "zz_random_sp80",
        "zz_random_sp81",
        "zz_random_sp82",
        "zz_random_sp83",
        "zz_random_sp84",
        "zz_random_sp85",
        "zz_random_sp86",
        "zz_random_sp87",
        "zz_random_sp88",
        "zz_random_sp89",
    ]

    MASTER_DATASET_DIR = "master_datasets"
    GT_DIR = "ground_truths"
    MTIST_DATASET_DIR = "mtist_datasets"
    TOY_DATASET_DIR = "toy_master_datasets"

    GT_NAMES = [
        "3_sp_gt_1",
        "3_sp_gt_2",
        "3_sp_gt_3",
        "10_sp_gt_1",
        "10_sp_gt_2",
        "10_sp_gt_3",
        "100_sp_gt",
    ]


def create_lv_dicts(aij, gr):
    """Convert aij and gr, represented in ndarrays, into dictionaries compatible with ecosims package

    Args:
        aij (ndarray): n_species x n_species Aij matrix (rows: focal species, columns: interacting species)
        gr (ndarray): nspecies x 1 matrix, growth rates for each species

    Returns:
        tuple: (ecosystem, growth_rate) dictionary tuple
    """
    n_species = len(gr)

    # 100 species requires diferent names since the order of
    # the species ecosims requires
    if n_species == 100:
        species_names = NAMES_100_SP.copy()
    else:
        species_names = [f"species_{i}" for i in range(n_species)]

    ecosystem = OrderedDict()
    # Outer loop, for the each focal species
    for i in range(n_species):
        ecosystem[species_names[i]] = OrderedDict()

        # Inner loop, for each of the interactions (j) for the ith focal species
        for j in range(n_species):
            ecosystem[species_names[i]][species_names[j]] = aij[i, j]

    growth_rates = OrderedDict()
    for i in range(n_species):
        growth_rates[species_names[i]] = gr[i]

    return ecosystem, growth_rates


def load_ground_truths(path=None):
    """Loads GTs from path with GTs saved in standardized format

    Args:
        path (path-like): path to ground truth folder. If None, defaults to the default folder.

    Returns:
        tuple: (aijs, grs) tuple of ground truth aijs and grs matrices indexed by gt names
    """

    if path is None:
        path = GLOBALS.GT_DIR

    path_to_aijs = lambda v: os.path.join(path, "interaction_coefficients", v + ".csv")
    path_to_grs = lambda v: os.path.join(path, "growth_rates", v + ".csv")

    gt_names = GLOBALS.GT_NAMES

    aij_names = pd.Series(gt_names).str.replace("gt", "aij").to_list()
    gr_names = pd.Series(gt_names).str.replace("gt", "gr").to_list()

    # aij_names = [
    #     "3_sp_aij_1",
    #     "3_sp_aij_2",
    #     "3_sp_aij_3",
    #     "10_sp_aij_1",
    #     "10_sp_aij_2",
    #     "10_sp_aij_3",
    #     "100_sp_aij",
    # ]

    # gr_names = [
    #     "3_sp_gr_1",
    #     "3_sp_gr_2",
    #     "3_sp_gr_3",
    #     "10_sp_gr_1",
    #     "10_sp_gr_2",
    #     "10_sp_gr_3",
    #     "100_sp_gr",
    # ]

    # gt_names = [
    #     "3_sp_gt_1",
    #     "3_sp_gt_2",
    #     "3_sp_gt_3",
    #     "10_sp_gt_1",
    #     "10_sp_gt_2",
    #     "10_sp_gt_3",
    #     "100_sp_gt",
    # ]

    # Create file names
    aij_fn_to_load = map(path_to_aijs, aij_names)
    gr_fn_to_load = map(path_to_grs, gr_names)

    aijs = OrderedDict(zip(gt_names, [np.loadtxt(fn, delimiter=",") for fn in aij_fn_to_load]))
    grs = OrderedDict(zip(gt_names, [np.loadtxt(fn, delimiter=",") for fn in gr_fn_to_load]))

    return aijs, grs


def load_dataset(csv):
    """from csv filepath to full_df, X"""
    full_df = pd.read_csv(csv).drop(columns="Unnamed: 0")

    species_cols = full_df.columns[full_df.columns.str.contains("species_")]
    X = full_df[species_cols].values

    time = full_df.iloc[:, 0].values

    # meta specific for this dataset
    meta_spec = full_df.drop(columns=["time"] + species_cols.to_list())

    return full_df, time, X, meta_spec


def load_master_dataset(mdid):

    full_df = pd.read_csv(
        os.path.join(GLOBALS.MASTER_DATASET_DIR, f"master_dataset_{mdid}.csv")
    ).drop(
        columns=["Unnamed: 0"],
    )

    n_species = full_df["n_species"].unique()[0]

    sp_cols = [f"species_{i}" for i in range(n_species)]

    X = full_df[sp_cols].values
    time = full_df["time"].values

    meta_cols = full_df.columns[~full_df.columns.str.contains("species_")].drop("time")
    meta_spec = full_df[meta_cols]

    return full_df, time, X, meta_spec


def simulate(aij, gr, seed, noise, tend, dt, sample_freq):
    """Simulate a timeseries for a specific aij, gr, seed, and noise level

    Since the 100-species names need that 'zz' on it, if len(gr)==100,
    species names will be initialized from global variable NAMES_100_SP.
    """

    n_species = len(gr)

    # Make ndarrays into dictionaries
    eco, gro = create_lv_dicts(aij, gr)

    # Get proper species_names
    if n_species == 100:
        species_names = NAMES_100_SP.copy()
    else:
        species_names = [f"species_{i}" for i in range(n_species)]

    # Initialize the proper initial conditions
    rng = np.random.default_rng(seed)
    yinit_specific = dict(zip(species_names, rng.integers(1, 10, n_species) / 100))

    # Run the simulation
    lv = lvsimulator.LV(ecosystem=eco.copy(), growth_rates=gro.copy())

    t, y, t_all, y_all, yinit = lv.run_lv(
        random_seed=seed,
        tend=tend,
        dt=dt,
        yinit_specific=yinit_specific,
        noise=noise,
        sample_freq=sample_freq,
    )

    return t, y


def calculate_es_score(true_aij, inferred_aij) -> float:
    """GRANT'S edited version to calculate ED score

    Calculate the ecological direction (EDₙ) score (n := number of species in ecosystem).

    Parameters
    ===============
    truth: pandas.DataFrame(index=species_names, columns=species_names), the ecosystem coefficient matrix used to generate data
    inferred: pandas.DataFrame(index=species_names, columns=species_names), the inferred ecosystem coefficient matrix
    Returns
    ===============
    ES_score: float
    """

    truth = pd.DataFrame(true_aij).copy()
    inferred = pd.DataFrame(inferred_aij).copy()

    # consider inferred coefficients
    mask = inferred != 0

    # compare sign: agreement when == -2 or +2, disagreement when 0
    nonzero_sign = np.sign(inferred)[mask] + np.sign(truth)[mask]
    corr_sign = (np.abs(nonzero_sign) == 2).sum().sum()
    opposite_sign = (np.abs(nonzero_sign) == 0).sum().sum()

    # count incorrect non-zero coefficients
    wrong_nz = (truth[mask] == 0).sum().sum()

    # combine
    unscaled_score = corr_sign - opposite_sign

    # scale by theoretical extrema
    truth_nz_counts = (truth != 0).sum().sum()
    truth_z_counts = len(truth.index) ** 2 - truth_nz_counts
    theoretical_min = -truth_nz_counts
    theoretical_max = truth_nz_counts

    ES_score = (unscaled_score - theoretical_min) / (theoretical_max - theoretical_min)

    return ES_score


def calculate_n_datasets():
    """Read current set of parameters across global variable class definitions,
    returns the number of datasets to be produced by that setup.

    Returns:
        int: number of datasets in MTIST as defined
    """

    number_of_params = dict(
        n_timeseries=len(am.ASSEMBLE_MTIST_DEFAULTS.N_TIMESERIES_PARAMS),
        n_timepoints=len(am.ASSEMBLE_MTIST_DEFAULTS.SAMPLING_FREQ_PARAMS),
        n_noises=len(mdg.MASTER_DATASET_DEFAULTS.NOISE_SCALES),
        n_ecosystems=len(GLOBALS.GT_NAMES),
        n_seq_depths=2,  # Hard coded,
        n_sampling_schemes=len(am.ASSEMBLE_MTIST_DEFAULTS.SAMPLING_SCHEME_PARAMS),
    )
    n_datasets = reduce(
        operator.mul, list(number_of_params.values())
    )  # fancy one-line multiplication

    return n_datasets


## SOME STUFF ##
NAMES_100_SP = GLOBALS.names_100_sp
