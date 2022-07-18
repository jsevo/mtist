import os

import numpy as np
import pandas as pd

from mtist import mtist_utils as mu


class ASSEMBLE_MTIST_DEFAULTS:
    RANDOM_SEED = 89237560
    seq_depth_th = 0.01
    # RNG = np.random.default_rng(RANDOM_SEED)

    N_TIMESERIES_PARAMS = [5, 10, 50]
    SAMPLING_FREQ_PARAMS = [5, 10, 15]
    SAMPLING_SCHEME_PARAMS = ["even", "random", "seq"]


### GLOBAL VARIABLES ###
# RNG = ASSEMBLE_MTIST_DEFAULTS.RNG


def gen_even_idx(sf):
    """sf is sampling_frequency"""
    idx = np.linspace(0, 99, sf, dtype=int)
    return idx


def gen_seq_idx(sf, rng):
    """sf is sampling_frequency
    This function will break with sf > 15"""

    # 25 days simulated
    morning_of_each_day = np.linspace(0, 99, 25, dtype=int)

    # randomly choose a starting day from the first ten days
    start = rng.integers(0, 10)

    # beginning at "start", give me sf number of mornings
    idx = morning_of_each_day[start : start + sf]

    return idx


def gen_random_idx(sf, rng):
    # 25 days simulated
    morning_of_each_day = np.linspace(0, 99, 25, dtype=int)
    idx = sorted(rng.choice(morning_of_each_day, size=sf, replace=False))
    return idx


def implement_low_seq_depth():

    n_datasets = mu.calculate_n_datasets()

    offset = int(n_datasets / 2)
    high_seq_depth_ends_at = int(n_datasets / 2)

    # fns = glob.glob(os.path.join(mu.GLOBALS.MTIST_DATASET_DIR, "dataset_*.csv"))
    fns = [
        os.path.join(mu.GLOBALS.MTIST_DATASET_DIR, f"dataset_{i}.csv")
        for i in range(high_seq_depth_ends_at)
    ]

    for fn in fns:
        df = pd.read_csv(fn).drop(columns="Unnamed: 0")
        did = df["did"].unique()[0]
        new_did = did + offset

        # Get the species from each dataframe
        sp_cols = df.columns.str.contains("species_")
        abun = df.iloc[:, sp_cols].copy()

        # Get a mask of those to remove
        mask = abun.apply(lambda r: r / r.sum(), axis=1) < ASSEMBLE_MTIST_DEFAULTS.seq_depth_th

        # Set those to 0
        abun[mask] = 0

        # Replace original df
        new_df = df.copy()
        new_df.iloc[:, sp_cols] = abun

        new_df["seq_depth"] = "low"
        new_df["did"] = new_did

        new_df.to_csv(os.path.join(mu.GLOBALS.MTIST_DATASET_DIR, f"dataset_{new_did}.csv"))


def generate_metadata():
    # fns = glob.glob(os.path.join(mu.GLOBALS.MTIST_DATASET_DIR, "dataset_*.csv"))

    n_datasets = mu.calculate_n_datasets()

    fns = [
        os.path.join(mu.GLOBALS.MTIST_DATASET_DIR, f"dataset_{i}.csv") for i in range(n_datasets)
    ]

    meta = pd.DataFrame([])
    i = 0
    for fn in fns:
        i = i + 1

        # This try/except block here is because some of my
        df = pd.read_csv(fn).drop(columns="Unnamed: 0")

        # Gather metadata
        sd = df["seq_depth"].unique()[0]
        did = df["did"].unique()[0]
        n_species = df["n_species"].unique()[0]
        noise = df["noise"].unique()[0]
        gt = df["ground_truth"].unique()[0]
        ss = df["sampling_scheme"].unique()[0]
        n_timepoints = df["n_timepoints"].unique()[0]
        n_timeseries = len(df["timeseries_id"].unique())

        # FIRST, crosscheck
        n_sp_crosscheck = df.columns.str.contains("species_").sum()
        n_tp_crosscheck = np.unique([len(subset) for (_, subset) in df.groupby("timeseries_id")])[0]

        assert (
            n_sp_crosscheck == n_species
        ), f"n_sp crosscheck failure: from df {n_species}, from crosscheck {n_sp_crosscheck}"

        assert (
            n_tp_crosscheck == n_timepoints
        ), f"n_tp crosscheck failure: from df {n_timepoints}, from crosscheck {n_tp_crosscheck}"

        # Check to make sure these "unique" arrays are all len() == 1
        to_check = [
            "seq_depth",
            "did",
            "n_species",
            "noise",
            "ground_truth",
            "n_timepoints",
            "sampling_scheme",
        ]
        for each in to_check:
            assert len(df[each].unique()) == 1, f"unique array len of {each} is not 1"

        # Create the next meta row
        cur_meta_row = pd.DataFrame(
            [did, n_species, gt, noise, n_timeseries, n_timepoints, ss, sd],
            index=[
                "did",
                "n_species",
                "ground_truth",
                "noise",
                "n_timeseries",
                "n_timepoints",
                "sampling_scheme",
                "seq_depth",
            ],
        ).T

        # Combine
        meta = pd.concat((meta, cur_meta_row))

    # meta = meta.set_index("did").sort_index()

    return meta


def assemble_mtist():

    # mdataset_fps = glob.glob(os.path.join(mu.GLOBALS.MASTER_DATASET_DIR, "master_dataset_*.csv"))

    ## Gather what master datasets/conditions will go into each mtist dataset ##
    master_meta = pd.read_csv(
        os.path.join(mu.GLOBALS.MASTER_DATASET_DIR, "master_metadata.csv")
    ).set_index("master_did")

    # Collect indices for the datasets per `name`, `noise`
    grp = master_meta.groupby(["name", "noise"])

    name_noise_dict = {}
    for (name, noise), df in grp:
        name_noise_dict[(name, noise)] = df.index

    # Distribute the n_timeseries throughout
    # the noise/ground truth combinations
    # n_timeseries_params = [5, 10, 50]
    n_timeseries_params = ASSEMBLE_MTIST_DEFAULTS.N_TIMESERIES_PARAMS

    name_noise_nts_dict = {}
    for name, noise in name_noise_dict.keys():

        # In this inner loop, make name_noise_nts an expanded
        # version of name_noise_dict that now also includes the
        # "n_timeseries variable"
        for each_n_timeseries in n_timeseries_params:
            name_noise_nts_dict[(name, noise, each_n_timeseries)] = name_noise_dict[name, noise][
                0:each_n_timeseries
            ]

    # Finally, duplicate out the conditions for all parameters
    # sampling_scheme_params = ["even", "random", "seq"]
    # sampling_freq_params = [5, 10, 15]
    sampling_freq_params = ASSEMBLE_MTIST_DEFAULTS.SAMPLING_FREQ_PARAMS
    sampling_scheme_params = ASSEMBLE_MTIST_DEFAULTS.SAMPLING_SCHEME_PARAMS

    full_conditions_dict = {}
    for name, noise, nts in name_noise_nts_dict.keys():
        for ss in sampling_scheme_params:
            for sf in sampling_freq_params:

                # Just copy those indices since they'll be the same for each
                # combination of sample_scheme and sample_frequency
                full_conditions_dict[name, noise, nts, ss, sf] = name_noise_nts_dict[
                    name, noise, nts
                ].copy()

    ## ASSEMBLE DATASETS AND SAVE ##

    # Preparation
    try:
        os.mkdir(mu.GLOBALS.MTIST_DATASET_DIR)
    except Exception as e:
        print(e)

    # Start counting mtist indices
    did = 0

    # Create each MTIST dataset given the conditions in the full_conditions_dict
    for name, noise, nts, ss, sf in full_conditions_dict.keys():
        mdids_to_load = full_conditions_dict[name, noise, nts, ss, sf]
        df = pd.DataFrame([])

        # Create a df to process by ss (sampling_scheme) and sf (sampling_frequency)
        for mdid in mdids_to_load:
            df = pd.concat(
                (
                    df,
                    pd.read_csv(
                        os.path.join(mu.GLOBALS.MASTER_DATASET_DIR, f"master_dataset_{mdid}.csv")
                    ).drop(columns="Unnamed: 0"),
                )
            )

        # Initiate rng
        cur_rng = np.random.default_rng(ASSEMBLE_MTIST_DEFAULTS.RANDOM_SEED)

        # Start sampling
        df_sampled = pd.DataFrame([])
        for tid, subset in df.groupby("timeseries_id"):
            sp_cols = subset.columns[subset.columns.str.contains("species_")]

            if ss == "even":
                idx = gen_even_idx(sf)

            elif ss == "random":
                idx = gen_random_idx(sf, cur_rng)

            elif ss == "seq":
                idx = gen_seq_idx(sf, cur_rng)

            df_sampled = pd.concat((df_sampled, subset.iloc[idx]))

        df_sampled = df_sampled.reset_index(drop=True).assign(
            did=did, seq_depth="high", n_timeseries=nts, n_timepoints=sf, sampling_scheme=ss
        )

        df_sampled.to_csv(os.path.join(mu.GLOBALS.MTIST_DATASET_DIR, f"dataset_{did}.csv"))

        did = did + 1

    ## IMPLEMENT LOW-SEQ-DEPTH ##
    implement_low_seq_depth()

    ## GENERATE METADATA ##
    meta = generate_metadata()
    meta.to_csv(os.path.join(mu.GLOBALS.MTIST_DATASET_DIR, "mtist_metadata.csv"))


def assemble_mtist_custom():
    """Function for compatibility reasons"""
    assemble_mtist()
