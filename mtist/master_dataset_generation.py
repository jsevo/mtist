import os

import numpy as np
import pandas as pd

from mtist.graphing_utils import despine, easy_subplots, savefig
from mtist import mtist_utils as mu


class MASTER_DATASET_DEFAULTS:

    dt = 0.1
    tend = 30
    sample_freq = 100

    # fmt: off
    random_seeds = [   36656,  2369231,   416304, 10488077,  8982779, 12733201,
        9845126,  9036584,  5140131,  8493390,  3049039,  2753893,
       11563241,  5589942,  2091765,  2905119,  4240255, 10011807,
        5576645,   591973,  4211685,  9275155, 10793741,    41300,
        2858482,  6550368,  3346496, 12305126,  8717317,  6543552,
        5614865,  9104526, 10435541, 11942766,  6667140, 10471522,
         115475,  2721265,   309357,  9668522,  2698393,  9638443,
       11499954,  1444356,  8745245,  7964854,  1768742,  8139908,
       10646715, 10999907]
    
    expanded_random_seeds = [36656, 2369231, 416304, 10488077, 8982779, 
    12733201, 9845126, 9036584, 5140131, 8493390, 
    3049039, 2753893, 11563241, 5589942, 20965,
    2905119, 4240255, 10011807, 5576645, 591973,
    4211685, 9275155, 10793741, 95498165, 2858482, 
    6550368, 3346496, 15949819, 8717317, 6543552, 
    5614865, 9104526, 10435541, 11942766, 6667140,
    10471522, 115475, 2721265, 309357, 9668522,
    2698393, 9638443, 11499954, 1444356, 8745245,
    7964854, 1768742, 8139908, 10646715, 10999907, 
    3891503, 508475, 2491670, 1362179, 167142, 
    2849045, 346009, 1379404, 5193332, 757320,
    479506, 4249123, 3971818, 454229, 2800725, 
    802453, 811789, 1729287, 382484, 605864, 
    644652, 2347953, 16178, 1789594, 4309409,
    126945, 4046783, 3827639, 550594, 5428888, 
    3040434, 1284913, 8251921, 1135245, 3667530, 
    452012, 1228708, 3303413, 3696988, 1189699,
    4435178, 639845, 6602376, 164941, 1166057,
    5125548, 2109804, 3712830, 5461680, 3889621]

    """Exapnded random seeds generated via: 

    old_random_seeds = random_seeds
    random.seed(89237560)
    extra_50_random_seeds = [round((random.random()*random.randint(500000,10000000))) for i in range(50)]
    expanded_random_seeds = old_random_seeds + extra_50_random_seeds"""

    # fmt: on

    NOISE_SCALES = [0.01, 0.10]


def generate_mtist_master_datasets(save_datasets=True, save_example_figures=True):
    """mu.simulate and save master datasets by parameters outlined in MASTER_DATASET_DEFAULTS class"""

    ### Gather current conditions ###
    random_seeds = MASTER_DATASET_DEFAULTS.random_seeds
    tend = MASTER_DATASET_DEFAULTS.tend
    dt = MASTER_DATASET_DEFAULTS.dt
    sample_freq = MASTER_DATASET_DEFAULTS.sample_freq
    noises = MASTER_DATASET_DEFAULTS.NOISE_SCALES

    # Initialize conditions
    conditions = []
    for seed in random_seeds:
        for noise in noises:
            conditions.append((seed, noise))

    # Load ground truths
    aijs, grs = mu.load_ground_truths(mu.GLOBALS.GT_DIR)

    gt_names = mu.GLOBALS.GT_NAMES

    ### DO THE SIMULATIONS ###
    # Index then by name, seed, noise
    results = {}
    for name, aij, gr in zip(gt_names, aijs.values(), grs.values()):
        for seed, noise in conditions:
            t, y = mu.simulate(aij, gr, seed, noise, tend, dt, sample_freq)
            results[(name, seed, noise)] = t, y

    ### MAKE RESULTS INTO FORMATTED DATAFRAME ###

    # Make preliminary results df
    df_results = pd.DataFrame.from_dict(results, orient="index", columns=["times", "abundances"])
    index_tuples = df_results.index  # to extract the tuples

    # Create columns from the original `results` dictionary keys
    # Combine with df_results
    expanded_tuple_index = pd.DataFrame(index_tuples.to_list(), columns=["name", "seed", "noise"])
    df_results = expanded_tuple_index.join(df_results.reset_index(drop=True))

    # add in the n_species name
    n_species_col = df_results["name"].str.split("_", expand=True)[0].to_frame(name="n_species")
    df_results = n_species_col.join(df_results)

    # Set the index right
    df_results.index.name = "master_did"

    ### SAVE IF NEEDED ###
    if save_datasets:
        try:
            os.mkdir(mu.GLOBALS.MASTER_DATASET_DIR)
        except Exception as e:
            print(e)

        # SAVE the metadata
        df_results[["name", "seed", "noise"]].to_csv(
            os.path.join(mu.GLOBALS.MASTER_DATASET_DIR, "master_metadata.csv")
        )

        # SAVE the master datasets
        for idx in df_results.index:

            # Get integer number of species
            n_species = int(df_results.loc[idx, "n_species"])

            # Obtain metadata
            name, seed, noise = df_results.loc[idx, ["name", "seed", "noise"]]

            # Create dataframe of only time/abundances
            time_and_abundances = np.hstack(
                (df_results.iloc[idx, :].times, df_results.iloc[idx, :].abundances)
            )

            # Combine time/abundances dataframe with the metadata
            formatted_master_df = pd.DataFrame(
                time_and_abundances, columns=["time"] + [f"species_{i}" for i in range(n_species)]
            ).assign(ground_truth=name, timeseries_id=seed, noise=noise, n_species=n_species)

            # Save each dataset indexed by master dataset index
            formatted_master_df.to_csv(
                os.path.join(mu.GLOBALS.MASTER_DATASET_DIR, f"master_dataset_{idx}.csv")
            )

    elif save_datasets is False:
        return df_results

    if save_example_figures:
        plot_master_datasets(df_results, save=True)


def plot_master_datasets(df_results, save=False):
    """Generate example figures from the final master dataset dataframe

    Args:
        df_results (pd.DataFrame): Results dataframe, requires generation by generate_mtist_master_datasets function
    """

    # Plot all ground truths
    grp = df_results.groupby(["name", "n_species", "noise"])

    k = 0
    for (name, n_species, noise), df in grp:

        #     Outer loop gets a 50-row dataset of all seeds at a single value of noise/ground truth (and thus, 'name')

        fig, axes = easy_subplots(ncols=5, nrows=10, base_figsize=(3, 2))
        for i, (ax, seed) in enumerate(zip(axes, df["seed"].unique())):
            n_species = int(n_species)

            cur_time = df.query("seed == @seed")["times"].values[0]
            cur_abundances = df.query("seed == @seed")["abundances"].values[0]

            [ax.plot(cur_time, cur_abundances[:, i_sp]) for i_sp in range(n_species)]
            ax.set_title(seed)

        fig.suptitle(f"ground_truth_{name}_noise_{noise}")

        despine(fig)

        if save:
            savefig(
                fig,
                os.path.join(
                    mu.GLOBALS.MASTER_DATASET_DIR, f"master_dataset_graphed_{name}_noise_{noise}"
                ),
                ft="jpg",
            )


class TOY_DATASET_DEFAULTS:

    gt_names = [
        "3_sp_gt_1",
        "3_sp_gt_2",
        "3_sp_gt_3",
        "10_sp_gt_1",
        "10_sp_gt_2",
        "10_sp_gt_3",
        "100_sp_gt",
    ]

    TOY_MASTER_DIR = "toy_master_dir"


def generate_toy_datasets(save_datasets=True, plot_example_figures=True):
    """Same code as the generate_master_datasets function, but allow for the exclusion
    of certain GTs for debugging purposes."""

    ### Gather current conditions ###
    random_seeds = MASTER_DATASET_DEFAULTS.random_seeds
    tend = MASTER_DATASET_DEFAULTS.tend
    dt = MASTER_DATASET_DEFAULTS.dt
    sample_freq = MASTER_DATASET_DEFAULTS.sample_freq
    noises = MASTER_DATASET_DEFAULTS.NOISE_SCALES

    # Initialize conditions
    conditions = []
    for seed in random_seeds:
        for noise in noises:
            conditions.append((seed, noise))

    # Load ground truths
    aijs, grs = mu.load_ground_truths(mu.GLOBALS.GT_DIR)

    # fmt: off
    gt_names = TOY_DATASET_DEFAULTS.gt_names # CHANGE EXISTS HERE

    ### DO THE SIMULATIONS ###
    # Index then by name, seed, noise
    results = {}
    for name in gt_names:               # CHANGE EXISTS HERE
        aij = aijs[name]                # CHANGE EXISTS HERE
        gr = grs[name]                  # CHANGE EXISTS HERE
        for seed, noise in conditions:
            t, y = mu.simulate(aij, gr, seed, noise, tend, dt, sample_freq)
            results[(name, seed, noise)] = t, y

    # fmt: on

    ### MAKE RESULTS INTO FORMATTED DATAFRAME ###

    # Make preliminary results df
    df_results = pd.DataFrame.from_dict(results, orient="index", columns=["times", "abundances"])
    index_tuples = df_results.index  # to extract the tuples

    # Create columns from the original `results` dictionary keys
    # Combine with df_results
    expanded_tuple_index = pd.DataFrame(index_tuples.to_list(), columns=["name", "seed", "noise"])
    df_results = expanded_tuple_index.join(df_results.reset_index(drop=True))

    # add in the n_species name
    n_species_col = df_results["name"].str.split("_", expand=True)[0].to_frame(name="n_species")
    df_results = n_species_col.join(df_results)

    # Set the index right
    df_results.index.name = "master_did"

    ### SAVE IF NEEDED ###
    if save_datasets:
        try:
            os.mkdir(mu.GLOBALS.TOY_DATASET_DIR)
        except Exception as e:
            print(e)

        # SAVE the metadata
        df_results[["name", "seed", "noise"]].to_csv(
            os.path.join(mu.GLOBALS.TOY_DATASET_DIR, "master_metadata.csv")
        )

        # SAVE the master datasets
        for idx in df_results.index:

            # Get integer number of species
            n_species = int(df_results.loc[idx, "n_species"])

            # Obtain metadata
            name, seed, noise = df_results.loc[idx, ["name", "seed", "noise"]]

            # Create dataframe of only time/abundances
            time_and_abundances = np.hstack(
                (df_results.iloc[idx, :].times, df_results.iloc[idx, :].abundances)
            )

            # Combine time/abundances dataframe with the metadata
            formatted_master_df = pd.DataFrame(
                time_and_abundances, columns=["time"] + [f"species_{i}" for i in range(n_species)]
            ).assign(ground_truth=name, timeseries_id=seed, noise=noise, n_species=n_species)

            # Save each dataset indexed by master dataset index
            formatted_master_df.to_csv(
                os.path.join(mu.GLOBALS.TOY_DATASET_DIR, f"master_dataset_{idx}.csv")
            )

    elif save_datasets is False:
        return df_results

    if plot_example_figures:
        plot_master_datasets(df_results, save=False)


def generate_mtist_master_datasets_selective(
    ecosystems, save_datasets=True, save_example_figures=True
):
    """mu.simulate and save master datasets by parameters outlined in MASTER_DATASET_DEFAULTS class"""

    ### Gather current conditions ###
    random_seeds = MASTER_DATASET_DEFAULTS.random_seeds
    tend = MASTER_DATASET_DEFAULTS.tend
    dt = MASTER_DATASET_DEFAULTS.dt
    sample_freq = MASTER_DATASET_DEFAULTS.sample_freq
    noises = MASTER_DATASET_DEFAULTS.NOISE_SCALES

    # Initialize conditions
    conditions = []
    for seed in random_seeds:
        for noise in noises:
            conditions.append((seed, noise))

    # Load ground truths
    aijs, grs = mu.load_ground_truths(mu.GLOBALS.GT_DIR)
    # gt_names = mu.GOLBALS.GT_NAMES # this gets overwritten by "ecosystems" and is unneeded

    # Only use a certain selection of ground truths
    aijs_selective = {each: aijs[each] for each in ecosystems}
    grs_selective = {each: grs[each] for each in ecosystems}
    gt_names = ecosystems

    aijs = aijs_selective
    grs = grs_selective

    ### DO THE SIMULATIONS ###
    # Index then by name, seed, noise
    results = {}
    for name, aij, gr in zip(gt_names, aijs.values(), grs.values()):
        for seed, noise in conditions:
            t, y = mu.simulate(aij, gr, seed, noise, tend, dt, sample_freq)
            results[(name, seed, noise)] = t, y

    ### MAKE RESULTS INTO FORMATTED DATAFRAME ###

    # Make preliminary results df
    df_results = pd.DataFrame.from_dict(results, orient="index", columns=["times", "abundances"])
    index_tuples = df_results.index  # to extract the tuples

    # Create columns from the original `results` dictionary keys
    # Combine with df_results
    expanded_tuple_index = pd.DataFrame(index_tuples.to_list(), columns=["name", "seed", "noise"])
    df_results = expanded_tuple_index.join(df_results.reset_index(drop=True))

    # add in the n_species name
    n_species_col = df_results["name"].str.split("_", expand=True)[0].to_frame(name="n_species")
    df_results = n_species_col.join(df_results)

    # Set the index right
    df_results.index.name = "master_did"

    ### SAVE IF NEEDED ###
    if save_datasets:
        try:
            os.mkdir(mu.GLOBALS.MASTER_DATASET_DIR)
        except Exception as e:
            print(e)

        # SAVE the metadata
        df_results[["name", "seed", "noise"]].to_csv(
            os.path.join(mu.GLOBALS.MASTER_DATASET_DIR, "master_metadata.csv")
        )

        # SAVE the master datasets
        for idx in df_results.index:

            # Get integer number of species
            n_species = int(df_results.loc[idx, "n_species"])

            # Obtain metadata
            name, seed, noise = df_results.loc[idx, ["name", "seed", "noise"]]

            # Create dataframe of only time/abundances
            time_and_abundances = np.hstack(
                (df_results.iloc[idx, :].times, df_results.iloc[idx, :].abundances)
            )

            # Combine time/abundances dataframe with the metadata
            formatted_master_df = pd.DataFrame(
                time_and_abundances, columns=["time"] + [f"species_{i}" for i in range(n_species)]
            ).assign(ground_truth=name, timeseries_id=seed, noise=noise, n_species=n_species)

            # Save each dataset indexed by master dataset index
            formatted_master_df.to_csv(
                os.path.join(mu.GLOBALS.MASTER_DATASET_DIR, f"master_dataset_{idx}.csv")
            )

    elif save_datasets is False:
        return df_results

    if save_example_figures:
        plot_master_datasets(df_results, save=True)
