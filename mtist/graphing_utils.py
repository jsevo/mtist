from collections.abc import Iterable

import matplotlib.pyplot as plt
import seaborn as sns

from mtist import mtist_utils as mu
import math


def easy_subplots(ncols=1, nrows=1, base_figsize=None, **kwargs):

    if base_figsize is None:
        base_figsize = (8, 5)

    fig, axes = plt.subplots(
        ncols=ncols,
        nrows=nrows,
        figsize=(base_figsize[0] * ncols, base_figsize[1] * nrows),
        **kwargs,
    )

    # Lazy way of doing this
    try:
        axes = axes.reshape(-1)
    except:
        pass

    return fig, axes


def despine(fig=None, axes=None):
    if fig is not None:
        sns.despine(trim=True, offset=0.5, fig=fig)

    elif axes is not None:

        if not isinstance(axes, Iterable):  # to generalize to a single ax
            axes = [axes]

        for ax in axes:
            sns.despine(trim=True, offset=0.5, ax=ax)

    else:
        fig = plt.gcf()
        sns.despine(trim=True, offset=0.5, fig=fig)


def savefig(fig, filename, ft=None):
    if ft is None:
        ft = "jpg"
    fig.savefig("{}.{}".format(filename, ft), dpi=300, bbox_inches="tight")


def score_heatmap(meta, df_es_scores, plot_floored=True, plot_low_seq_depth=True, **kwargs):
    """meta should have index of did
    No longer works now that sep-depth removed"""

    # Get heatmaps across seq_depth low, high and raw, floored scores

    hm_high_raw = (
        df_es_scores.join(meta)
        .query('seq_depth == "high"')
        .groupby(["n_species", "noise", "n_timeseries", "sampling_scheme", "n_timepoints"])
        .median()
        .pivot_table(
            index=["noise", "sampling_scheme", "n_timepoints"],
            columns=["n_species", "n_timeseries"],
            values="raw",
        )
    )

    hm_low_raw = (
        df_es_scores.join(meta)
        .query('seq_depth == "low"')
        .groupby(["n_species", "noise", "n_timeseries", "sampling_scheme", "n_timepoints"])
        .median()
        .pivot_table(
            index=["noise", "sampling_scheme", "n_timepoints"],
            columns=["n_species", "n_timeseries"],
            values="raw",
        )
    )

    hm_high_floored = (
        df_es_scores.join(meta)
        .query('seq_depth == "high"')
        .groupby(["n_species", "noise", "n_timeseries", "sampling_scheme", "n_timepoints"])
        .median()
        .pivot_table(
            index=["noise", "sampling_scheme", "n_timepoints"],
            columns=["n_species", "n_timeseries"],
            values="floored",
        )
    )

    hm_low_floored = (
        df_es_scores.join(meta)
        .query('seq_depth == "low"')
        .groupby(["n_species", "noise", "n_timeseries", "sampling_scheme", "n_timepoints"])
        .median()
        .pivot_table(
            index=["noise", "sampling_scheme", "n_timepoints"],
            columns=["n_species", "n_timeseries"],
            values="floored",
        )
    )

    if plot_floored:
        to_plot = [hm_high_raw, hm_low_raw, hm_high_floored, hm_low_floored]

        ax_titles = [
            "SeqDepth==High, Non-floored ES Score",
            "SeqDepth==Low, Non-floored ES Score",
            "SeqDepth==High, Floored ES Score",
            "SeqDepth==Low, Floored ES Score",
        ]

        nrows = 2
        ncols = 2

    else:
        to_plot = [hm_high_raw, hm_low_raw]

        ax_titles = [
            "SeqDepth==High, Non-floored ES Score",
            "SeqDepth==Low, Non-floored ES Score",
        ]

        nrows = 1
        ncols = 2

    fig, axes = easy_subplots(
        base_figsize=(10, 10), nrows=nrows, ncols=ncols, sharex=True, sharey=True
    )

    plotting_kwargs = {"center": 0.5, "cmap": "coolwarm", "annot": True}

    # Will only evaluate if kwargs is NOT empty
    if kwargs:
        plotting_kwargs.update(kwargs)

    for i, ax in enumerate(axes):
        sns.heatmap(to_plot[i], ax=ax, **plotting_kwargs)
        ax.set_title(ax_titles[i])

    plt.tight_layout()

    return fig


def score_heatmap_expanded(meta, df_es_scores, plot_floored=True, return_ax=False, **kwargs):
    """meta should have index of did"""

    # Get heatmaps across seq_depth low, high and raw, floored scores

    hm_high_raw = (
        df_es_scores.join(meta)
        .query('seq_depth == "high"')
        .groupby(["ground_truth", "noise", "n_timeseries", "sampling_scheme", "n_timepoints"])
        .median()
        .pivot_table(
            index=["noise", "sampling_scheme", "n_timepoints"],
            columns=["ground_truth", "n_timeseries"],
            values="raw",
        )
    )

    hm_low_raw = (
        df_es_scores.join(meta)
        .query('seq_depth == "low"')
        .groupby(["ground_truth", "noise", "n_timeseries", "sampling_scheme", "n_timepoints"])
        .median()
        .pivot_table(
            index=["noise", "sampling_scheme", "n_timepoints"],
            columns=["ground_truth", "n_timeseries"],
            values="raw",
        )
    )

    hm_high_floored = (
        df_es_scores.join(meta)
        .query('seq_depth == "high"')
        .groupby(["ground_truth", "noise", "n_timeseries", "sampling_scheme", "n_timepoints"])
        .median()
        .pivot_table(
            index=["noise", "sampling_scheme", "n_timepoints"],
            columns=["ground_truth", "n_timeseries"],
            values="floored",
        )
    )

    hm_low_floored = (
        df_es_scores.join(meta)
        .query('seq_depth == "low"')
        .groupby(["ground_truth", "noise", "n_timeseries", "sampling_scheme", "n_timepoints"])
        .median()
        .pivot_table(
            index=["noise", "sampling_scheme", "n_timepoints"],
            columns=["ground_truth", "n_timeseries"],
            values="floored",
        )
    )

    if plot_floored:
        to_plot = [hm_high_raw, hm_low_raw, hm_high_floored, hm_low_floored]

        ax_titles = [
            "SeqDepth==High, Non-floored ES Score",
            "SeqDepth==Low, Non-floored ES Score",
            "SeqDepth==High, Floored ES Score",
            "SeqDepth==Low, Floored ES Score",
        ]

        nrows = 2
        ncols = 2

    else:
        to_plot = [hm_high_raw, hm_low_raw]

        ax_titles = [
            "SeqDepth==High, Non-floored ES Score",
            "SeqDepth==Low, Non-floored ES Score",
        ]

        nrows = 1
        ncols = 2

    fig, axes = easy_subplots(
        base_figsize=(10, 10), nrows=nrows, ncols=ncols, sharex=True, sharey=True
    )

    plotting_kwargs = {"center": 0.5, "cmap": "coolwarm", "annot": True}

    # Will only evaluate if kwargs is NOT empty
    if kwargs:
        plotting_kwargs.update(kwargs)

    for i, ax in enumerate(axes):
        sns.heatmap(to_plot[i], ax=ax, **plotting_kwargs)
        ax.set_title(ax_titles[i])

    plt.tight_layout()

    if return_ax:
        return fig, axes
    else:
        return fig


def plot_dataset(did):

    full_df, _, _, meta_spec = mu.load_dataset_by_did(did)

    n_species = meta_spec["n_species"].unique()[0]
    n_timeseries = meta_spec["n_timeseries"].unique()[0]
    timeseries_ids = meta_spec["timeseries_id"].unique()

    species_cols = full_df.columns[full_df.columns.str.contains("species_")]

    if n_timeseries % 2 == 0:
        nrows = int(n_timeseries / 2)
        ncols = 2
    else:
        ncols = 4
        nrows = math.floor(n_timeseries / 3)

    fig, axes = easy_subplots(
        ncols=ncols, nrows=nrows, base_figsize=(3, 3), sharex=True, sharey=True
    )

    for i in range(n_timeseries):
        ax = axes[i]
        cur_timeseries_id = timeseries_ids[i]

        subset = full_df.query("timeseries_id == @cur_timeseries_id")

        cur_X = subset[species_cols].values
        cur_time = subset.iloc[:, 0].values

        [ax.scatter(cur_time, cur_X[:, i]) for i in range(n_species)]
        [ax.plot(cur_time, cur_X[:, i], alpha=0.6) for i in range(n_species)]
        ax.set_title(f"Timeseries ID: {cur_timeseries_id}")

        ax.set_xlabel("time")
        ax.set_ylabel("abundance")

    fig.suptitle(f"MTIST Dataset did={did}")
    plt.tight_layout()
    despine()

    return fig, axes