# Author: Jonas Schluter <jonas.schluter+github@gmail.com>
#        www.ramenlabs.science
#
# License: MIT

import warnings

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate as sp
import seaborn as sns

# PNAS figure guidelines as ref for sizes
#
# 1 column wide (20.5 picas / 3.42” / 8.7cm)
# 1.5 columns wide (27 picas / 4.5” / 11.4cm)
# 2 columns wide (42.125 picas / 7” / 17.8cm)

small_square_fig = (3.42, 3.42)
small_wide_fig = (3.42, 2.5)
small_long_fig = (3.42, 8)
large_wide_fig = (7, 3.5)
font_axes = {
    "family": "Arial",
    "color": "black",
    "weight": "normal",
    "size": 12,
}

font_ticks = {
    "family": "Arial",
    "color": "black",
    "weight": "normal",
    "size": 8,
}
ecosystem = {
    "Antilope": {
        "Antilope": -1,
        "Baboon": 0,
        "Bison": -2,
        "Buffalo": -2,
        "Cheetah": -0.1,
        "Duck": 0,
        "Eagle": 1,
        "Lion": -0.2,
        "Tiger": -0.2,
        "Zebra": -2,
    },
    "Baboon": {
        "Antilope": 0,
        "Baboon": -1,
        "Bison": 0,
        "Buffalo": 0,
        "Cheetah": -0.5,
        "Duck": 0,
        "Eagle": -1,
        "Lion": -0.5,
        "Tiger": -0.5,
        "Zebra": 0,
    },
    "Bison": {
        "Antilope": -0.5,
        "Baboon": 0,
        "Bison": -0.5,
        "Buffalo": -2,
        "Cheetah": -0.5,
        "Duck": 0,
        "Eagle": 1,
        "Lion": -1,
        "Tiger": -1,
        "Zebra": -2,
    },
    "Buffalo": {
        "Antilope": -0.5,
        "Baboon": 0,
        "Bison": -2,
        "Buffalo": -0.5,
        "Cheetah": -0.5,
        "Duck": 0,
        "Eagle": 1,
        "Lion": -1,
        "Tiger": -1,
        "Zebra": -2,
    },
    "Cheetah": {
        "Antilope": 0.5,
        "Baboon": 0.125,
        "Bison": 1.25,
        "Buffalo": 1.5,
        "Cheetah": -2.5,
        "Duck": 0.15,
        "Eagle": 0,
        "Lion": 0,
        "Tiger": 0,
        "Zebra": 1.5,
    },
    "Duck": {
        "Antilope": -0.01,
        "Baboon": 0,
        "Bison": -0.02,
        "Buffalo": -0.02,
        "Cheetah": -0.01,
        "Duck": -5,
        "Eagle": 0,
        "Lion": -0.15,
        "Tiger": -0.15,
        "Zebra": -0.02,
    },
    "Eagle": {
        "Antilope": 0.01,
        "Baboon": 0.5,
        "Bison": 0,
        "Buffalo": 0,
        "Cheetah": 0,
        "Duck": 0,
        "Eagle": -1,
        "Lion": 0,
        "Tiger": 0,
        "Zebra": 0,
    },
    "Lion": {
        "Antilope": 1,
        "Baboon": 0.25,
        "Bison": 2.5,
        "Buffalo": 3,
        "Cheetah": 0,
        "Duck": 0.15,
        "Eagle": 0,
        "Lion": -1,
        "Tiger": -15,
        "Zebra": 3,
    },
    "Tiger": {
        "Antilope": 1,
        "Baboon": 0.25,
        "Bison": 2.5,
        "Buffalo": 3,
        "Cheetah": 0,
        "Duck": 0.15,
        "Eagle": 0,
        "Lion": -15,
        "Tiger": -1,
        "Zebra": 3,
    },
    "Zebra": {
        "Antilope": -0.25,
        "Baboon": 0,
        "Bison": -1,
        "Buffalo": -1,
        "Cheetah": -0.5,
        "Duck": -0.1,
        "Eagle": 0,
        "Lion": -1,
        "Tiger": -1,
        "Zebra": -0.5,
    },
}
growth_rates = {
    "Antilope": 0.85,
    "Baboon": 1,
    "Bison": 0.6,
    "Buffalo": 0.6,
    "Cheetah": -0.75,
    "Duck": 0.3,
    "Eagle": 0.8,
    "Lion": -1.5,
    "Tiger": -1.5,
    "Zebra": 0.65,
}
species_styles = {
    "Antilope": {"c": "#ceb301", "s": "-", "w": 1},
    "Baboon": {"c": "#a00498", "s": "-", "w": 1},
    "Bison": {"c": "#88b378", "s": "-", "w": 3},
    "Buffalo": {"c": "#88b378", "s": "--", "w": 3},
    "Cheetah": {"c": "#d9544d", "s": ":", "w": 2},
    "Duck": {"c": "#a8a495", "s": "-", "w": 1},
    "Eagle": {"c": "#9dbcd4", "s": "-", "w": 1},
    "Lion": {"c": "#ef4026", "s": "-", "w": 3},
    "Tiger": {"c": "#ef4026", "s": "--", "w": 3},
    "Zebra": {"c": "#070d0d", "s": "--", "w": 1},
}


def _lv(**kwargs):
    """Helper function to set up ODE system"""
    params = {
        "_A": np.array([[-1, 1, 1], [-1, -1, -1], [3, -3, -1]]),
        "_mu": np.array([0.1, 2, 0.04]),
        "_noise_scale": 0,
    }
    params.update(kwargs)

    def lv_model_full_system(t, y):
        A = params["_A"]
        mu = params["_mu"]
        dy = np.multiply(np.dot(A, y) + mu, y)
        return dy

    return lv_model_full_system


def run_lv(
    A,
    mu,
    species_names,
    random_seed=0,
    tend=5,
    dt=0.1,
    yinit_specific={},
    noise=0,
    sample_freq=100,
):
    """Solve ODE ecosystem."""
    if sample_freq > 100:
        warnings.warn(
            "Currently noise is added 100 times over the total simulation period. Choosing a sample frequency higher than that is not supported at this point.",
            RuntimeWarning,
        )
    np.random.seed(seed=random_seed)
    # Assign random initial densities
    # update if specific starting densities are chosen
    yinit = np.random.randint(1, 10, len(mu)) / 100
    yinit = dict(zip(sorted(species_names), yinit))
    yinit.update(yinit_specific)
    yinit = [yinit[s] for s in sorted(yinit.keys())]
    first_yinit = yinit
    ### solve
    sol_t_all = np.array([])
    sol_y_all = np.array([])
    sol_t = np.array([])
    sol_y = np.array([])
    # f
    r = sp.ode(_lv(_A=A, _mu=mu)).set_integrator("vode")
    r.set_initial_value(yinit, 0)
    sol_t = np.append(sol_t, 0)
    sol_y = np.append(sol_y, yinit)
    timepoints = np.linspace(0, tend, 100)
    for i, t in enumerate(timepoints[0:-1]):
        tend = timepoints[i + 1]
        r = sp.ode(_lv(_A=A, _mu=mu)).set_integrator("vode")
        r.set_initial_value(yinit, timepoints[i])
        while r.successful() and r.t < tend:
            sol_t_all = np.append(sol_t_all, r.t)
            sol_y_all = np.append(sol_y_all, r.y)
            r.t + dt
            r.integrate(r.t + dt)
        yinit = r.y
        perturbation = [_y * noise * np.random.randn() for _y in yinit]
        yinit = yinit + perturbation
        yinit = [_y if _y >= 2.5e-3 else 0 for _y in yinit]
        sol_t = np.append(sol_t, r.t)
        sol_y = np.append(sol_y, yinit)
    sol_y = np.reshape(sol_y, [-1, len(mu)])
    sol_t = np.reshape(sol_t, [-1, 1])
    sol_y_all = np.reshape(sol_y_all, [-1, len(mu)])
    sol_t_all = np.reshape(sol_t_all, [-1, 1])
    sol_y = sol_y[np.linspace(0, sol_y.shape[0] - 1, sample_freq, dtype=int)]
    sol_t = sol_t[np.linspace(0, sol_t.shape[0] - 1, sample_freq, dtype=int)]
    return (sol_t, sol_y, sol_t_all, sol_y_all, first_yinit)


def construct_ecosystem(species, ecosystem):
    """Return A list of lists (interaction matrix) with the interaction parameters from the ecosystem dictionary.
    Each list contains the ordered interaction parameters
    with all other species for a focal species. The order is defined by alphabetically ordered species.

    Parameters
    ----------
    species : list, species names
    ecosystem : dict, contains interactions for all species

    Returns
    ----------
    A : list of lists, interaction matrix
    """
    A = [[ecosystem[s][ss] for ss in sorted(species)] for s in sorted(species)]
    return A


def plot_lv(t, y, ecosystem, species_styles, y_init, figure=None, axes=None):
    """Plot time series data of ODE simulations.

    Parameters
    ----------
    t: list like, time (x values) for plot
    y: array like, columns are the abundance values for different species over time(rows)
    ecosystem : dict, contains interactions for all species
    species_styles: dict(dicts), define aesthetics of the species
    y_init: array like, initial abundances for each species
    figure: optional, figure to plot to
    axes: optional, [ax, ax] for barchart and time series plot

    Returns
    ----------
    fig, ax, ax: current figure, axes for two plots

    """
    sns.set_style("white")
    if (figure == None) | (axes == None):
        fig = plt.figure()
        gs = gridspec.GridSpec(1, 4)
        ax = plt.subplot(gs[0, 1::])
        ax2 = plt.subplot(gs[0, 0])
        fig.set_size_inches(3, 1)

    else:
        fig = figure
        [ax2, ax] = axes
    ## Time series
    for i, s in enumerate(sorted(ecosystem.keys())):
        assign_species_name = lambda s: s if "random" not in s else "zz_random_sp"
        s = assign_species_name(s)
        if "random" not in s:
            ax.plot(
                t,
                y[:, i],
                linestyle=species_styles[s]["s"],
                linewidth=species_styles[s]["w"],
                color=species_styles[s]["c"],
                label=s,
            )
        else:
            ax.plot(
                t,
                y[:, i],
                alpha=0.2,
                linestyle=species_styles[s]["s"],
                linewidth=species_styles[s]["w"],
                color=species_styles[s]["c"],
                label="random_species",
            )

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(
        [by_label[k] for k in list(sorted(by_label.keys()))],
        [k for k in list(sorted(by_label.keys()))],
        bbox_to_anchor=(1.05, 1),
        loc=2,
        borderaxespad=0.0,
    )
    ax.set_xlabel("Time", fontdict=font_axes)
    ax.set_xlim([0, max(t)])
    ax.set_ylabel("Density", fontdict=font_axes)
    yticks = ax.get_yticks()
    ax.set_yticks([yticks[0], yticks[-1]])
    ax.get_yaxis().set_ticks_position("right")
    ax.spines["top"].set_visible(False)
    ## Ininitial Conditions
    bar_species = [s for s in list(sorted(ecosystem.keys())) if "random" not in s]
    ax2.bar(
        range(0, len(bar_species)),
        y_init[0 : len(bar_species)],
        color=[species_styles[s]["c"] for s in sorted(bar_species)],
    )
    ax2.set_xticks([x + 0.25 for x in range(0, len(bar_species))])
    ax2.set_xticklabels(sorted(bar_species), rotation="vertical", fontdict=font_ticks)
    yticks = ax2.get_yticks()
    ax2.set_yticks([yticks[0], yticks[-1]])
    return (fig, ax, ax2)


def simple_plot_lv(t, y, ecosystem, species_styles, y_init):
    """Plot time series data of ODE simulations."""
    sns.set_style("white")
    fig, ax1 = plt.subplots()
    gs = gridspec.GridSpec(1, 3)
    ## Time series
    ax = plt.subplot(gs[0, 0::])

    for i, s in enumerate(sorted(ecosystem.keys())):
        assign_species_name = lambda s: s if "random" not in s else "zz_random_sp"
        s = assign_species_name(s)
        if "random" not in s:
            ax.plot(
                t,
                y[:, i],
                linestyle=species_styles[s]["s"],
                linewidth=species_styles[s]["w"],
                color=species_styles[s]["c"],
                label=s,
            )
        else:
            ax.plot(
                t,
                y[:, i],
                alpha=0.1,
                linestyle=species_styles[s]["s"],
                linewidth=species_styles[s]["w"],
                color=species_styles[s]["c"],
                label="random_species",
            )

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(
        [by_label[k] for k in list(sorted(by_label.keys()))],
        [k for k in list(sorted(by_label.keys()))],
        bbox_to_anchor=(1.05, 1),
        loc=2,
        borderaxespad=0.0,
    )
    ax.set_xlabel("Time", fontdict=font_axes)
    ax.set_xlim([0, max(t)])
    ax.set_ylabel("Density", fontdict=font_axes)
    ax.set_yticks([])
    ## Ininitial Conditions
    return (fig, ax)
