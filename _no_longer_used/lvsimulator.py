# Author: Jonas Schluter <jonas.schluter+github@gmail.com>
#         http://www.ramenlabs.science
#
# License: MIT
import matplotlib
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt

import mtist.lvutils as util

import numpy as np
import pandas as pd
import scipy.integrate as sp
import seaborn as sns


class LV(object):
    """Lotka Volterra species system.

    Parameters
    ----------
    ecosystem : dict, optional
        Dictionary of dictionaries describing species interactions,
        e.g. {'Tiger': {'Tiger': -1, 'Buffalo': 2},
              'Buffalo': {'Tiger': -1, 'Buffalo': -1}}.

    growth_rates : dict, optional
        Dictionary describing max. exponential growth rates of species,
        e.g. {'Tiger': -1, 'Buffalo': 2}

    species_style : dict, optional
        Dictionary describing plotting styles per species
        (color, linestyle, width),
        e.g. 'Tiger':    {'c':'#ef4026', 's':'--', 'w':3}
    Attributes
    ----------
    A : list, species interaction matrix
    species_names : list

    Notes
    --------

    --------
    """

    def __init__(
        self,
        ecosystem=util.ecosystem,
        growth_rates=util.growth_rates,
        species_styles=util.species_styles,
    ):
        # TODO: Refactor all code to make 'ecosystem' private and use get / set.
        self.ecosystem = ecosystem
        self.growth_rates = growth_rates
        self.species_style = species_styles
        self.species_names = list(sorted(self.ecosystem.keys()))
        self.A = util.construct_ecosystem(self.species_names, self.ecosystem)

    # def __call__(self, *args, **kwargs):
    #     print("Setting up simulator for Lotka Volterra dynamics with ecosystem: \n")
    #     print(self.ecosystem)

    def add_n_random_species(self, n, random_seed=0, interaction_threshold=None):
        """
        Add species to the ecosystem. Newyly added species will interact rarely and
        relatively weakly with other species, and will have -1 for interaction with self.

        Parameters
        ----------
        n : Int, number of species to be added
        random_seed : Int
        """
        ecosystem = self.ecosystem
        growth_rates = self.growth_rates
        species_style = self.species_style
        np.random.seed(random_seed)
        all_interactions = util.construct_ecosystem(ecosystem.keys(), ecosystem)
        mean_interactions_magnitude = np.mean(np.mean(np.abs(all_interactions)))
        if not interaction_threshold is None:
            iat = interaction_threshold
        else:
            iat = 0.7
        iA = lambda x: x if np.abs(x) < iat * mean_interactions_magnitude else 0
        for i in range(0, n):
            ecosystem["zz_random_sp" + str(i)] = {}
            growth_rates["zz_random_sp" + str(i)] = np.random.randn() * 0.25
        curr_species = ecosystem.keys()
        for i in range(0, n):
            for k in list(curr_species):
                ecosystem["zz_random_sp" + str(i)][k] = iA(np.random.randn())
                ecosystem[k]["zz_random_sp" + str(i)] = iA(np.random.randn())
            ecosystem["zz_random_sp" + str(i)]["zz_random_sp" + str(i)] = -1
        species_style["zz_random_sp"] = {"c": "#03719c", "s": "-", "w": 0.75}
        self.ecosystem = ecosystem
        self.growth_rates = growth_rates
        self.species_style = species_style
        self.species_names = list(sorted(ecosystem.keys()))
        self.A = util.construct_ecosystem(self.species_names, self.ecosystem)

    def set_ecosystem(self, newecosystem, newgrowthrates):
        self.ecosystem = newecosystem
        self.growth_rates = newgrowthrates
        self.species_names = list(sorted(newecosystem.keys()))
        self.A = util.construct_ecosystem(self.species_names, self.ecosystem)

    def remove_random_species(self):
        """removes all the 'random' species which interacted weakly with the core ecosystem."""
        core_eco = self.ecosystem
        drop_rand = lambda d: dict([(k, d[k]) for k in d.keys() if "random" not in k])
        self.ecosystem = dict(
            [(k, drop_rand(core_eco[k])) for k in core_eco.keys() if "zz" not in k]
        )
        self.growth_rates = drop_rand(self.growth_rates)
        self.species_style = drop_rand(self.species_style)
        self.species_names = list(sorted(self.ecosystem.keys()))
        self.A = util.construct_ecosystem(self.species_names, self.ecosystem)

    def get_ecosystem_as_df(self):
        """Returns a pandas data frame of the ecosystem."""
        n_species = len(self.species_names)
        A_full = pd.DataFrame(
            np.zeros((n_species, n_species + 1)),
            index=self.species_names,
            columns=["mu"] + self.species_names,
        )
        for i, focalspecies in enumerate(self.species_names):
            A_full.loc[focalspecies][:] = np.append(self.growth_rates[focalspecies], self.A[i])
        return A_full

    def run_lv(self, random_seed=0, tend=5, dt=0.1, yinit_specific={}, noise=0, sample_freq=50):
        """Simulate ecosystem dynamics over time using the VODE solver.

        Parameters
        ----------
        random_seed : Int, optional
        tend : Float, optional; end time of simulation
        dt : Float, optional; time increments for solver output
        yinit_specific : dict, optional; initial species abundance dictionary.
        noise : float, optional; 'noise' may be added to species abundances. A sample from a Normal distribution with zero mean and 'noise' as standard deviation is scaled by the current abundance of a focal species, and added to the current abundance creating noise proportional to the abundance of a species. As this can render abundances negative, negative abundances are set to zero, and the ODE system is calculated from this new state.
        sample_freq : Int, optional; Number of times community states are sampled. Doubles also as the frequency
                      at which noise is added to the system.
        Returns
        ----------
        sol_t : time points at which species abundances in the ecosystem were sampled
        sol_y : species abundances
        sol_t_all : timepoints of intermediate ODE solutions
        sol_y_all : intermediate ODE state variables
        first_y_init : initial conditions
        """
        A = self.A
        mu = [self.growth_rates[s] for s in sorted(self.growth_rates.keys())]
        species_names = self.species_names
        sol_t, sol_y, sol_t_all, sol_y_all, first_yinit = util.run_lv(
            A,
            mu,
            species_names,
            random_seed=random_seed,
            tend=tend,
            dt=dt,
            yinit_specific=yinit_specific,
            noise=noise,
            sample_freq=sample_freq,
        )
        return (sol_t, sol_y, sol_t_all, sol_y_all, first_yinit)

    def plot_lv(self, t=None, y=None, yinit=[], savepath=None, figure=None, axes=None):
        """Plot LV timeseries and initial conditions.

        Parameters
        ----------
        t : array, time points
        y : array, species abundances

        Returns
        ----------
        fig, ax1, ax2 : matplotlib.figure, ax handles

        """
        if len(yinit) == 0:
            t, y, sol_t_all, sol_y_all, yinit = self.run_lv()
        fig, ax1, ax2 = util.plot_lv(
            t, y, self.ecosystem, self.species_style, yinit, figure=figure, axes=axes
        )
        if savepath:
            fig.savefig(savepath, bbox_inches="tight", dpi=300)
        return (fig, ax1, ax2)

    def simple_plot_lv(self, t=None, y=None, yinit=[], savepath=None):
        """Plot LV timeseries and initial conditions.

        Parameters
        ----------
        t : array, time points
        y : array, species abundances

        Returns
        ----------
        fig, ax1, ax2 : matplotlib.figure, ax handles

        """
        if len(yinit) == 0:
            t, y, sol_t_all, sol_y_all, yinit = self.run_lv()
        fig, ax1 = util.simple_plot_lv(t, y, self.ecosystem, self.species_style, yinit)
        if savepath:
            fig.savefig(savepath, bbox_inches="tight", dpi=300)
        return (fig, ax1)

    def plot_ecosystem(
        self,
        species_to_plot=None,
        savepath=None,
        rename_species=False,
        annotate=False,
        format="png",
    ):
        """Plot a simple heatmap showing between-species interactions."""
        A_true = self.get_ecosystem_as_df()
        if rename_species:
            if type(rename_species) == dict:
                rename_dict = rename_species
            else:
                rename_dict = dict(
                    [(z, "rs" + z[-4::]) if "zz" in z else ("mu", "$\mu$") for z in A_true.columns]
                )
                A_true = A_true.rename(rename_dict, columns=rename_dict)
        if species_to_plot:
            A_true = A_true[["mu"] + species_to_plot]
        fig, ax = plt.subplots()
        gs = gridspec.GridSpec(1, 15)
        ## Time series
        axl = plt.subplot(gs[0, 14])
        axp = plt.subplot(gs[0, 0:14])
        sns.heatmap(
            A_true.rename(columns={"mu": "$\mu$"}),
            vmax=3,
            vmin=-6,
            annot=annotate,
            annot_kws={"size": 10},
            ax=axp,
            cbar_ax=axl,
            cbar_kws={"orientation": "vertical"},
        )
        plt.xticks(rotation=90)
        plt.yticks(rotation=0)
        fig.set_size_inches(5, 5)
        axp.set_title("Interaction matrix", fontdict=util.font_axes)
        axl.set_yticks([-6, -5, -2, 0, 2, 4, 6])
        axl.set_yticklabels(["-15", "-8", "-2", "0", "2", "4", "6"], fontdict=util.font_ticks)
        # cbar = axl.collections[0].colorbar
        # cbar.set_ticks([-6,-5, 0,  3])
        # cbar.set_ticklabels([-15,-5,  0,  3])
        if savepath:
            if (format == "eps") | (format == "pdf"):
                matplotlib.rcParams["pdf.fonttype"] = 42
                matplotlib.rcParams["ps.fonttype"] = 42
            fig.savefig(savepath, bbox_inches="tight", format=format, dpi=300)
