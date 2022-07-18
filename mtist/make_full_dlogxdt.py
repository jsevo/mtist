# Author: Jonas Schluter <jonas.schluter+github@gmail.com>
#        www.ramenlabs.science
#
# License: MIT

import pandas as pd
import numpy as np

from scipy import stats


def make_full_dlogxdt(df, key, timekey, columns, specieskey="species"):
    """Calculate the linear system for LV regression.
    Parameters
    ----------
    df : pandas.DataFrame, time series data of species abundances. Rows represent
         individual samples. Columns should have a time point (time key column), and
         an identifier for the experiment (if for example experiments were repeated and thus
         many time series were sampled).
    key : identifier of a single timeseries
    time key : identifier for the column containing time info
    columns : array to indicate columns of species to be investigated.
    Returns
    ----------
    full_dlogxdt : pandas.DataFrame of interval information,
                   contains a column with the loged differences of species abundances and geometric means for
                   abundances between timepoints.
    """
    full_dlogxdt = pd.DataFrame()
    for i, (v, g) in enumerate(df.groupby(key)):
        gsorted = g.sort_values(timekey)
        dt = gsorted[timekey].diff()
        # calculate the time discrete log differences
        dlogxdt = gsorted[columns].apply(np.log).diff().apply(lambda x: np.divide(x, dt)).T.stack()
        # get the geommetric means of species abundances between timepoints
        geomm = (
            gsorted[columns]
            .rolling(center=False, window=2)
            .apply(lambda x: stats.mstats.gmean(x))
            .T.stack()
        )
        # combine
        tmp = dlogxdt.T.swaplevel(0, 1)
        tmp = (
            pd.DataFrame(tmp)
            .reset_index()
            .rename(columns={"level_0": "timeinterval", "level_1": specieskey, 0: "dlogxdt"})
        )
        tmp[key] = np.repeat(v, len(tmp))
        geomm = geomm.unstack().T.reset_index().rename(columns={"index": "timeinterval"})
        _dlogxdt = pd.merge(tmp, geomm, how="left", on="timeinterval")
        # append
        full_dlogxdt = full_dlogxdt.append(_dlogxdt, ignore_index=True)
    full_dlogxdt = full_dlogxdt.replace([np.inf, -np.inf], np.nan).dropna(how="any")
    return full_dlogxdt
