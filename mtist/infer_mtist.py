import os

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import xarray as xr

# from mtist.mtist_utils import mu.GLOBALS, mu.load_dataset, mu.load_ground_truths, mu.calculate_n_datasets
from mtist import mtist_utils as mu


def calc_dlogydt(x, times):
    """x: an n_timepoints long column vector, from one timeseries, of one species
    times: an n_timepoints long column vector of corresponding timepoints

    returns: (dlogydts, dts, times, valid_idxs_list)
    """

    n_intervals = len(x) - 1

    dlogydts = []
    dts = []
    valid_idxs = set()

    for i in range(n_intervals):

        # Check if valid interval
        validity_check = x[i] > 0 and x[i + 1] > 0
        if validity_check:

            # Calculate dt, dlogydt
            dt = times[i + 1] - times[i]
            dts.append(dt)

            dlogydt = (np.log(x[i + 1]) - np.log(x[i])) / dt
            dlogydts.append(dlogydt)

            # Save valid indices
            valid_idxs.update([i, i + 1])

        else:
            pass

    # Return empty arrays if all else fails
    if len(dlogydts) <= 0:
        # Return empty arrays
        return (np.array([]), np.array([]), np.array([]), np.array([], dtype=bool))
    else:

        valid_idxs_list = list(valid_idxs)
        times = times[valid_idxs_list].copy()
        times = np.array(times)
        dlogydts = np.array(dlogydts)
        dts = np.array(dts)

        return (dlogydts, dts, times, valid_idxs_list)


def prepare_data_for_inference(did):

    # `full_df`: is the full pd.DataFrame from cvs file
    # `full_time_column`: is time column (ndarray nrows x 1)
    # `X`: is JUST the abundances (ndarray nrows x n_species)
    # `meta_spec`: is all of the metadata contained in `full_df`
    path_to_did_dataset = os.path.join(mu.GLOBALS.MTIST_DATASET_DIR, f"dataset_{did}.csv")
    full_df, full_time_column, X, meta_spec = mu.load_dataset(path_to_did_dataset)

    # Premise:
    # We must calclate, for every time series (delineated in
    # the dataset by `timeseries_id), a dlogydt matrix.

    # We must save this with a corresponding `time` matrix.

    # Reason: One dataset contains multple time series, and
    # we must keep those time series separate to pool information
    # across them properly.

    # Get unique codes for all `timeseries_id`s
    codes = meta_spec["timeseries_id"].astype("category").cat.codes.values

    # NOW: Loop over each `timeseries_id`, each species and calculate proper
    # dlogydt and time matrices.

    _, n_species = X.shape

    # These subsequent collectors will be indexed by
    # the `timeseries_id` code
    dlogydt = {}
    dt = {}
    times = {}
    nz_masks = {}
    gmeans = {}

    # FOR EVERY TIMESERIES INDIVIDUALLY
    for i_code in range(len(np.unique(codes))):

        # Get individual timeseries
        mask = codes == i_code
        cur_timeseries = X[mask].copy()

        # Calculate number of (t, t+1) intervals
        cur_n_intervals = cur_timeseries.shape[0] - 1

        # Create lists to hold calculated quantities PER SPECIES
        dlogydt[i_code] = []
        dt[i_code] = []  # the numerical time between each interval
        times[i_code] = []  # times matrix for each
        nz_masks[i_code] = []  # non-zero masks
        gmeans[i_code] = []  # geometric means

        # FOR EACH SPECIES WITHIN A SINGLE TIME SERIES IN `i_code`
        for i_species in range(n_species):

            cur_species = cur_timeseries[:, i_species]

            # Calculate dlogydt, dt, times#
            x, y, z, _ = calc_dlogydt(cur_species, full_time_column[mask])

            dlogydt[i_code].append(x)
            dt[i_code].append(y)  # never used again
            times[i_code].append(z)

            # Calculate geom_mean for each individal interval
            # (pair of timepoints). TODO: This could be refactored.
            gmeans_tmp = np.ones(cur_n_intervals)
            valid_int_idx_tmp = np.ones(cur_n_intervals, dtype=bool)
            for j in range(cur_n_intervals):
                cur_gmean = np.sqrt(cur_species[j] * cur_species[j + 1])
                gmeans_tmp[j] = cur_gmean

                if cur_species[j] <= 0 or cur_species[j + 1] <= 0:
                    valid_int_idx_tmp[j] = False

            # Save the gmeans, nz_masks
            gmeans[i_code].append(gmeans_tmp)
            nz_masks[i_code].append(valid_int_idx_tmp)

    # Turn into dfs because that's easy for me
    cols = pd.Index(range(len(np.unique(codes))), name="timeseries_id")
    idx = pd.Index(range(X.shape[1]), name="species_id")

    # df_times = pd.DataFrame(times, columns=cols, index=idx)
    df_dlogydt = pd.DataFrame(dlogydt, columns=cols, index=idx)
    df_nzmask = pd.DataFrame(nz_masks, columns=cols, index=idx, dtype=bool)
    df_geom = pd.DataFrame(gmeans, columns=cols, index=idx)

    return df_geom, df_dlogydt, df_nzmask, n_species


def infer_from_did(did, debug=False):
    """Returns  `inferred` tuple (interaction_coefficients, growth_rates)

    Will return info as well if debug=True"""

    df_geom, df_dlogydt, df_nzmask, n_species = prepare_data_for_inference(did)

    ####### BEGIN THE INFERENCE!!!!! #######
    regs = []
    intercepts = []
    slopes = []

    # For debugging if needed
    info = dict(dlogydts=[], masks=[], gmeans=[], species=[], shapes=[])

    # Begin inference for each and every focal_species
    for focal_species in range(n_species):

        # Get the y to be predicted
        cur_dlogydt = np.concatenate(df_dlogydt.loc[focal_species].values)
        cur_mask = np.concatenate(df_nzmask.loc[focal_species].values)

        # Get the X to predict, only take valid intervals
        cur_gmeans = np.array(
            [np.concatenate(df_geom.loc[i, :].values) for i in range(n_species)]
        ).T

        cur_gmeans = cur_gmeans[cur_mask, :].copy()

        # Update debug info
        info["dlogydts"].append(cur_dlogydt)
        info["masks"].append(cur_mask)
        info["gmeans"].append(cur_gmeans)
        info["species"].append(focal_species)
        info["shapes"].append(
            dict(dlogydt=cur_dlogydt.shape, mask=cur_mask.shape, gmeans=cur_gmeans.shape)
        )

        # If focal_species has no intervals, return NaNs for inferred.
        if len(cur_dlogydt) <= 1:
            regs.append(np.nan)
            slopes.append(np.repeat(np.nan, n_species))
            intercepts.append(np.array([np.nan]))

        # Otherwise, regress.
        else:
            try:
                reg = LinearRegression().fit(cur_gmeans, cur_dlogydt)
            except ValueError:
                return ("broken", did, info)

            regs.append(reg)
            intercepts.append(reg.intercept_)
            slopes.append(reg.coef_)

    # Return all solutions!

    # make em arrays
    slopes = np.vstack(slopes)
    intercepts = np.vstack(intercepts)

    inferred = (slopes, intercepts)

    if debug:
        return (inferred, info)
    else:
        return inferred


def infer_from_did_ols_with_p(did, debug=False, save=False, th=None):
    """Returns  `inferred` tuple (interaction_coefficients, growth_rates)
    Will return info as well if debug=True"""

    if th is None:
        th = 0.001

    df_geom, df_dlogydt, df_nzmask, n_species = prepare_data_for_inference(did)

    ####### BEGIN THE INFERENCE!!!!! #######
    regs = []
    intercepts = []
    slopes = []
    pvals = []

    # For debugging if needed
    info = dict(dlogydts=[], masks=[], gmeans=[], species=[], shapes=[])

    # Begin inference for each and every focal_species
    for focal_species in range(n_species):

        # Get the y to be predicted
        cur_dlogydt = np.concatenate(df_dlogydt.loc[focal_species].values)
        cur_mask = np.concatenate(df_nzmask.loc[focal_species].values)

        # Get the X to predict, only take valid intervals
        cur_gmeans = np.array(
            [np.concatenate(df_geom.loc[i, :].values) for i in range(n_species)]
        ).T

        cur_gmeans = cur_gmeans[cur_mask, :].copy()

        # Update debug info
        info["dlogydts"].append(cur_dlogydt)
        info["masks"].append(cur_mask)
        info["gmeans"].append(cur_gmeans)
        info["species"].append(focal_species)
        info["shapes"].append(
            dict(dlogydt=cur_dlogydt.shape, mask=cur_mask.shape, gmeans=cur_gmeans.shape)
        )

        # If focal_species has no intervals, or a single interval, return NaNs for inferred.
        if len(cur_dlogydt) <= 1:
            regs.append(np.nan)
            slopes.append(np.repeat(np.nan, n_species))
            intercepts.append(np.array([np.nan]))
            pvals.append(np.repeat(np.nan, n_species))

        # Otherwise, regress.
        else:
            try:
                cur_gmeans_with_intercept = sm.add_constant(cur_gmeans)
                model = sm.OLS(cur_dlogydt, cur_gmeans_with_intercept)
                reg = model.fit()

                cur_intercept = reg.params[0]
                cur_slopes = reg.params[1::]  # only for coefs
                cur_pvals = reg.pvalues[1::]  # only for coefs
            except ValueError:
                return ("broken", did, info)

            regs.append(reg)
            intercepts.append(cur_intercept)
            slopes.append(cur_slopes)
            pvals.append(cur_pvals)

    # Return all solutions!

    # make em arrays
    slopes = np.vstack(slopes)
    intercepts = np.vstack(intercepts)
    pvals = np.vstack(pvals)

    # Now set the pvals below a threshold, here hard-coded as 0.05, to 0.

    # print(slopes)
    # print(pvals)
    slopes = pd.DataFrame(slopes)
    pvals = pd.DataFrame(pvals)

    # keep only slopes that are "significant"
    slopes = slopes[pvals < th].fillna(0)
    slopes = slopes.values

    # print(slopes)

    inferred = (slopes, intercepts)

    if save:
        try:
            os.mkdir(
                os.path.join(
                    mu.GLOBALS.MTIST_DATASET_DIR,
                    f"ols_pvals",
                )
            )
        except Exception as e:
            print(e)

        np.savetxt(
            os.path.join(
                mu.GLOBALS.MTIST_DATASET_DIR,
                f"ols_pvals",
                f"ols_pvals_{did}.csv",
            ),
            pvals,
            delimiter=",",
        )

    if debug:
        return (inferred, info)
    else:
        return inferred


def infer_from_did_lasso(did, debug=False):
    """Returns  `inferred` tuple (interaction_coefficients, growth_rates)

    Will return info as well if debug=True"""

    df_geom, df_dlogydt, df_nzmask, n_species = prepare_data_for_inference(did)

    ####### BEGIN THE INFERENCE!!!!! #######
    regs = []
    intercepts = []
    slopes = []

    # For debugging if needed
    info = dict(dlogydts=[], masks=[], gmeans=[], species=[], shapes=[])

    # Begin inference for each and every focal_species
    for focal_species in range(n_species):

        # Get the y to be predicted
        cur_dlogydt = np.concatenate(df_dlogydt.loc[focal_species].values)
        cur_mask = np.concatenate(df_nzmask.loc[focal_species].values)

        # Get the X to predict, only take valid intervals
        cur_gmeans = np.array(
            [np.concatenate(df_geom.loc[i, :].values) for i in range(n_species)]
        ).T

        cur_gmeans = cur_gmeans[cur_mask, :].copy()

        # Update debug info
        info["dlogydts"].append(cur_dlogydt)
        info["masks"].append(cur_mask)
        info["gmeans"].append(cur_gmeans)
        info["species"].append(focal_species)
        info["shapes"].append(
            dict(dlogydt=cur_dlogydt.shape, mask=cur_mask.shape, gmeans=cur_gmeans.shape)
        )

        # If focal_species has no intervals, return NaNs for inferred.
        if len(cur_dlogydt) <= 1:
            regs.append(np.nan)
            slopes.append(np.repeat(np.nan, n_species))
            intercepts.append(np.array([np.nan]))

        # Otherwise, regress.
        else:
            try:
                reg = linear_model.Lasso().fit(cur_gmeans, cur_dlogydt)
            except ValueError:
                return ("broken", did, info)

            regs.append(reg)
            intercepts.append(reg.intercept_)
            slopes.append(reg.coef_)

    # Return all solutions!

    # make em arrays
    slopes = np.vstack(slopes)
    intercepts = np.vstack(intercepts)

    inferred = (slopes, intercepts)

    if debug:
        return (inferred, info)
    else:
        return inferred


def infer_from_did_lasso_cv(did, debug=False):
    """Returns  `inferred` tuple (interaction_coefficients, growth_rates)

    Will return info as well if debug=True"""

    df_geom, df_dlogydt, df_nzmask, n_species = prepare_data_for_inference(did)

    regs = []
    intercepts = []
    slopes = []

    # For debugging if needed
    info = dict(dlogydts=[], masks=[], gmeans=[], species=[], shapes=[])

    # Begin inference for each and every focal_species
    for focal_species in range(n_species):

        # Get the y to be predicted
        cur_dlogydt = np.concatenate(df_dlogydt.loc[focal_species].values)
        cur_mask = np.concatenate(df_nzmask.loc[focal_species].values)

        # Get the X to predict, only take valid intervals
        cur_gmeans = np.array(
            [np.concatenate(df_geom.loc[i, :].values) for i in range(n_species)]
        ).T

        cur_gmeans = cur_gmeans[cur_mask, :].copy()

        # Update debug info
        info["dlogydts"].append(cur_dlogydt)
        info["masks"].append(cur_mask)
        info["gmeans"].append(cur_gmeans)
        info["species"].append(focal_species)
        info["shapes"].append(
            dict(dlogydt=cur_dlogydt.shape, mask=cur_mask.shape, gmeans=cur_gmeans.shape)
        )

        # If focal_species has no intervals, return NaNs for inferred.
        # if len(cur_dlogydt) <= 1:
        # regs.append(np.nan)
        # slopes.append(np.repeat(np.nan, n_species))
        # intercepts.append(np.array([np.nan]))

        # Else if we have at least 10 points (required for train-test split), run ElasticNetCV.
        if len(cur_dlogydt) >= 10:

            X_train, X_val, y_train, y_val = train_test_split(
                cur_gmeans, cur_dlogydt, train_size=0.8, random_state=0
            )

            lassocv = linear_model.LassoCV(
                cv=5,
                fit_intercept=True,
                max_iter=10 ** 7,
            )

            reg = lassocv.fit(X_train, y_train)
            r2 = lassocv.score(X_val, y_val)

            regs.append(reg)
            intercepts.append(reg.intercept_)
            slopes.append(reg.coef_)

        # With sample points between 1 and 10, run a simple linear regression.
        else:
            # reg = linear_model.LinearRegression().fit(cur_gmeans, cur_dlogydt)

            # regs.append(reg)
            # intercepts.append(reg.intercept_)
            # slopes.append(reg.coef_)

            regs.append(np.nan)
            slopes.append(np.repeat(np.nan, n_species))
            intercepts.append(np.array([np.nan]))

    # make em arrays
    slopes = np.vstack(slopes)
    intercepts = np.vstack(intercepts)

    inferred = (slopes, intercepts)

    if debug:
        return (inferred, info)
    else:
        return inferred


def infer_from_did_ridge(did, debug=False):
    """Returns  `inferred` tuple (interaction_coefficients, growth_rates)

    Will return info as well if debug=True"""

    df_geom, df_dlogydt, df_nzmask, n_species = prepare_data_for_inference(did)

    ####### BEGIN THE INFERENCE!!!!! #######
    regs = []
    intercepts = []
    slopes = []

    # For debugging if needed
    info = dict(dlogydts=[], masks=[], gmeans=[], species=[], shapes=[])

    # Begin inference for each and every focal_species
    for focal_species in range(n_species):

        # Get the y to be predicted
        cur_dlogydt = np.concatenate(df_dlogydt.loc[focal_species].values)
        cur_mask = np.concatenate(df_nzmask.loc[focal_species].values)

        # Get the X to predict, only take valid intervals
        cur_gmeans = np.array(
            [np.concatenate(df_geom.loc[i, :].values) for i in range(n_species)]
        ).T

        cur_gmeans = cur_gmeans[cur_mask, :].copy()

        # Update debug info
        info["dlogydts"].append(cur_dlogydt)
        info["masks"].append(cur_mask)
        info["gmeans"].append(cur_gmeans)
        info["species"].append(focal_species)
        info["shapes"].append(
            dict(dlogydt=cur_dlogydt.shape, mask=cur_mask.shape, gmeans=cur_gmeans.shape)
        )

        # If focal_species has no intervals, return NaNs for inferred.
        if len(cur_dlogydt) <= 1:
            regs.append(np.nan)
            slopes.append(np.repeat(np.nan, n_species))
            intercepts.append(np.array([np.nan]))

        # Otherwise, regress.
        else:
            try:
                reg = linear_model.Ridge().fit(cur_gmeans, cur_dlogydt)
            except ValueError:
                return ("broken", did, info)

            regs.append(reg)
            intercepts.append(reg.intercept_)
            slopes.append(reg.coef_)

    # Return all solutions!

    # make em arrays
    slopes = np.vstack(slopes)
    intercepts = np.vstack(intercepts)

    inferred = (slopes, intercepts)

    if debug:
        return (inferred, info)
    else:
        return inferred


def infer_from_did_ridge_cv(did, debug=False):
    """Returns  `inferred` tuple (interaction_coefficients, growth_rates)

    Will return info as well if debug=True"""

    df_geom, df_dlogydt, df_nzmask, n_species = prepare_data_for_inference(did)

    regs = []
    intercepts = []
    slopes = []

    # For debugging if needed
    info = dict(dlogydts=[], masks=[], gmeans=[], species=[], shapes=[])

    # Begin inference for each and every focal_species
    for focal_species in range(n_species):

        # Get the y to be predicted
        cur_dlogydt = np.concatenate(df_dlogydt.loc[focal_species].values)
        cur_mask = np.concatenate(df_nzmask.loc[focal_species].values)

        # Get the X to predict, only take valid intervals
        cur_gmeans = np.array(
            [np.concatenate(df_geom.loc[i, :].values) for i in range(n_species)]
        ).T

        cur_gmeans = cur_gmeans[cur_mask, :].copy()

        # Update debug info
        info["dlogydts"].append(cur_dlogydt)
        info["masks"].append(cur_mask)
        info["gmeans"].append(cur_gmeans)
        info["species"].append(focal_species)
        info["shapes"].append(
            dict(dlogydt=cur_dlogydt.shape, mask=cur_mask.shape, gmeans=cur_gmeans.shape)
        )

        # If focal_species has no intervals, return NaNs for inferred.
        # if len(cur_dlogydt) <= 1:
        # regs.append(np.nan)
        # slopes.append(np.repeat(np.nan, n_species))
        # intercepts.append(np.array([np.nan]))

        # Else if we have at least 10 points (required for train-test split), run ElasticNetCV.
        if len(cur_dlogydt) >= 10:

            X_train, X_val, y_train, y_val = train_test_split(
                cur_gmeans, cur_dlogydt, train_size=0.8, random_state=0
            )

            ridgecv = linear_model.RidgeCV(
                cv=5,
                fit_intercept=True,
            )

            reg = ridgecv.fit(X_train, y_train)
            r2 = ridgecv.score(X_val, y_val)

            regs.append(reg)
            intercepts.append(reg.intercept_)
            slopes.append(reg.coef_)

        # With sample points between 1 and 10, run a simple linear regression.
        else:
            # reg = linear_model.LinearRegression().fit(cur_gmeans, cur_dlogydt)
            #
            # regs.append(reg)
            # intercepts.append(reg.intercept_)
            # slopes.append(reg.coef_)

            regs.append(np.nan)
            slopes.append(np.repeat(np.nan, n_species))
            intercepts.append(np.array([np.nan]))

    # make em arrays
    slopes = np.vstack(slopes)
    intercepts = np.vstack(intercepts)

    inferred = (slopes, intercepts)

    if debug:
        return (inferred, info)
    else:
        return inferred


def infer_from_did_elasticnet(did, debug=False):
    """Returns  `inferred` tuple (interaction_coefficients, growth_rates)

    Will return info as well if debug=True"""
    df_geom, df_dlogydt, df_nzmask, n_species = prepare_data_for_inference(did)

    ####### BEGIN THE INFERENCE!!!!! #######
    regs = []
    intercepts = []
    slopes = []

    # For debugging if needed
    info = dict(dlogydts=[], masks=[], gmeans=[], species=[], shapes=[])

    # Begin inference for each and every focal_species
    for focal_species in range(n_species):

        # Get the y to be predicted
        cur_dlogydt = np.concatenate(df_dlogydt.loc[focal_species].values)
        cur_mask = np.concatenate(df_nzmask.loc[focal_species].values)

        # Get the X to predict, only take valid intervals
        cur_gmeans = np.array(
            [np.concatenate(df_geom.loc[i, :].values) for i in range(n_species)]
        ).T

        cur_gmeans = cur_gmeans[cur_mask, :].copy()

        # Update debug info
        info["dlogydts"].append(cur_dlogydt)
        info["masks"].append(cur_mask)
        info["gmeans"].append(cur_gmeans)
        info["species"].append(focal_species)
        info["shapes"].append(
            dict(dlogydt=cur_dlogydt.shape, mask=cur_mask.shape, gmeans=cur_gmeans.shape)
        )

        # If focal_species has no intervals, return NaNs for inferred.
        if len(cur_dlogydt) <= 1:
            regs.append(np.nan)
            slopes.append(np.repeat(np.nan, n_species))
            intercepts.append(np.array([np.nan]))

        # Otherwise, regress.
        else:
            try:
                reg = linear_model.ElasticNet().fit(cur_gmeans, cur_dlogydt)
            except ValueError:
                return ("broken", did, info)

            regs.append(reg)
            intercepts.append(reg.intercept_)
            slopes.append(reg.coef_)

    # Return all solutions!

    # make em arrays
    slopes = np.vstack(slopes)
    intercepts = np.vstack(intercepts)

    inferred = (slopes, intercepts)

    if debug:
        return (inferred, info)
    else:
        return inferred


def infer_from_did_elasticnet_cv(did, debug=False):
    """Returns  `inferred` tuple (interaction_coefficients, growth_rates)

    Will return info as well if debug=True"""

    df_geom, df_dlogydt, df_nzmask, n_species = prepare_data_for_inference(did)

    regs = []
    intercepts = []
    slopes = []

    # For debugging if needed
    info = dict(dlogydts=[], masks=[], gmeans=[], species=[], shapes=[])

    # Begin inference for each and every focal_species
    for focal_species in range(n_species):

        # Get the y to be predicted
        cur_dlogydt = np.concatenate(df_dlogydt.loc[focal_species].values)
        cur_mask = np.concatenate(df_nzmask.loc[focal_species].values)

        # Get the X to predict, only take valid intervals
        cur_gmeans = np.array(
            [np.concatenate(df_geom.loc[i, :].values) for i in range(n_species)]
        ).T

        cur_gmeans = cur_gmeans[cur_mask, :].copy()

        # Update debug info
        info["dlogydts"].append(cur_dlogydt)
        info["masks"].append(cur_mask)
        info["gmeans"].append(cur_gmeans)
        info["species"].append(focal_species)
        info["shapes"].append(
            dict(dlogydt=cur_dlogydt.shape, mask=cur_mask.shape, gmeans=cur_gmeans.shape)
        )

        # If focal_species has no intervals, return NaNs for inferred.
        # if len(cur_dlogydt) <= 1:
        # regs.append(np.nan)
        # slopes.append(np.repeat(np.nan, n_species))
        # intercepts.append(np.array([np.nan]))

        # Else if we have at least 10 points (required for train-test split), run ElasticNetCV.
        if len(cur_dlogydt) >= 10:

            X_train, X_val, y_train, y_val = train_test_split(
                cur_gmeans, cur_dlogydt, train_size=0.8, random_state=0
            )

            enet = linear_model.ElasticNetCV(
                l1_ratio=0.5,
                eps=1e-3,
                cv=5,
                fit_intercept=True,
                max_iter=10 ** 7,
            )

            reg = enet.fit(X_train, y_train)
            r2 = enet.score(X_val, y_val)

            regs.append(reg)
            intercepts.append(reg.intercept_)
            slopes.append(reg.coef_)

        # With sample points between 1 and 10, run a simple linear regression.
        else:
            # reg = linear_model.LinearRegression().fit(cur_gmeans, cur_dlogydt)

            # regs.append(reg)
            # intercepts.append(reg.intercept_)
            # slopes.append(reg.coef_)

            regs.append(np.nan)
            slopes.append(np.repeat(np.nan, n_species))
            intercepts.append(np.array([np.nan]))

    # make em arrays
    slopes = np.vstack(slopes)
    intercepts = np.vstack(intercepts)

    inferred = (slopes, intercepts)

    if debug:
        return (inferred, info)
    else:
        return inferred


def run_mkspikeseq(X, y, progressbar=False, zellner=False, draws=5000, tune=1000):
    import pymc3 as pm

    """regresses X on y using MKSpikeSeq, returns trace"""

    # NORMALIZE X
    X = (X - np.mean(X)) / np.std(X)

    # print(X.shape)

    # Set up priors for model
    if zellner:
        Sigma_taxa = 0.5 * np.matmul(X.T, X)
        Sigma_taxa += np.diag(np.diag(Sigma_taxa))
        Sigma_taxa = np.linalg.inv(Sigma_taxa)

    else:
        Sigma_taxa = np.eye(X.shape[1])  # just use identity matrix
    # Sigma_taxa = np.identity(len(Sigma_taxa)) # alternatively use identity matrix (results same)

    # Sigma_drugs = 0.5 * np.matmul(X_drugs.T.values, X_drugs.values)
    # Sigma_drugs += np.diag(np.diag(Sigma_drugs))
    # Sigma_drugs = np.linalg.inv(Sigma_drugs)
    # Sigma_drugs = np.identity(len(Sigma_drugs)) # alternatively use identity matrix (results same)

    # For calculating growth rate initialisation
    X_growth = sm.add_constant(X)  # empty row of 1's for intercept
    # Y = my_all_Y[focal_species].copy()
    # r_model = sm.OLS(Y, X_growth)
    r_model = sm.OLS(y, X_growth)

    results = r_model.fit()
    init_r = results.params[0]  # First param will be the INTERCEPT
    if init_r < 0:
        init_r = 0

    init_r_std = np.std(results.params)  # inflate initial prior for intercept

    with pm.Model() as model:

        xi_taxa = pm.Bernoulli("xi_taxa", 0.5, shape=X.shape[1])
        tau_taxa = pm.HalfCauchy("tau_taxa", 1)
        beta_taxa = pm.MvNormal("beta_taxa", 0, tau_taxa * Sigma_taxa, shape=X.shape[1])
        mean_taxa = pm.math.dot(X, xi_taxa * beta_taxa)

        # xi_drugs = pm.Bernoulli("xi_drugs", 0.5, shape=X.shape[1])
        # tau_drugs = pm.HalfCauchy("tau_drugs", 1)
        # beta_drugs = pm.MvNormal(
        #     "beta_drugs", 0, tau_drugs * Sigma_drugs, shape=X.shape[1]
        # )
        # mean_drugs = pm.math.dot(X_drugs, xi_drugs * beta_drugs)

        my_sigma = pm.HalfNormal("my_sigma", 10)
        intercp = pm.Bound(pm.Normal, lower=0.0)("intercp", mu=1.0, tau=(init_r_std ** 2) * 1e2)

        # my_var = pm.Normal("my_var", mean_drugs + mean_taxa + intercp, my_sigma, observed=y)
        my_var = pm.Normal("my_var", mean_taxa + intercp, my_sigma, observed=y)

        trace = pm.sample(
            # draws=15000,
            # tune=3000,
            # draws=5000,
            # tune=1000,
            draws=draws,
            tune=tune,
            init="adapt_diag",
            cores=-1,
            return_inferencedata=True,
            progressbar=progressbar,
        )

    return trace


def infer_mkspikeseq_by_did(
    did, debug=False, progressbar=False, save_trace=True, draws=5000, tune=1000
):

    # Zero cutoff factor
    cutoff = 0.01 / 19.161194626965624

    df_geom, df_dlogydt, df_nzmask, n_species = prepare_data_for_inference(did)

    ####### BEGIN THE INFERENCE!!!!! #######
    regs = []
    intercepts = []
    slopes = []

    # # For debugging if needed
    # info = dict(dlogydts=[], masks=[], gmeans=[], species=[], shapes=[])

    # Begin inference for each and every focal_species
    for focal_species in range(n_species):

        # Get the y to be predicted
        cur_dlogydt = np.concatenate(df_dlogydt.loc[focal_species].values)
        cur_mask = np.concatenate(df_nzmask.loc[focal_species].values)  # based on valid y's

        # Get the X to predict, only take valid intervals
        cur_gmeans = np.array(
            [np.concatenate(df_geom.loc[i, :].values) for i in range(n_species)]
        ).T
        cur_gmeans = cur_gmeans[cur_mask, :].copy()

        # If focal_species has no intervals, return NaNs for inferred.
        if len(cur_dlogydt) <= 1:
            regs.append(np.nan)
            slopes.append(np.repeat(np.nan, n_species))
            intercepts.append(np.array([np.nan]))

        # Otherwise, regress.
        else:
            try:
                trace = run_mkspikeseq(
                    cur_gmeans, cur_dlogydt, progressbar=progressbar, draws=draws, tune=tune
                )

                cur_slopes = (
                    trace["posterior"]["beta_taxa"].values.reshape(-1, n_species).mean(axis=0)
                )
                cur_intercept = trace["posterior"]["intercp"].values.mean()

            except ValueError:
                # return ("broken", did, info)
                reg = ["broken"]

            # regs.append(trace)
            intercepts.append(cur_intercept)
            slopes.append(cur_slopes)
            regs.append(trace)

    slopes = np.vstack(slopes)
    intercepts = np.vstack(intercepts)

    # set values below the cutoff to zero
    cutoff_slopes = pd.DataFrame(slopes)[~pd.DataFrame(np.abs(slopes) < cutoff)].fillna(0).values

    if save_trace:

        try:
            os.mkdir(
                os.path.join(
                    mu.GLOBALS.MTIST_DATASET_DIR,
                    f"{INFERENCE_DEFAULTS.INFERENCE_PREFIX}inference_result",
                )
            )
        except Exception as e:
            print(e)

        try:

            import pickle

            def save_pickle(thing, fn):
                with open("{}.pickle".format(fn), "wb") as handle:
                    pickle.dump(thing, handle, protocol=3)

            save_pickle(
                regs,
                os.path.join(
                    mu.GLOBALS.MTIST_DATASET_DIR,
                    f"{INFERENCE_DEFAULTS.INFERENCE_PREFIX}inference_result",
                    f"regs_{did}",
                ),
            )

            save_pickle(
                slopes,
                os.path.join(
                    mu.GLOBALS.MTIST_DATASET_DIR,
                    f"{INFERENCE_DEFAULTS.INFERENCE_PREFIX}inference_result",
                    f"slopes_{did}",
                ),
            )

            save_pickle(
                intercepts,
                os.path.join(
                    mu.GLOBALS.MTIST_DATASET_DIR,
                    f"{INFERENCE_DEFAULTS.INFERENCE_PREFIX}inference_result",
                    f"intercepts_{did}",
                ),
            )
        except Exception as e:
            print(e)

        # for cur_trace_number, trace in enumerate(regs):
        #     trace["posterior"].to_netcdf(
        #         os.path.join(
        #             mu.GLOBALS.MTIST_DATASET_DIR,
        #             f"{INFERENCE_DEFAULTS.INFERENCE_PREFIX}inference_result",
        #             f"{INFERENCE_DEFAULTS.INFERENCE_PREFIX}trace_{cur_trace_number}_for_{did}.nc",
        #         ),
        #     )

        #     trace["sample_stats"].to_netcdf(
        #         os.path.join(
        #             mu.GLOBALS.MTIST_DATASET_DIR,
        #             f"{INFERENCE_DEFAULTS.INFERENCE_PREFIX}inference_result",
        #             f"{INFERENCE_DEFAULTS.INFERENCE_PREFIX}sample_stats_{cur_trace_number}_for_{did}.nc",
        #         ),
        #     )

        # np.savetxt(
        #     os.path.join(
        #         mu.GLOBALS.MTIST_DATASET_DIR,
        #         f"{INFERENCE_DEFAULTS.INFERENCE_PREFIX}inference_result",
        #         f"{INFERENCE_DEFAULTS.INFERENCE_PREFIX}inferred_for_{did}.csv",
        #     ),
        #     slopes,
        #     delimiter=",",
        # )

        # np.savetxt(
        #     os.path.join(
        #         mu.GLOBALS.MTIST_DATASET_DIR,
        #         f"{INFERENCE_DEFAULTS.INFERENCE_PREFIX}inference_result",
        #         f"{INFERENCE_DEFAULTS.INFERENCE_PREFIX}inferred_for_{did}_cutoff.csv",
        #     ),
        #     cutoff_slopes,
        #     delimiter=",",
        # )

        # np.savetxt(
        #     os.path.join(
        #         mu.GLOBALS.MTIST_DATASET_DIR,
        #         f"{INFERENCE_DEFAULTS.INFERENCE_PREFIX}inference_result",
        #         f"{INFERENCE_DEFAULTS.INFERENCE_PREFIX}growth_rates_for_{did}.csv",
        #     ),
        #     intercepts,
        #     delimiter=",",
        # )

    if debug:
        return (slopes, intercepts, regs)
    else:
        return (slopes, intercepts)


def calculate_es_score(true_aij, inferred_aij) -> float:
    """GRANT'S edited version to calculate ED score

    Calculate the ecological direction (EDₙ) score (n := number of species in ecosystem).

    Parameters
    ===============
    truth: ndarray(axis0=species_names, axis1=species_names), the ecosystem coefficient matrix used to generate data
    inferred: ndarray(axis0=species_names, axis1=species_names), the inferred ecosystem coefficient matrix
    Returns
    ===============
    ES_score: float
    """

    truth = pd.DataFrame(true_aij).copy()
    inferred = pd.DataFrame(inferred_aij).copy()

    if truth.shape != inferred.shape:
        raise ValueError("truth and inferred must be the same shape")

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


def infer_and_score_all(save_inference=True, save_scores=True):
    """returns df_es_scores, inferred_aijs"""

    # Load meta and gts
    meta = pd.read_csv(os.path.join(mu.GLOBALS.MTIST_DATASET_DIR, "mtist_metadata.csv")).set_index(
        "did"
    )
    aijs, _ = mu.load_ground_truths(mu.GLOBALS.GT_DIR)

    # Begin inference

    n_datasets = mu.calculate_n_datasets()
    fns = [
        os.path.join(mu.GLOBALS.MTIST_DATASET_DIR, f"dataset_{i}.csv") for i in range(n_datasets)
    ]

    # th = INFERENCE_DEFAULTS.inference_threshold  # for the floored_scores
    raw_scores = {}
    # floored_scores = {}
    inferred_aijs = {}

    for fn in fns:

        # Complete the inference
        did = int(fn.split(".csv")[0].split("dataset_")[-1])
        inferred_aij, _ = INFERENCE_DEFAULTS.INFERENCE_FUNCTION(did)

        # Obtain gt used in the dataset
        gt_used = meta.loc[did, "ground_truth"]
        true_aij = aijs[gt_used]

        # Calculate raw ES score
        es_score = calculate_es_score(true_aij, inferred_aij)

        # Calculate floored ES score
        # floored_inferred_aij = inferred_aij.copy()  # copy aij
        # mask = np.abs(floored_inferred_aij) < th  # determine where to floor
        # floored_inferred_aij[mask] = 0  # floor below the th
        # es_score_floored = calculate_es_score(true_aij, floored_inferred_aij)

        # Save the scores
        raw_scores[did] = es_score
        # floored_scores[did] = es_score_floored
        inferred_aijs[did] = inferred_aij.copy()

    df_es_scores = pd.DataFrame(
        [
            raw_scores,
            # floored_scores
        ],
        index=[
            "raw",
            # "floored"
        ],
    ).T.sort_index()

    if save_inference:

        try:
            os.mkdir(
                os.path.join(
                    mu.GLOBALS.MTIST_DATASET_DIR,
                    f"{INFERENCE_DEFAULTS.INFERENCE_PREFIX}inference_result",
                )
            )
        except Exception as e:
            print(e)

        for key in inferred_aijs.keys():
            did = key
            np.savetxt(
                os.path.join(
                    mu.GLOBALS.MTIST_DATASET_DIR,
                    f"{INFERENCE_DEFAULTS.INFERENCE_PREFIX}inference_result",
                    f"{INFERENCE_DEFAULTS.INFERENCE_PREFIX}inferred_aij_{did}.csv",
                ),
                inferred_aijs[key],
                delimiter=",",
            )

    if save_scores:
        df_es_scores.to_csv(
            os.path.join(
                mu.GLOBALS.MTIST_DATASET_DIR,
                f"{INFERENCE_DEFAULTS.INFERENCE_PREFIX}inference_result",
                f"{INFERENCE_DEFAULTS.INFERENCE_PREFIX}es_scores.csv",
            )
        )

    return (df_es_scores, inferred_aijs)


def infer_and_save_portion(dids, save_inference=True, save_scores=True):
    """returns df_es_scores, inferred_aijs"""

    # Load meta and gts
    meta = pd.read_csv(os.path.join(mu.GLOBALS.MTIST_DATASET_DIR, "mtist_metadata.csv")).set_index(
        "did"
    )
    aijs, _ = mu.load_ground_truths(mu.GLOBALS.GT_DIR)

    # Begin inference

    fns = [os.path.join(mu.GLOBALS.MTIST_DATASET_DIR, f"dataset_{i}.csv") for i in dids]

    # th = INFERENCE_DEFAULTS.inference_threshold  # for the floored_scores

    raw_scores = {}
    # floored_scores = {}
    inferred_aijs = {}

    for fn in fns:

        # Complete the inference
        did = int(fn.split(".csv")[0].split("dataset_")[-1])
        inferred_aij, _ = INFERENCE_DEFAULTS.INFERENCE_FUNCTION(did)

        # Obtain gt used in the dataset
        gt_used = meta.loc[did, "ground_truth"]
        true_aij = aijs[gt_used]

        # Calculate raw ES score
        es_score = calculate_es_score(true_aij, inferred_aij)

        # Calculate floored ES score
        # floored_inferred_aij = inferred_aij.copy()  # copy aij
        # mask = np.abs(floored_inferred_aij) < th  # determine where to floor
        # floored_inferred_aij[mask] = 0  # floor below the th
        # es_score_floored = calculate_es_score(true_aij, floored_inferred_aij)

        # Save the scores
        raw_scores[did] = es_score
        # floored_scores[did] = es_score_floored
        inferred_aijs[did] = inferred_aij.copy()

        # test saving per did

    df_es_scores = pd.DataFrame(
        [
            raw_scores,
            #  floored_scores
        ],
        index=[
            "raw",
            #   "floored"
        ],
    ).T.sort_index()

    if save_inference:

        try:
            os.mkdir(
                os.path.join(
                    mu.GLOBALS.MTIST_DATASET_DIR,
                    f"{INFERENCE_DEFAULTS.INFERENCE_PREFIX}inference_result",
                )
            )
        except Exception as e:
            print(e)

        for key in inferred_aijs.keys():
            did = key
            np.savetxt(
                os.path.join(
                    mu.GLOBALS.MTIST_DATASET_DIR,
                    f"{INFERENCE_DEFAULTS.INFERENCE_PREFIX}inference_result",
                    f"{INFERENCE_DEFAULTS.INFERENCE_PREFIX}inferred_aij_{did}.csv",
                ),
                inferred_aijs[key],
                delimiter=",",
            )

    # Need to engineer an "appending" method for this

    # if save_scores:
    #     df_es_scores.to_csv(
    #         os.path.join(
    #             mu.GLOBALS.MTIST_DATASET_DIR,
    #             f"{INFERENCE_DEFAULTS.INFERENCE_PREFIX}inference_result",
    #             f"{INFERENCE_DEFAULTS.INFERENCE_PREFIX}es_scores.csv",
    #         )
    #     )

    return (df_es_scores, inferred_aijs)


class INFERENCE_DEFAULTS:

    # Set INFERENCE_FUNCTION to a handle that takes did and spits out
    # an inferred Aij

    # If changing the inference function and wish to save the result,
    # change the inference prefix (end it in like a _ or something)
    INFERENCE_FUNCTION = infer_from_did
    INFERENCE_PREFIX = ""

    # inference_threshold = 1 / 3
