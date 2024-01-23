#!/usr/bin/env python
# coding: utf-8

# Author: Joao Valeriano
#
# Plots of ES score curves against interaction strength, varying
# the number of timeseries included in the used dataset.
# Plots are made with respect to both ground-truth and inferred
# interaction strengths.
# ----------------------------------------------------------------

#%%

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import re
from mtist.infer_mtist import calculate_es_score
from tqdm import tqdm # Progressbars

#%%

# Sort files in natural order (increasing multi-digit integers)
def natural_sort(l):
    def convert(text): return int(text) if text.isdigit() else text.lower()

    def alphanum_key(key): return [convert(c)
                                   for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)


#%%


dataset_path = "../mtist_datasets_extended"
gt_path = "../ground_truths_extended"

dataset_files = [i for i in os.listdir(dataset_path) if "dataset_" in i and ".csv" in i]
dataset_files = natural_sort(dataset_files)
dataset_files = np.array(dataset_files)

metadata = pd.read_csv(f"{dataset_path}/mtist_metadata.csv")


#%%

# Consider only files with the chosen number of species (n_sp)
n_sp = 10
n_sp_files = np.array([pd.read_csv(f"{dataset_path}/{file}")["n_species"][0] for file in dataset_files])
dataset_files_nsp = dataset_files[n_sp_files==n_sp]


#%%

# Collect parameters of each dataset
dset_params = []

for i in tqdm(range(len(dataset_files_nsp))):
    file = dataset_files_nsp[i]
    df_file = pd.read_csv(f"{dataset_path}/{file}", index_col=0)
    
    dset_params.append(df_file[["noise", "n_timeseries", "n_timepoints", "did", "sampling_scheme"]].values[0])

dset_params = np.array(dset_params)


#%%

# Inference and plotting

# Lists with no. of timeseries (ts), no. of timepoints (tp) and sampling schemes (ss), respectively, of each dataset
n_ts_list = np.unique(dset_params[:,1])
n_tp_list = np.unique(dset_params[:,2])
ss_list = np.unique(dset_params[:,4])

print(f"n_timeseries: {n_ts_list}")
print(f"n_timepoints: {n_tp_list}")
print(f"sampling_schemes: {ss_list}")

n_nts = len(n_ts_list)
n_ntp = len(n_tp_list)
n_ss = len(ss_list)

# Path for saving results, create if needed
save_path = f"figures_ES_vs_interaction_strength/{n_sp}_sp"
if save_path.split("/")[0] not in os.listdir():
    os.mkdir(save_path.split("/")[0])
    os.mkdir(save_path)

# Choose inference methods to be considered for plotting results
infer_methods = ["default", "ridge_CV", "lasso_CV", "elasticnet_CV"]

for infer_method in infer_methods:

    # Path for saving figures, create if nonexisting
    figure_path = f"{save_path}/{infer_method}"
    
    if infer_method not in os.listdir(save_path):
        os.mkdir(figure_path)
    
    es_scores = pd.read_csv(f"{dataset_path}/{infer_method}_inference_result/{infer_method}_es_scores.csv", index_col=0)

    # Get inference results for each dataset
    infer_files = [i for i in os.listdir(f"{dataset_path}/{infer_method}_inference_result") if "inferred" in i]
    infer_files = natural_sort(infer_files)
    infer_files = np.array(infer_files)
    
    # Get maximum absolute interaction strength value for each ground truth 
    a_max = []
    for file in [i for i in os.listdir(f"{gt_path}/interaction_coefficients/") if f"{n_sp}_sp" in i]:
        a = np.loadtxt(f"{gt_path}/interaction_coefficients/{file}", delimiter=",")
        a_max.append(np.max(np.abs(a)))

    
    #---------------------------------------------------
    
    
    nl = 100 # Number of intervals

    # Arrays to store ES score against interaction strength, stratified by n_ts, n_tp and ss
    es_gt = np.zeros((n_nts, n_ntp, n_ss, nl))
    es_gt_perc = np.zeros((n_nts, n_ntp, n_ss, nl))
    es_inf_perc = np.zeros((n_nts, n_ntp, n_ss, nl))
    
    l = np.linspace(0, min(a_max), nl) # Absolute interaction value intervals, based on ground truth
                                       # min(a_max): so that every intervals has contributions from all gt's
    lp = np.linspace(0, 99, nl) # Percentile intervals

    # ES score is averaged over all dids with the same set of parameters [noise, n_ts, n_tp, ss]
    for i, n_ts in tqdm(enumerate(n_ts_list)):
        for j, n_tp in enumerate(n_tp_list):
            for k, ss in enumerate(ss_list):
                    params = np.array([0.01, n_ts, n_tp, ss], dtype="object") # [noise, n_timeseries, n_timepoitns, sampling_scheme]
                    count = 0 # variable necessary for iterative average calculation
                    
                    for did in dset_params[(dset_params[:,[0,1,2,4]]==params).all(1),3]:
                        p = pd.read_csv(f"{dataset_path}/{infer_method}_inference_result/{infer_files[did]}", header=None).values # inferred coeffs
                        a_ = np.loadtxt(f"{gt_path}/interaction_coefficients/{metadata['ground_truth'].iloc[did].replace('gt', 'aij')}.csv",
                                        delimiter=",") # gt coeffs

                        for m in range(len(l)):
                            # threshold based on raw ground-truth
                            a = a_.copy()
                            a[np.abs(a_)<l[m]] = 0 # discarding interactions weaker than a given threshold
                            es_gt[i,j,k,m] += (calculate_es_score(a, p)-es_gt[i,j,k,m])/(count+1) # iterative average calculation

                            # threshold based on percentiles of ground-truth
                            a = a_.copy()
                            a[np.abs(a_)<np.percentile(np.abs(a_), lp[m])] = 0
                            es_gt_perc[i,j,k,m] += (calculate_es_score(a, p)-es_gt_perc[i,j,k,m])/(count+1)

                            # threshold based on percentiles of inferred
                            a = a_.copy()
                            a[np.abs(p)<np.percentile(np.abs(p), lp[m])] = 0
                            es_inf_perc[i,j,k,m] += (calculate_es_score(a, p)-es_inf_perc[i,j,k,m])/(count+1)
                            
                        count += 1
    
    
    # Plot parameters
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "Times New Roman"
    })
    
    # Names of sampling schemes for plot labels
    sampling_names = ["Even", "Random", "Sequential"]
    
    
    # Actual plotting below

    # Creating plot object just for generating colorbar later
    im = plt.contourf(es_gt[0,0], levels=np.arange(0.5, n_nts+0.5), cmap="viridis", vmin=1, vmax=n_nts+1)
    plt.clf()

    for j, n_tp in enumerate(n_tp_list):
        plt.subplots(2, 2, figsize=(16,10))
        plt.subplots_adjust(hspace=0.4)
        for k, ss in enumerate(ss_list):
            plt.subplot(2, 2, k+1)
            for i, n_ts in enumerate(n_ts_list):
                plt.plot(l, es_gt[i,j,k], lw=3, color=plt.cm.viridis(np.linspace(0, 1, n_nts))[i])
        
            if k==len(ss_list)-1:
                cax = plt.gca().inset_axes([1.01, 0., 0.05, 1])
                cb = plt.colorbar(im, cax=cax, ticks=np.arange(1, n_ts+2, 4))
                cb.ax.tick_params(labelsize=14)
                cb.set_label("Number of Timeseries", fontsize=20)
            
            plt.axhline(1, c="gray", ls="--", lw=2)
            plt.ylim(0.5, 1.02)
            plt.xlim(0, min(a_max))
            plt.xticks(fontsize=16)
            plt.yticks(fontsize=16)
            plt.xlabel("$A_{ij}$ = Interaction Strength", fontsize=20)
            plt.ylabel("ES($\{A_{ij}', |A_{ij}'| > |A_{ij}|\}$)", fontsize=20)
        
            plt.title(sampling_names[k], fontsize=24)
        
        plt.subplot(2, 2, 4)
        plt.axis("off")
            
        plt.savefig(f"{figure_path}/ES_vs_int_strength_{n_sp}_sp_{infer_method}_{n_tp}_tp.png", format="png", dpi=300, bbox_inches="tight")
        # plt.savefig(f"{figure_path}/ES_vs_int_strength_{n_sp}_sp_{infer_method}_{n_tp}_tp.svg", format="svg", dpi=300, bbox_inches="tight")
        # plt.show()
        plt.close()
    
        plt.subplots(2, 2, figsize=(16,10))
        plt.subplots_adjust(hspace=0.4)
        for k, ss in enumerate(ss_list):
            plt.subplot(2, 2, k+1)
            for i, n_ts in enumerate(n_ts_list):
                plt.plot(lp, es_gt_perc[i,j,k], lw=3, color=plt.cm.viridis(np.linspace(0, 1, n_nts))[i])
        
            if k==len(ss_list)-1:
                cax = plt.gca().inset_axes([1.01, 0., 0.05, 1])
                cb = plt.colorbar(im, cax=cax, ticks=np.arange(1, n_ts+2, 4))
                cb.ax.tick_params(labelsize=14)
                cb.set_label("Number of Timeseries", fontsize=20)
            
            plt.axhline(1, c="gray", ls="--", lw=2)
            plt.ylim(0.5, 1.02)
            plt.xlim(0, 100)
            plt.xticks(fontsize=16)
            plt.yticks(fontsize=16)
            plt.xlabel("$x$ = Percentile of Interaction Strength $|A_{ij}|$", fontsize=20)
            plt.ylabel("ES($\{A_{ij}', |A_{ij}'| > \mathrm{Perc}(|A_{ij}|, x)\}$)", fontsize=20)
        
            plt.title(sampling_names[k], fontsize=24)
            
        plt.subplot(2, 2, 4)
        plt.axis("off")
            
        plt.savefig(f"{figure_path}/ES_vs_int_strength_gt_perc_{n_sp}_sp_{infer_method}_{n_tp}_tp.png", format="png", dpi=300, bbox_inches="tight")
        # plt.savefig(f"{figure_path}/ES_vs_int_strength_gt_perc_{n_sp}_sp_{infer_method}_{n_tp}_tp.svg", format="svg", dpi=300, bbox_inches="tight")
        # plt.show()
        plt.close()
    
        plt.subplots(2, 2, figsize=(16,10))
        plt.subplots_adjust(hspace=0.4)
        for k, ss in enumerate(ss_list):
            plt.subplot(2, 2, k+1)
            for i, n_ts in enumerate(n_ts_list):
                plt.plot(lp, es_inf_perc[i,j,k], lw=3, color=plt.cm.viridis(np.linspace(0, 1, n_nts))[i])
        
            if k==len(ss_list)-1:
                cax = plt.gca().inset_axes([1.01, 0., 0.05, 1])
                cb = plt.colorbar(im, cax=cax, ticks=np.arange(1, n_ts+2, 4))
                cb.ax.tick_params(labelsize=14)
                cb.set_label("Number of Timeseries", fontsize=20)
            
            plt.axhline(1, c="gray", ls="--", lw=2)
            plt.ylim(0.5, 1.02)
            plt.xlim(0, 100)
            plt.xticks(fontsize=16)
            plt.yticks(fontsize=16)
            plt.xlabel("$x$ = Percentile of Inferred Interaction Strength $|A_{ij}|$", fontsize=20)
            plt.ylabel("ES($\{A_{ij}', |A_{ij}'| > \mathrm{Perc}(|A_{ij}|, x)\}$)", fontsize=20)
        
            plt.title(sampling_names[k], fontsize=24)
        
        plt.subplot(2, 2, 4)
        plt.axis("off")
        
        plt.savefig(f"{figure_path}/ES_vs_int_strength_inf_perc_{n_sp}_sp_{infer_method}_{n_tp}_tp.png", format="png", dpi=300, bbox_inches="tight")
        # plt.savefig(f"{figure_path}/ES_vs_int_strength_inf_perc_{n_sp}_sp_{infer_method}_{n_tp}_tp.svg", format="svg", dpi=300, bbox_inches="tight")
        # plt.show()
        plt.close()

    for ss in ss_list:
        if ss not in os.listdir(figure_path):
            os.mkdir(f"{figure_path}/{ss}")
    
    for j, n_tp in enumerate(n_tp_list):
        for k, ss in enumerate(ss_list):
            im = plt.contourf(es_gt[0,0], levels=np.arange(0.5, n_nts+0.5), cmap="viridis", vmin=1, vmax=n_nts+1)
            plt.clf()
            plt.figure(figsize=(10,6))
            for i, n_ts in enumerate(n_ts_list):
                plt.plot(l, es_gt[i,j,k], lw=3, color=plt.cm.viridis(np.linspace(0, 1, n_nts))[i])
    
            cax = plt.gca().inset_axes([1.01, 0., 0.05, 1])
            cb = plt.colorbar(im, cax=cax, ticks=np.arange(1, n_ts+2, 4))
            cb.ax.tick_params(labelsize=14)
            cb.set_label("Number of Timeseries", fontsize=20)
            
            plt.axhline(1, c="gray", ls="--", lw=2)
            plt.ylim(top=1.02)
            plt.xlim(0, min(a_max))
            plt.xticks(fontsize=16)
            plt.yticks(fontsize=16)
            plt.xlabel("$A_{ij}$ = Interaction Strength", fontsize=20)
            plt.ylabel("ES($\{A_{ij}', |A_{ij}'| > |A_{ij}|\}$)", fontsize=20)
            
            plt.savefig(f"{figure_path}/{ss}/ES_vs_int_strength_{n_sp}_sp_{infer_method}_{n_tp}_tp_{ss}.png", format="png", dpi=300, bbox_inches="tight")
            # plt.savefig(f"{figure_path}/{ss}/ES_vs_int_strength_{n_sp}_sp_{infer_method}_{n_tp}_tp_{ss}.svg", format="svg", dpi=300, bbox_inches="tight")
            # plt.show()
            plt.close()

    for j, n_tp in enumerate(n_tp_list):
        for k, ss in enumerate(ss_list):
            im = plt.contourf(es_gt[0,0], levels=np.arange(0.5, n_nts+0.5), cmap="viridis", vmin=1, vmax=n_nts+1)
            plt.clf()
            plt.figure(figsize=(10,6))
            for i, n_ts in enumerate(n_ts_list):
                plt.plot(lp, es_gt_perc[i,j,k], lw=3, color=plt.cm.viridis(np.linspace(0, 1, n_nts))[i])
    
            cax = plt.gca().inset_axes([1.01, 0., 0.05, 1])
            cb = plt.colorbar(im, cax=cax, ticks=np.arange(1, n_ts+2, 4))
            cb.ax.tick_params(labelsize=14)
            cb.set_label("Number of Timeseries", fontsize=20)
            
            plt.axhline(1, c="gray", ls="--", lw=2)
            plt.ylim(top=1.02)
            plt.xlim(0, 100)
            plt.xticks(fontsize=16)
            plt.yticks(fontsize=16)
            plt.xlabel("$x$ = Percentile of Interaction Strength $|A_{ij}|$", fontsize=20)
            plt.ylabel("ES($\{A_{ij}', |A_{ij}'| > \mathrm{Perc}(|A_{ij}|, x)\}$)", fontsize=20)
            
            plt.savefig(f"{figure_path}/{ss}/ES_vs_int_strength_gt_perc_{n_sp}_sp_{infer_method}_{n_tp}_tp_{ss}.png", format="png", dpi=300, bbox_inches="tight")
            # plt.savefig(f"{figure_path}/{ss}/ES_vs_int_strength_gt_perc_{n_sp}_sp_{infer_method}_{n_tp}_tp_{ss}.svg", format="svg", dpi=300, bbox_inches="tight")
            # plt.show()
            plt.close()

    for j, n_tp in enumerate(n_tp_list):
        for k, ss in enumerate(ss_list):
            im = plt.contourf(es_gt[0,0], levels=np.arange(0.5, n_nts+0.5), cmap="viridis", vmin=1, vmax=n_nts+1)
            plt.clf()
            plt.figure(figsize=(10,6))
            for i, n_ts in enumerate(n_ts_list):
                plt.plot(lp, es_inf_perc[i,j,k], lw=3, color=plt.cm.viridis(np.linspace(0, 1, n_nts))[i])
    
            cax = plt.gca().inset_axes([1.01, 0., 0.05, 1])
            cb = plt.colorbar(im, cax=cax, ticks=np.arange(1, n_ts+2, 4))
            cb.ax.tick_params(labelsize=14)
            cb.set_label("Number of Timeseries", fontsize=20)
            
            plt.axhline(1, c="gray", ls="--", lw=2)
            plt.ylim(top=1.02)
            plt.xlim(0, 100)
            plt.xticks(fontsize=16)
            plt.yticks(fontsize=16)
            plt.xlabel("$x$ = Percentile of Inferred Interaction Strength $|A_{ij}|$", fontsize=20)
            plt.ylabel("ES($\{A_{ij}', |A_{ij}'| > \mathrm{Perc}(|A_{ij}|, x)\}$)", fontsize=20)
            
            plt.savefig(f"{figure_path}/{ss}/ES_vs_int_strength_inf_perc_{n_sp}_sp_{infer_method}_{n_tp}_tp_{ss}.png", format="png", dpi=300, bbox_inches="tight")
            # plt.savefig(f"{figure_path}/{ss}/ES_vs_int_strength_inf_perc_{n_sp}_sp_{infer_method}_{n_tp}_tp_{ss}.svg", format="svg", dpi=300, bbox_inches="tight")
            # plt.show()
            plt.close()
    
