#!/usr/bin/env python
# coding: utf-8

# Author: Joao Valeriano
#
# Plots of ES score curves against number of timeseries combined
# in the dataset used for inference, varying the number of timepoints
# per timeseries and the percentile of stronger interactions to be
# considered for evaluating the ES score.
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


# In[28]:

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
save_path = f"figures_ES_vs_dset_params/{n_sp}_sp"
if save_path.split("/")[0] not in os.listdir():
    os.mkdir(save_path.split("/")[0])
    os.mkdir(save_path)

# Interaction strength percentiles to consider for analysis
strong_interaction_percentiles = [0, 50, 90, 95]

infer_methods = ["default", "ridge_CV", "lasso_CV", "elasticnet_CV"]

for infer_method in infer_methods:
    es_scores = pd.read_csv(f"{dataset_path}/{infer_method}_inference_result/{infer_method}_es_scores.csv", index_col=0)
    
    infer_files = [i for i in os.listdir(f"{dataset_path}/{infer_method}_inference_result") if "inferred" in i]
    infer_files = natural_sort(infer_files)
    infer_files = np.array(infer_files)

    # Path for saving figures, create if nonexisting
    figure_path = f"{save_path}/{infer_method}"
    if infer_method not in os.listdir(save_path):
        os.mkdir(figure_path)

    
    for strong_interaction_percentile in strong_interaction_percentiles:
        es_gt_perc = np.zeros((n_nts, n_ntp, n_ss, int(len(dset_params)/n_nts/n_ntp/n_ss/2)))
        es_inf_perc = np.zeros((n_nts, n_ntp, n_ss, int(len(dset_params)/n_nts/n_ntp/n_ss/2)))
        
        for i, n_ts in tqdm(enumerate(n_ts_list)):
            for j, n_tp in enumerate(n_tp_list):
                for k, ss in enumerate(ss_list):
                    params = np.array([0.01, n_ts, n_tp, ss], dtype="object") # noise, n_timeseries, n_timepoints, sampling_scheme
                    
                    for m, did in enumerate(dset_params[(dset_params[:,[0,1,2,4]]==params).all(1),3]):
                        p = pd.read_csv(f"{dataset_path}/{infer_method}_inference_result/{infer_files[did]}", header=None).values        
                        a_ = np.loadtxt(f"{gt_path}/interaction_coefficients/{metadata['ground_truth'].iloc[did].replace('gt', 'aij')}.csv", delimiter=",")
            
                        a = a_.copy()
                        a[np.abs(a_)<np.percentile(np.abs(a_), strong_interaction_percentile)] = 0
                        es_gt_perc[i,j,k,m] = calculate_es_score(a, p)
            
                        a = a_.copy()
                        a[np.abs(p)<np.percentile(np.abs(p), strong_interaction_percentile)] = 0
                        es_inf_perc[i,j,k,m] = calculate_es_score(a, p)
        
        
        # Plot parameters
        plt.rcParams.update({
            "text.usetex": True,
            "font.family": "Times New Roman"
        })
        
        # Names of sampling schemes for plot labels
        sampling_names = ["Even", "Random", "Sequential"]
        
        
        # Actual plotting below
        
        plt.subplots(2, 2, figsize=(16,10))
        plt.subplots_adjust(hspace=0.4)
        for i in range(3):
            plt.subplot(2, 2, i+1)
            for j, n_tp in enumerate(n_tp_list):
                plt.plot(n_ts_list, es_gt_perc[:,j,i].mean(-1), marker="o", ms=7, lw=2, label=f"{n_tp:.0f} time points")
            plt.text(0.02, 0.95, f"({97+i:c})", va="top", ha="left", fontsize=30, transform=plt.gca().transAxes)
            plt.axhline(1, lw=2, ls="--", color="gray")
            plt.xticks(fontsize=16)
            plt.yticks(fontsize=16)
            plt.xlabel("Number of Time Series", fontsize=20)
            plt.ylabel("ES Score", fontsize=20)
            plt.ylim(0.5,1.02)
            if i==0:
                plt.legend(loc="lower right", fontsize=16)
            plt.title(f"{sampling_names[i]} Sampling", fontsize=24)
        
        plt.subplot(2, 2, 4)
        for i in range(3):
            ntp15 = np.random.choice(es_gt_perc[-1,-1,i], 100000)
            ntp5 = np.random.choice(es_gt_perc[-1,0,i], 100000)
            
            bplot = plt.boxplot((ntp15-ntp5)/ntp5, positions=[i], whis=[0,100], widths=0.5, 
                                patch_artist=True, showmeans=True, meanline=True, 
                                medianprops=dict(ls=""), meanprops=dict(color="k", lw=2), 
                                boxprops=dict(lw=2), flierprops=dict(lw=2), 
                                whiskerprops=dict(lw=2), capprops=dict(lw=2))
            for patch in bplot["boxes"]:
                patch.set_facecolor(f"C{i}")
            plt.axhline(np.mean((ntp15-ntp5)/ntp5), ls="--", color=f"C{i}")
        
        plt.text(0.02, 0.98, "(d)", va="top", ha="left", fontsize=30, transform=plt.gca().transAxes)
        plt.ylabel("$\Delta$ES Score", fontsize=20)
        plt.yticks(fontsize=16)
        plt.ylim(-0.3, 0.85)
        plt.xlabel("Sampling Scheme", fontsize=20)
        plt.xticks(range(3), sampling_names, fontsize=16)
        plt.xlim(-0.5, 2.5)
        
        # plt.savefig(f"{figure_path}/ES_score_gt_perc_{strong_interaction_percentile}%_vs_dset_params_{n_sp}_sp_{infer_method}.svg", format="svg", bbox_inches="tight")
        plt.savefig(f"{figure_path}/ES_score_gt_perc_{strong_interaction_percentile}%_vs_dset_params_{n_sp}_sp_{infer_method}.png", format="png", bbox_inches="tight")
        # plt.show()
        plt.close()
        
        plt.subplots(2, 2, figsize=(16,10))
        plt.subplots_adjust(hspace=0.4)
        for i in range(3):
            plt.subplot(2, 2, i+1)
            for j, n_tp in enumerate(n_tp_list):
                plt.plot(n_ts_list, es_inf_perc[:,j,i].mean(-1), marker="o", ms=7, lw=2, label=f"{n_tp:.0f} time points")
            plt.text(0.02, 0.95, f"({97+i:c})", va="top", ha="left", fontsize=30, transform=plt.gca().transAxes)
            plt.axhline(1, lw=2, ls="--", color="gray")
            plt.xticks(fontsize=16)
            plt.yticks(fontsize=16)
            plt.xlabel("Number of Time Series", fontsize=20)
            plt.ylabel("ES Score", fontsize=20)
            plt.ylim(0.5,1.02)
            if i==0:
                plt.legend(loc="lower right", fontsize=16)
            plt.title(f"{sampling_names[i]} Sampling", fontsize=24)
        
        plt.subplot(2, 2, 4)
        for i in range(3):
            ntp15 = np.random.choice(es_gt_perc[-1,-1,i], 100000)
            ntp5 = np.random.choice(es_gt_perc[-1,0,i], 100000)
            
            bplot = plt.boxplot((ntp15-ntp5)/ntp5, positions=[i], whis=[0,100], widths=0.5, 
                                patch_artist=True, showmeans=True, meanline=True, 
                                medianprops=dict(ls=""), meanprops=dict(color="k", lw=2), 
                                boxprops=dict(lw=2), flierprops=dict(lw=2), 
                                whiskerprops=dict(lw=2), capprops=dict(lw=2))
            for patch in bplot["boxes"]:
                patch.set_facecolor(f"C{i}")
            plt.axhline(np.mean((ntp15-ntp5)/ntp5), ls="--", color=f"C{i}")
        
        plt.text(0.02, 0.98, "(d)", va="top", ha="left", fontsize=30, transform=plt.gca().transAxes)
        plt.ylabel("$\Delta$ES Score", fontsize=20)
        plt.yticks(fontsize=16)
        plt.ylim(-0.3, 0.85)
        plt.xlabel("Sampling Scheme", fontsize=20)
        plt.xticks(range(3), sampling_names, fontsize=16)
        plt.xlim(-0.5, 2.5)
        
        # plt.savefig(f"{figure_path}/ES_score_inf_perc_{strong_interaction_percentile}%_vs_dset_params_{n_sp}_sp_{infer_method}.svg", format="svg", bbox_inches="tight")
        plt.savefig(f"{figure_path}/ES_score_inf_perc_{strong_interaction_percentile}%_vs_dset_params_{n_sp}_sp_{infer_method}.png", format="png", bbox_inches="tight")
        # plt.show()
        plt.close()