#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  7 15:56:45 2018

@author: gr
"""

import os
import pandas as pd
import sys
import os
from subprocess import call

NAMES = {
    'arnold_T50': 'Cytochrome P450 Thermostability Yang 2018',
    'arnold_absorption': 'Bacterial Rhodopsin Peak Absorption Wavelength Task; Yang 2018',
    'arnold_enantioselectivity': 'Epoxide Hydrolase Enantioselectivity Task; Yang 2018',
    'arnold_localization': 'Channelrhodopsin Plasma Membrane Localization Task; Yang 2018',
    'fowler': 'Missense Variant Effect Prediction Tasks; Gray 2018',
    'homstrad': 'Protein Family Prediction from HOMSTRAD Database; Mizuguchi 1998',
    'leuenberger': 'Thermostability Prediction Across Organisms Task; Leuenberger 2017',
    'oxbench': "OXBENCH Database Reference Alignment Clustering Task; Raghava 2003",
    'rocklin_ssm2': "Natural Protein Domain Thermostability Prediction Task; Rocklin 2017",
    'rocklin_all_rds': "Designed Protein Domain Thermostability Prediction Task; Rocklin 2017",
    'RABmark_sup':'',
    'RABmark_twi':'',
    'solubility':'Protein Solubility Prediction Task in Yeast and E. Coli; Uemura 2018, Niwa 2009',
    'liao': 'SCOP 1.53 Family Benchmark; Liao and Noble 2003',
    'handstad_folds': 'SCOP 1.67 Superfamily Benchmark; Håndstad 2007',
    'handstad_superfamily': 'SCOP 1.67 Folds Benchmark; Håndstad 2007',

    "RGN": "Recurrent Geometric Network Representation; AlQuraishi 2018",
    "64_avg_hidden": "64-unit mLSTM Representation - Average Hidden State Representation",
    "64_final_hidden": "64-unit mLSTM Representation - Final Hidden State Representation",
    "64_final_cell": "64-unit mLSTM Representation - Final Cell State Representation",
    "256_avg_hidden": "256-unit mLSTM Representation - Average Hidden State Representation",
    "256_final_hidden": "256-unit mLSTM Representation - Final Hidden State Representation",
    "256_final_cell": "256-unit mLSTM Representation - Final Cell State Representation",
    "avg_hidden": "1900-unit mLSTM Representation - Average Hidden State Representation",
    "final_hidden": "1900-unit mLSTM Representation - Final Hidden State Representation",
    "final_cell": "1900-unit mLSTM Representation - Final Cell State Representation",
    "arnold_original_3_7": "Top4 Performing Embedding #1, Yang 2018",
    "arnold_scrambled_3_5": "Top4 Performing Embedding #2, Yang 2018",
    "arnold_random_3_7": "Top4 Performing Embedding #3, Yang 2018",
    "arnold_uniform_4_1": "Top4 Performing Embedding #4, Yang 2018",
    "simple_freq_plus": "Aminoacid Frequency and Predicted Biophysical Parameters Baseline Representation",
    "simple_freq_and_len": "Aminoacid Frequency and Protein Length Baseline Representation",
    "tfidf_2grams": "2-gram Baseline Representation with TF-IDF weighting",
    "2grams": "2-gram Baseline Representation",
    "tfidf_3grams": "3-gram Baseline Representation with TF-IDF weighting",
    "3grams": "3-gram Baseline Representation",
    "bidir_lstm": "Bidirectional LSTM Benchmark",

    "mse": "Mean Squared Error",
    "exp_var": "Explained Variance",
    "r2": "R^2 Coefficient of Determination",
    'transfer_ratio_avg': 'Average Transfer Ratio \n in Hold-One-Out Task',
    'indomain_ratio_avg': 'Average In-domain Ratio \n in Hold-One-Out Task',
     'perf_ratio': 'Performance Ratio: In/Out-do-domain \n in Hold-One-Out Task',
    'mean':'Mean Rep'
}
import sys
sys.path.append("../")

from common_v2.validation_tools import regr_datasets, subsets, metrics, reps
import common_v2.validation_tools

import numpy as np

import random

random.seed(42)
np.random.seed(42)

path_to_pieces = f"../../data/pieces_new/"

n_resamp = 30

std_results = pd.DataFrame(columns=metrics.keys())

print(reps)

for run_type in {'validate','test'}:

    for d in regr_datasets:
        ss = list(subsets[d])
        if len(subsets[d]) > 1:
            ss = ss + ['full']

        for s in  ss:
            print(d,s)
            name = f"{'__'.join([d,s])}.pkl"
            path = os.path.join(path_to_pieces,name)
            train, validate, test = common_v2.validation_tools.get_tvt(path,
                                                             d,
                                                             s,
                                                             '64_avg_hidden', dataset_filetype='pkl', verbose=False)

            if run_type == 'test':
                validate = test.copy()

            for rep in reps:
                print(rep)

                res_df_for_std = []
                predictions = np.load(f"./predictions/{d}__{s}__{rep}__{run_type}__predictions.npy")

                for i in range(n_resamp):

                    ind = validate.reset_index().sample(frac=1.0/2, random_state=i).index

                    results = pd.Series()

                    for metric_name in metrics.keys():

                        results.loc[metric_name] = metrics[metric_name](
                                   validate['target'].iloc[ind],
                                   predictions[ind])

                    res_df_for_std.append(results)

                std_results.loc[f"{d}__{s}__{rep}__{run_type}"] = pd.concat(res_df_for_std, axis=1).std(axis=1)

std_results.to_csv("std_results_val_resamp.csv")

