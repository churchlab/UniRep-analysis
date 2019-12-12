#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 19 10:47:00 2018
@author: gr

Trains in-sample regression models for all datasets, all subsets

"""
import numpy as np
import pandas as pd
from sklearn.externals import joblib
import os
from sklearn.linear_model import LassoLarsCV, LogisticRegressionCV
from sklearn.metrics import mean_squared_error, accuracy_score, r2_score, explained_variance_score

import sys
sys.path.append('../')

from common_v2.validation_tools import subsets, regr_datasets, pearson, reps, none_model_reps, metrics, tvt_modifier_baseline_reps, tvt_modifier_return_mean
import common_v2.validation_tools as validation_tools

from sklearn.model_selection import KFold

path_to_pieces = f"../../data/pieces_new/"
run_type='test'

to_train = True # whether to train or to inference on trained models in models dir

metrics = {
'mse':mean_squared_error,
'r2':r2_score,
'exp_var':explained_variance_score,
'pearson_r':pearson}

for d in regr_datasets:
    print(d, "\n")
    ss = subsets[d]
    preload_dataset = False
    if len(subsets[d]) > 1:

        preload_dataset = True

        if d != 'rocklin_ssm2_nat_eng': # no need to do overall rocklin regression two times
            ss = list(ss)
            ss.append('full')

        preloaded_dataset = pd.read_pickle(
                os.path.join(path_to_pieces,f"{'__'.join([d,'full'])}.pkl")
            )
        print('preloaded dataset')
    for rep in reps:
        print(rep)

        for s in ss:
            print("        ",s)
            name = f"{'__'.join([d,s])}.pkl"
            path = os.path.join(path_to_pieces,name)
            dataset_filetype='pkl'

            if preload_dataset:
                dataset_filetype = 'loaded_full_dataset'
                path = preloaded_dataset


            train, validate, test = validation_tools.get_tvt(path,
                                                         d,
                                                         s,
                                                         rep, dataset_filetype=dataset_filetype, verbose=False,
                                                         modifiers=[tvt_modifier_baseline_reps,
                                                                    tvt_modifier_return_mean]
                        )
            if rep in none_model_reps:
                predictions, model, results = validation_tools.make_predictions(
                        train, validate, test,
                        metrics,
                        None,
                        run_type=run_type
                    )
            else:

                model_fname = f"./models/{d}__{s}__{rep}__model.pkl"

                if to_train == True:
            
                    kfold = KFold(n_splits=10, random_state=42, shuffle=True)
     
                    model_to_pass = LassoLarsCV(
                                            fit_intercept = True,
                                            normalize = True,
                                            n_jobs=-1,
                                            max_n_alphas=6000,
                                            cv=kfold
                                        )
                else:
                    model_to_pass = joblib.load(model_fname)

                predictions, model, results =validation_tools.make_predictions(
                            train, validate, test,
                            metrics,
                            model=model_to_pass,
                            run_type=run_type,
                            to_train=to_train
                        )
                if to_train == True:
                    joblib.dump(model, model_fname)

            np.save(f"./predictions/{d}__{s}__{rep}__{run_type}__predictions.npy", predictions)
            results.to_csv(f"./results/{d}__{s}__{rep}__{run_type}__regression_results.csv")

